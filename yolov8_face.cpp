#include "yolov8_face.h"

using namespace cv;
using namespace dnn;
using namespace std;

static inline float sigmoid_x(float x)
{
    return static_cast<float>(1.f / (1.f + expf(-x)));
}

YOLOv8_face::YOLOv8_face(string modelpath, float confThreshold, float nmsThreshold)
{
    this->confThreshold = confThreshold;
    this->nmsThreshold = nmsThreshold;
    this->net = cv::dnn::readNet(modelpath);

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

cv::Mat YOLOv8_face::resize_image(const cv::Mat srcimg, int *newh, int *neww, int *padh, int *padw)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = this->inpHeight;
    *neww = this->inpWidth;
    cv::Mat dstimg;
    if (this->keep_ratio && srch != srcw) {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1) {
            *newh = this->inpHeight;
            *neww = int(this->inpWidth / hw_scale);
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *padw = int((this->inpWidth - *neww) * 0.5);
            copyMakeBorder(dstimg, dstimg, 0, 0, *padw, this->inpWidth - *neww - *padw, BORDER_CONSTANT, 0);
        } else {
            *newh = (int)this->inpHeight * hw_scale;
            *neww = this->inpWidth;
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *padh = (int)(this->inpHeight - *newh) * 0.5;
            copyMakeBorder(dstimg, dstimg, *padh, this->inpHeight - *newh - *padh, 0, 0, BORDER_CONSTANT, 0);
        }
    } else {
        resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
    }
    return dstimg;
}

void YOLOv8_face::getRotatedFace(const cv::Mat &frame, cv::Mat &output, const cv::Rect &roi, const vector<Point> landmark)
{
    // Eyes
    Point d = landmark[0]-landmark[1];
    float angle = (atan2f((float)d.y, (float)d.x) * 180.f / CV_PI) - 180.f + 360.f;
    float dist = sqrtf(powf(d.y, 2)+powf(d.x, 2));

    if (angle>180.0) angle=-(360.f-angle);
    angle=avg_angle.add(angle);

    Point nose = landmark[2];
    Point center=(landmark[0]+landmark[1]+landmark[2])*0.3333;

    //printf("EYES: %f.2 deg, dist: %f.2 \n", angle, dist);

    // Mouth
    Point md = landmark[3]-landmark[4];
    //float angle = (atan2f((float)d.y, (float)d.x) * 180.f / CV_PI) - 180.f;
    float mdist = sqrtf(powf(md.y, 2)+powf(md.x, 2));

    // Rotation
    cv::RotatedRect rbbox=cv::RotatedRect(nose, frame.size(), angle);
    cv::Rect bbox=rbbox.boundingRect();

    rot=getRotationMatrix2D(center, angle, 1.0);

    rot.at<double>(0,2) += bbox.width/2.0 - nose.x;
    rot.at<double>(1,2) += bbox.height/2.0 - nose.y;

    cv::Mat dst2;
    warpAffine(frame, dst2, rot, bbox.size(), INTER_CUBIC);

    int fx,fy,wh, fhw, fhh;
    wh=std::max(roi.width, roi.height);
    fhw=wh+dist/2.0;
    fhh=wh+dist;
    fx=dst2.size().width/2.0-fhw/2.0;
    fy=dst2.size().height/2.0-fhh/2.0;

    cv::Rect froi(fx, fy, fhw, fhh);

    // Make sure we keep froi inside the frame
    froi=froi & cv::Rect(0, 0, dst2.size().width, dst2.size().height);

    output=dst2(froi);
}

cv::Mat YOLOv8_face::getFaceMat(int idx, const cv::Mat &frame)
{
    cv::Rect roi=boxes[idx];
    roi=roi & cv::Rect(0, 0, frame.size().width, frame.size().height);
    return frame(roi);
}

cv::Point2f YOLOv8_face::getNosePosition(int faceIndex)
{
    vector<Point> l=landmarks[faceIndex];

    cv::Point n=l[2];
    cv::Point2f nose;

    nose.x=((float)n.x/srcRatiow)-0.5f;
    nose.y=((float)n.y/srcRatioh)-0.5f;

    return nose;
}

void YOLOv8_face::generate_proposal(const Mat &out, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw)
{
    const int feat_h = out.size[2];
    const int feat_w = out.size[3];
    //cout << out.size[1] << "," << out.size[2] << "," << out.size[3] << endl;
    const int stride = (int)ceil((float)inpHeight / feat_h);
    const int area = feat_h * feat_w;
    float* ptr = (float*)out.data;
    float* ptr_cls = ptr + area * reg_max * 4;
    float* ptr_kp = ptr + area * (reg_max * 4 + num_class);

    for (int i = 0; i < feat_h; i++) {
        for (int j = 0; j < feat_w; j++) {
            const int index = i * feat_w + j;
            int cls_id = -1;
            float max_conf = -10000;
            for (int k = 0; k < num_class; k++) {
                float conf = ptr_cls[k*area + index];
                if (conf > max_conf) {
                    max_conf = conf;
                    cls_id = k;
                }
            }
            float box_prob = sigmoid_x(max_conf);
            if (box_prob > this->confThreshold) {
                float pred_ltrb[4];
                float* dfl_value = new float[reg_max];
                float* dfl_softmax = new float[reg_max];
                for (int k = 0; k < 4; k++) {
                    for (int n = 0; n < reg_max; n++) {
                        dfl_value[n] = ptr[(k*reg_max + n)*area + index];
                    }
                    softmax_(dfl_value, dfl_softmax, reg_max);
                    float dis = 0.f;
                    for (int n = 0; n < reg_max; n++) {
                        dis += n * dfl_softmax[n];
                    }
                    pred_ltrb[k] = dis * stride;
                }
                float cx = (j + 0.5f)*stride;
                float cy = (i + 0.5f)*stride;
                float xmin = max((cx - pred_ltrb[0] - padw)*ratiow, 0.f);
                float ymin = max((cy - pred_ltrb[1] - padh)*ratioh, 0.f);
                float xmax = min((cx + pred_ltrb[2] - padw)*ratiow, float(imgw - 1));
                float ymax = min((cy + pred_ltrb[3] - padh)*ratioh, float(imgh - 1));
                Rect box = Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin));
                boxes.push_back(box);
                confidences.push_back(box_prob);

                vector<Point> kpts(5);
                for (int k = 0; k < 5; k++) {
                    float x = ((ptr_kp[(k * 3)*area + index] * 2 + j)*stride - padw)*ratiow;
                    float y = ((ptr_kp[(k * 3 + 1)*area + index] * 2 + i)*stride - padh)*ratioh;
                    float pt_conf = sigmoid_x(ptr_kp[(k * 3 + 2)*area + index]);
                    kpts[k] = Point(int(x), int(y));
                    //cout << k << ": " << pt_conf << " - " << ptr_kp[(k * 3)*area + index] << "," << ptr_kp[(k * 3)*area + index] << endl;
                }

                landmarks.push_back(kpts);
            }
        }
    }
}

int YOLOv8_face::detect(cv::Mat &srcimg)
{
    int newh = 0, neww = 0, padh = 0, padw = 0;
    cv::Mat dst = this->resize_image(srcimg, &newh, &neww, &padh, &padw);
    cv::Mat blob;

    cv::dnn::blobFromImage(dst, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);

    vector<cv::Mat> outs;
    // this->net.enableWinograd(false);
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

    boxes.clear();
    confidences.clear();
    landmarks.clear();
    faces.clear();

    srcRatioh=srcimg.rows;
    srcRatiow=srcimg.cols;

    float rh = (float)srcimg.rows / newh;
    float rw = (float)srcimg.cols / neww;

    generate_proposal(outs[0], srcimg.rows, srcimg.cols, rh, rw, padh, padw);
    generate_proposal(outs[1], srcimg.rows, srcimg.cols, rh, rw, padh, padw);
    generate_proposal(outs[2], srcimg.rows, srcimg.cols, rh, rw, padh, padw);

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, faces);
    faceCount=faces.size();

    return faceCount;
}

int YOLOv8_face::getLargestFace()
{
    int area=96*96, largest=faceCount>0 ? 0 : -1;

    for (size_t i = 0; i < faces.size(); ++i) {
        int idx = faces[i];
        const cv::Rect box = boxes[idx];
        int a=box.width*box.height;
        if (a>area && confidences[idx]>0.60f) {
            area=a;
            largest=idx;
        }
    }

    faceArea=area/inpArea;

    return largest;
}

void YOLOv8_face::softmax_(const float *x, float *y, int length)
{
    float sum = 0;
    int i = 0;
    for (i = 0; i < length; i++) {
        y[i] = expf(x[i]);
        sum += y[i];
    }
    for (i = 0; i < length; i++) {
        y[i] /= sum;
    }
}

cv::Rect YOLOv8_face::getROI(int faceIndex)
{
    float conf=confidences[faceIndex];
    std::vector<cv::Point> landmark=landmarks[faceIndex];
    cv::Rect roi=boxes[faceIndex];

    return roi;
}

void YOLOv8_face::drawPred(cv::Mat &frame, int faceIndex)
{
    float conf=confidences[faceIndex];
    std::vector<cv::Point> landmark=landmarks[faceIndex];
    cv::Rect roi=boxes[faceIndex];

    // Rectangle of bounding box
    roi=roi & cv::Rect(0, 0, frame.size().width, frame.size().height);

    // cv::Mat iroi = frame(roi);

    theFace = frame(roi);

    //rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);
    //rectangle(frame, roi, inFocus ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);
    rectangle(frame, roi, Scalar(0, 0, 255), 2);

    //Get the label for the class name and its confidence
    std::string label = format("F:%.2f", conf);

    //Display the label at the top of the bounding box
    /*int baseLine;
Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
top = max(top, labelSize.height);
rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);*/

    putText(frame, label, Point(roi.x, roi.y-5), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
    // Eyes
    circle(frame, landmark[0], 2, Scalar(255, 0, 0), 1);
    circle(frame, landmark[1], 2, Scalar(255, 0, 0), 1);
    line(frame, landmark[0], landmark[1], Scalar(0, 255, 0));

    // Nose
    circle(frame, landmark[2], 8, Scalar(0, 255, 255), 1);

    // Mouth
    circle(frame, landmark[3], 2, Scalar(0, 255, 255), 1);
    circle(frame, landmark[4], 2, Scalar(0, 255, 255), 1);
    line(frame, landmark[3], landmark[4], Scalar(0, 255, 0));
}
