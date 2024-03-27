#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <unistd.h>

#include <libpq-fe.h>

#include "openface.hpp"
#include "moving_average.hpp"
#include "selfiesegment.hpp"
#include "focus_check.hpp"

using namespace cv;
using namespace dnn;
using namespace std;

static const string kWinName = "Face detection use OpenCV";
static const string kWinRoi = "ROI";
static const string kWinMask = "Mask";

int simulatedFocus=0;
int peakThreshold=6;

int avgc=0;
cv::Mat p;
cv::Mat pavg;

PGconn *conn;

class YOLOv8_face
{
public:
YOLOv8_face(string modelpath, float confThreshold, float nmsThreshold);
int detect(Mat& frame);
cv::Mat theFace;
cv::Rect lgbox;
double variance = 0.0;
int faces=0;

private:
Mat resize_image(Mat srcimg, int *newh, int *neww, int *padh, int *padw);
const bool keep_ratio = true;
const int inpWidth = 640;
const int inpHeight = 640;
float confThreshold;
float nmsThreshold;
const int num_class = 1;
const int reg_max = 16;
Net net;
cv::Mat rot;

vector<cv::Rect> boxes;
vector<float> confidences;
vector< vector<cv::Point>> landmarks;

void softmax_(const float* x, float* y, int length);
void generate_proposal(Mat out, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw);
void drawPred(float conf, cv::Rect &roi, Mat& frame, vector<Point> landmark);

MovingAverage avg_angle;
};

static inline float sigmoid_x(float x)
{
return static_cast<float>(1.f / (1.f + expf(-x)));
}

YOLOv8_face::YOLOv8_face(string modelpath, float confThreshold, float nmsThreshold)
{
this->confThreshold = confThreshold;
this->nmsThreshold = nmsThreshold;
this->net = readNet(modelpath);

net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

Mat YOLOv8_face::resize_image(Mat srcimg, int *newh, int *neww, int *padh, int *padw)
{
int srch = srcimg.rows, srcw = srcimg.cols;
*newh = this->inpHeight;
*neww = this->inpWidth;
Mat dstimg;
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

void YOLOv8_face::drawPred(float conf, cv::Rect &roi, Mat& frame, vector<Point> landmark)
{
// Rectangle of bounding box
//cv::Rect roi(left, top, right-left, bottom-top);
roi=roi & cv::Rect(0, 0, frame.size().width, frame.size().height);

cv::Mat iroi = frame(roi);

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

//	convertScaleAbs(lap, lapim);

//cv::Mat theface = frame(bbox);

Mat dst2;
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

theFace = dst2(froi);
imshow(kWinRoi, theFace);

//rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);
//rectangle(frame, roi, inFocus ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);
rectangle(frame, roi, Scalar(0, 0, 255), 2);

//Get the label for the class name and its confidence
string label = format("F:%.2f E: %.1f V: %.3f", conf, angle, variance);

//Display the label at the top of the bounding box
/*int baseLine;
Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
top = max(top, labelSize.height);
rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);*/

putText(frame, label, Point(roi.x, roi.y-5), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
// Eyes
circle(frame, landmark[0], 2, Scalar(255, 0, 0), 2);
circle(frame, landmark[1], 2, Scalar(255, 0, 0), 2);
line(frame, landmark[0], landmark[1], Scalar(0, 255, 0));

// Nose
circle(frame, landmark[2], 8, Scalar(0, 255, 255), 1);
circle(frame, center, 8, Scalar(128, 128, 255), 2);

// Mouth
circle(frame, landmark[3], 2, Scalar(0, 255, 255), 2);
circle(frame, landmark[4], 2, Scalar(0, 255, 255), 2);
line(frame, landmark[3], landmark[4], Scalar(0, 255, 0));
}

void YOLOv8_face::softmax_(const float* x, float* y, int length)
{
float sum = 0;
int i = 0;
for (i = 0; i < length; i++) {
	y[i] = exp(x[i]);
	sum += y[i];
}
for (i = 0; i < length; i++) {
	y[i] /= sum;
}
}

void YOLOv8_face::generate_proposal(Mat out, int imgh,int imgw, float ratioh, float ratiow, int padh, int padw)
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

int YOLOv8_face::detect(Mat& srcimg)
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

float ratioh = (float)srcimg.rows / newh;
float ratiow = (float)srcimg.cols / neww;

generate_proposal(outs[0], srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
generate_proposal(outs[1], srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
generate_proposal(outs[2], srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);

// Perform non maximum suppression to eliminate redundant overlapping boxes with
// lower confidences
vector<int> indices;
cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
faces=indices.size();

int area=96*96, largest=faces>0 ? 0 : -1;

for (size_t i = 0; i < indices.size(); ++i) {
	int idx = indices[i];
	cv::Rect box = boxes[idx];
	int a=box.width*box.height;
	if (a>area && confidences[idx]>0.60f) {
		area=a;
		largest=idx;
	}
}

if (largest>-1) {
	lgbox = boxes[largest];
	this->drawPred(confidences[largest], lgbox, srcimg, landmarks[largest]);
}

return largest;
}

void focus_peaking(cv::Mat &image)
{
cv::Mat gray, edges, er;

cv::cvtColor(image, gray, COLOR_BGR2GRAY);

cv::GaussianBlur(gray, edges, Size(7, 7), 1.5, 1.5);
cv::Canny(edges, edges, 10, 160, 3, true);

cv::cvtColor(edges, er, COLOR_GRAY2BGR);
er=er.mul(cv::Scalar(0, 0, 255), 1);

cv::bitwise_or(image, er, image, edges);
}

void detect_from_image(YOLOv8_face &face, OpenFace &of, const char *file)
{
string imgpath = file;
Mat image = imread(imgpath);
Mat scaled;
double scale = 1024.0f/image.size().width;

resize(image, scaled, Size(), scale, scale, INTER_AREA);

int f=face.detect(scaled);
printf("Faces: %d\n", f);

of.detect(face.theFace);

imshow(kWinName, scaled);

waitKey(0);
}


void dump_face(Mat vec, int faceid)
{
PGresult *res;
std::string s;
std::string e;

e << vec;

s="INSERT INTO faces (person, embedding) VALUES ("+std::to_string(faceid)+",'"+e+"');";

// std::cout << e << std::endl;

res=PQexec(conn, s.c_str());
if (PQresultStatus(res) != PGRES_COMMAND_OK) {
	fprintf(stderr, "Connection to database failed: %s", PQerrorMessage(conn));
	PQclear(res);
	// exit(2);
}
}

void detect_from_video(YOLOv8_face &face, OpenFace &of, SelfieSegment &ss, int camera, string file="")
{
bool run=true;
bool embeddings=false;
bool peaking=true;
Mat frame;
VideoCapture cap;
FocusCheck focus;

if (camera>-1) {
	cap.open(camera, CAP_V4L2);
	cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
} else {
	cap.open(file);
}

if (!cap.isOpened()) {
	printf("Failed to open video input\n");
	return;
}

TickMeter tm;

while (cap.read(frame) && run) {
	cv::Mat vec;
	cv::Mat scaled;

	tm.start();

	double scale = 1024.0f/frame.size().width;
	resize(frame, scaled, Size(), scale, scale, INTER_AREA);

	int f=face.detect(scaled);

	//printf("Faces: (%d) %d\n", f, face.faces);
	if (f>0) {
        focus.isFocused(face.theFace, true);
		vec=of.detect(face.theFace);
		if (conn && embeddings) {
			dump_face(vec, 1);
		}
		//cv::Mat s2, ssm;
		//ssm=ss.detect(scaled);
		// imshow(kWinMask, ssm);
    }
	if (peaking) {
		focus_peaking(scaled);
	}
	imshow(kWinName, scaled);

	tm.stop();

	// printf("FPS: %f\n", tm.getFPS());

	int key = waitKey(20);
	switch (key) {
	case 'q':
		run=false;
	break;
	case 's':
		if (f>0)
			of.store(vec);
	break;
	case 'e':
		embeddings=!embeddings;
	break;
    case 'p':
		peaking=!peaking;
	break;
	case 't':
		embeddings=false;
		of.train();
	break;
	case 'c':
		p+=vec;
		avgc++;
		if (avgc>5) {
			printf("Average: %d\n", avgc);
			p.convertTo(pavg, CV_32F, avgc);
			cout << avgc << pavg << endl;
		}
	break;
	case 'w':
		if (f>0)
			of.predict(pavg);
	break;
	case 'r':
		if (f>0)
			of.predict(vec);
	break;
	case 'a':
		of.label++;
		printf("CL: %d\n", of.label);
	break;
	}
}
cap.release();
}


int connect_db(char *cinfo)
{
conn=PQconnectdb(cinfo ? cinfo : "");
if (PQstatus(conn) != CONNECTION_OK) {
	fprintf(stderr, "Connection to database failed: %s", PQerrorMessage(conn));
	PQfinish(conn);
	conn=NULL;
	return -1;
}

return 0;
}

int main(int argc, char **argv)
{
int opt,camera_id=0;
char *dbopts=NULL;
char *input=NULL;

YOLOv8_face face("weights/yolov8n-face.onnx", 0.45, 0.5);
OpenFace of("weights/nn4.v2.t7");

SelfieSegment ss("/data/AI/selfie_segmenter.tflite");

if (argc>1) {
	input=argv[1];
	camera_id=-1;
	optind=2;
}

while ((opt = getopt(argc, argv, "d:c:")) != -1) {
	switch(opt) {
	case 'd':
		dbopts=optarg;
	break;
	case 'c':
		camera_id=atoi(optarg);
	break;
	}
}

printf("DB: %s\n", dbopts);
connect_db(dbopts);

namedWindow(kWinName, WINDOW_NORMAL);
namedWindow(kWinRoi, WINDOW_NORMAL);
namedWindow(kWinMask, WINDOW_NORMAL);

createTrackbar("Focus:", kWinName, &simulatedFocus, 400);
createTrackbar("Threshold:", kWinName, &peakThreshold, 10);

p=cv::Mat(1, 128, CV_64F);

detect_from_video(face, of, ss, camera_id, input ? input : "");

destroyAllWindows();

if (conn)
	PQfinish(conn);

return 0;
}
