#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <opencv2/tracking/tracking_by_matching.hpp>

#include <unistd.h>

#include <pqxx/pqxx>

#include "mqtt.h"

#include "openface.hpp"
#include "selfiesegment.hpp"
#include "focus_check.hpp"
#include "yolov8_face.h"
#include "persons.h"

using namespace cv;
using namespace dnn;
using namespace std;

static const string kWinName = "Face detection use OpenCV";
static const string kWinRoi = "ROI";
static const string kWinMask = "Mask";

struct opts {
    bool use_gpu=false;
    bool trackFace=false;
    int skip_frame=0;
    bool embeddings=false;
    bool store=false;
    bool oneshot=true;
    int person=-1;

    float faceThreshold=0.65;
    float nmsThreshold=0.5;
};

struct opts ao;

int simulatedFocus=0;
int imageBrightness=0;
int imageContrast=33;

int avgc=0;
cv::Mat p;
cv::Mat pavg;

pqxx::connection *cx;

mqtt mqtt;

Ptr<cv::Tracker> tracker;
bool trackFace=false;

int skip_frame=0;

Persons *pe;

class Face
{
public:
    Point2d center;
    Point2d nose;
    Point2d mouth;

    float confidence;

    MovingAverage ma_h;
    MovingAverage ma_v;

    cv::Mat face;
    cv::Mat e;

    int area;

    long frame;
};

Face theFace;

std::vector<Face> faces;

void focus_peaking(cv::Mat &image, bool inFocus)
{
    cv::Mat gray, edges, er;

    cv::cvtColor(image, gray, COLOR_BGR2GRAY);

    cv::GaussianBlur(gray, edges, Size(3, 3), 1.5, 1.5);
    cv::Canny(edges, edges, 10, 160, 3, true);

    cv::cvtColor(edges, er, COLOR_GRAY2BGR);
    er=er.mul(inFocus ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);

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

void visualize_embedding(cv::Mat &frame, const cv::Mat &e, int ypos=0)
{
    cv::Mat eg;
    e.convertTo(eg, CV_8U, 127.5, 127.5);
    cv::resize(eg, eg, cv::Size(256, 32), 0, 0, cv::INTER_NEAREST);
    cv::Mat roi=frame(cv::Rect(0, ypos, 256, 32));
    cv:cvtColor(eg, eg, cv::COLOR_GRAY2BGR);
    eg.copyTo(roi);
    // imshow("Embedding", eg);
}

void detect_from_video(YOLOv8_face &face, OpenFace &of, SelfieSegment &ss, int camera, string file="")
{
    cv::Mat frame;
    cv::TickMeter tm;

    VideoCapture cap;
    FocusCheck focus;

    bool run=true;
    bool peaking=true;
    bool haveface=false;
    bool tracking=false;
    bool predict=false;

    long frames=0;
    int label=0,fps=30,tracked=0,f=0;

    const cv::Scalar purple	(128.0, 0.0, 128.0);

    cv::Mat se; // compare to

    if (camera>-1) {
        cap.open(camera, CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
#ifdef FHD
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
#else
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
#endif
    } else {
        cap.open(file);
    }

    if (!cap.isOpened()) {
        printf("Failed to open video input\n");
        return;
    }

    label=pe->current();
    se=pe->get_embedding(label);

    double capw=cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double caph=cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    printf("Input resolution is: %f, %f\n", capw, caph);

    while (cap.read(frame) && run) {
        cv::Mat vec;
        cv::Mat scaled;
        float sdcos,dcos;
        YoloFace yf;

        frames++;

        if (skip_frame==1 && (frames & 1))
            continue;

        tm.start();

#if 0 
        // double scale = 1024.0f/frame.size().width;
        double scale=0.5;
        resize(frame, scaled, Size(), scale, scale, INTER_AREA);
#else
        scaled=frame;
#endif

        if (imageContrast!=33 || imageBrightness!=0) {
            scaled.convertTo(scaled, -1, (float)imageContrast/33.0, imageBrightness);
        }

        if (!tracking || tracked>fps) {

            f=face.detect(scaled); // scaled

            // Re-aquire face roi for tracker
            if (tracking && tracked>fps) {
                tracked=0;
                trackFace=true;
                tracking=false;
            }

            if (f>0) {
                const int i=face.getLargestFace();
                yf=face.getFace(i);
                theFace.face=face.getFaceMat(i, scaled);

                focus.simulatedFocus=simulatedFocus;
                // focus.isFocused(theFace.face, peaking);

                if (ao.embeddings) {
                    cv::Mat rface, fe, af;

                    //face.getRotatedFace(scaled, rface, i);
                    //imshow("RotatedFace", rface);

                    face.getAlignedFace(scaled, af, i);

                    fe=of.detect(af);
                    visualize_embedding(scaled, fe, 0);

                    if (!theFace.e.empty()) {
                        cv::detail::tracking::tbm::CosDistance cosd = cv::detail::tracking::tbm::CosDistance(fe.size());
                        dcos = cosd.compute(fe, theFace.e);
                        // printf("CosDist: %f\n", dcos);
                        //cout << "Current: " << fe << "\nPrevious: " << theFace.e << endl;
                        fe.copyTo(theFace.e);
                    } else {
                        //cout << "Initial e" << fe << endl;
                        fe.copyTo(theFace.e);
                    }

                    if (cx && ao.embeddings && ao.store) {
                        pe->save(fe);
                        if (ao.oneshot)
                            ao.store=false;
                    }
                    if (cx && ao.embeddings && predict) {
                        of.predict(fe);
                    }

                    if (!se.empty()) {
                        cv::detail::tracking::tbm::CosDistance cosd = cv::detail::tracking::tbm::CosDistance(fe.size());
                        sdcos = cosd.compute(fe, se);
                        //printf("CompareFaceDist: %f\n", sdcos);
                        visualize_embedding(scaled, se, 32);
                    }

                }

                if (trackFace && tracking==false) {
                    cv::Mat trackFaceRoi;

                    auto faceRoi=face.getROI(i);
                    //tracker = cv:: TrackerKCF::create();
                    tracker = cv:: TrackerCSRT::create();
                    tracker->init(scaled, faceRoi);
                    tracking=true;
                    trackFace=false;
                    printf("Using tracker to track face \n");

                    trackFaceRoi=scaled(faceRoi);
                    imshow("TRACK", trackFaceRoi);

                    tm.reset();
                }

                //cv::Mat s2, ssm;
                //ssm=ss.detect(scaled);
                // imshow(kWinMask, ssm);

                haveface=true;

                auto n=face.getNosePosition(i);
                theFace.nose=n;
                theFace.confidence=yf.confidence;

                theFace.ma_h.add((double)n.x);
                theFace.ma_v.add((double)n.y);

                mqtt.publish_point("face", n, yf.area, yf.confidence);

                //printf("Face size: %f (%f, %f) (%f) (%f,%f)\n", yf.area, n.x, n.y, yf.confidence, theFace.ma_h.get(), theFace.ma_v.get());

                face.drawPred(scaled, i);

            } else if (f==0 && haveface==true) {
                int r;
                const char *ja="{}";
                r=mqtt.publish_string("face", ja);
                haveface=false;
            } else if (f==0 && peaking) {
                focus_peaking(scaled, focus.inFocus);
            }
        }

        if (tracking) {
            cv::Rect2i troi;

            printf("Tracking update: ");

            const bool ok=tracker->update(scaled, troi);
            if (ok) {
                printf("OK\n");
                cv::rectangle(scaled, troi, purple);
                tracked++;
            } else {
                printf("LOST\n");
                tracking=false;
                tracked=false;
            }
        }

        tm.stop();

        const float closedist=0.1;
        putText(scaled, std::to_string(f), Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(128, 255, 128));
        putText(scaled, std::to_string(tm.getFPS()), Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(192, 255, 192));
        if (ao.embeddings) {
            putText(scaled, std::to_string(pe->current()), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(128, 255, 128));
            putText(scaled, pe->current_name(), Point(30, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(sdcos<closedist ? 255 : 128, 255, 128));
            putText(scaled, std::to_string(dcos), Point(10, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(192, 255, 192));
            putText(scaled, std::to_string(sdcos), Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(192, 255, 192));
        }
        if (f>0) {
            putText(scaled, std::to_string(theFace.nose.x), Point(10, 160), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(128, 255, 128));
            putText(scaled, std::to_string(theFace.nose.y), Point(10, 180), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(128, 255, 128));
            putText(scaled, std::to_string(yf.area), Point(10, 200), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(128, 255, 128));
        }

        imshow(kWinName, scaled);

        //printf("FPS: %f, Faces: (%d) %d\n", tm.getFPS(), f, face.faceCount);

        mqtt.loop();

        int key = waitKey(10);
        switch (key) {
        case 'q':
            run=false;
            break;
        case 'b':
            if (f>0 && ao.embeddings) {
                int id=pe->query_closest(theFace.e);
                printf("db says: %d\n", id);
            }
            break;
        case 's':
            if (f>0 && ao.embeddings) {
                printf("Adding face with label: %d\n", label);
                of.store(theFace.e, label);
            }
            break;
        case 'w':
            ao.store=!ao.store;
            printf("Embeddings store to database enabled: %d\n", ao.store);
            break;
        case 'e':
            ao.embeddings=!ao.embeddings;
            printf("Embeddings enabled: %d\n", ao.embeddings);
            break;
        case 'p':
            peaking=!peaking;
            break;
        case 'z':
            of.train();
            break;
        case 'c':
            p+=vec;
            avgc++;
            if (avgc>5) {
                printf("Average: %d\n", avgc);
                p.convertTo(pavg, CV_32F, avgc);
                cv::normalize(pavg, pavg, 1.0, 0.0, NORM_L2);
                cout << avgc << pavg << endl;
            }
            break;
        case 't':
            if (f>0 && pavg.rows>0)
                of.predict(pavg);
            break;
        case 'r':
            predict=!predict;
            break;
        case '+':
            label=pe->next();
            printf("Person ID: %d\n", pe->current());
            pe->get_name(label);
            se=pe->get_embedding(label);
            break;
        case '-':
            label=pe->previous();
            printf("Person ID: %d\n", pe->current());
            pe->get_name(label);
            se=pe->get_embedding(label);
            break;
        case 'm':
            trackFace=true;
            printf("Using tracker\n");
            break;
        }
    }
    cap.release();
}


int connect_db(char *cinfo)
{
    try {
        cx=new pqxx::connection(cinfo);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 1;
}

int main(int argc, char **argv)
{
    int opt,camera_id=0;
    char *dbopts=NULL;
    char *input=NULL;

    while ((opt = getopt(argc, argv, "f:d:c:p:sewg")) != -1) {
        switch(opt) {
        case 'f':
            input=optarg;
            camera_id=-1;
            break;
        case 'p':
            ao.person=atoi(optarg);
            break;
        case 'd':
            dbopts=optarg;
            break;
        case 'c':
            camera_id=atoi(optarg);
            break;
        case 's':
            skip_frame=1;
            break;
        case 'e':
            ao.embeddings=true;
            break;
        case 'w':
            ao.store=true;
            ao.oneshot=false;
            break;
        case 'g':
            ao.use_gpu=true;
            break;
        }
    }

    YoloFaceOptions yfo;

    yfo.model="weights/yolov8n-face.onnx";
    yfo.use_gpu=ao.use_gpu;
    yfo.confThreshold=ao.faceThreshold;
    yfo.nmsThreshold=ao.nmsThreshold;

    YOLOv8_face face(yfo);

    OpenFaceOptions ofo;

    ofo.model="weights/nn4.v2.t7";
    ofo.use_gpu=ao.use_gpu;

    OpenFace of(ofo);
    SelfieSegment ss("/data/AI/selfie_segmenter.tflite");

    printf("DB: %s\n", dbopts);
    printf("Camera: %d, skip: %d\n", camera_id, skip_frame);

    if (connect_db(dbopts)>0) {
        pe=new Persons(&of, cx);
        int r=pe->load();
        printf("Loaded %d persons\n", r);

        if (ao.person>0) {
            int t=pe->find(ao.person);
            if (t>0)
                printf("Default user set to %d\n", t);
        }

    } else {
        printf("No database\n");
    }

    mqtt.connect();

    namedWindow(kWinName, WINDOW_NORMAL);
    //namedWindow(kWinRoi, WINDOW_NORMAL);
    //namedWindow(kWinMask, WINDOW_NORMAL);

    createTrackbar("Focus:", kWinName, NULL, 400, [](int v, void *data){ simulatedFocus=v; });
    createTrackbar("Contrast:", kWinName, &imageContrast, 100, [](int v, void *data){ imageContrast=v; });
    createTrackbar("Brightness:", kWinName, NULL, 100, [](int v, void *data){ imageBrightness=v; });

    p=cv::Mat(1, 128, CV_64F);

    detect_from_video(face, of, ss, camera_id, input ? input : "");

    destroyAllWindows();

    if (cx) {
        cx->disconnect();
        delete cx;
    }

    return 0;
}
