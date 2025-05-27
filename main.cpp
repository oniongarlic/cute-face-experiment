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

#include <mosquitto.h>

#include "openface.hpp"
#include "selfiesegment.hpp"
#include "focus_check.hpp"
#include "yolov8_face.h"

using namespace cv;
using namespace dnn;
using namespace std;

static const string kWinName = "Face detection use OpenCV";
static const string kWinRoi = "ROI";
static const string kWinMask = "Mask";

static struct mosquitto *mqtt=NULL;
const char *mqtt_host="localhost";
const char *mqtt_clientid="face-detecor";
const char *mqtt_topic_prefix="video/0/facedetect";

int simulatedFocus=0;
int imageBrightness=0;
int imageContrast=33;

int avgc=0;
cv::Mat p;
cv::Mat pavg;

PGconn *conn;

int skip_frame=0;

class Face
{
public:
    Point center;
    Point nose;
    Point mouth;

    int area;
};

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

int mqtt_publish_info_topic_point(struct mosquitto *mqtt, const char *prefix, const char *topic, cv::Point2f p, float area, float conf)
{
    int r;
    char ftopic[80];
    char data[256];

    snprintf(ftopic, sizeof(ftopic), "%s/%s", prefix, topic);
    snprintf(data, sizeof(data), "{\"face\": [%f,%f,%f,%f]}", p.x, p.y, area, conf);

    r=mosquitto_publish(mqtt, NULL, ftopic, strlen(data), data, 0, false);
    if (r!=MOSQ_ERR_SUCCESS)
        fprintf(stderr, "MQTT Publish for info [%s] failed with %s\n", topic, mosquitto_strerror(r));

    return r;
}

int mqtt_publish_info_topic_int(struct mosquitto *mqtt, const char *prefix, const char *topic, int value)
{
    int r;
    char ftopic[80];
    char data[256];

    snprintf(ftopic, sizeof(ftopic), "%s/%s", prefix, topic);
    snprintf(data, sizeof(data), "%d", value);

    r=mosquitto_publish(mqtt, NULL, ftopic, strlen(data), data, 0, false);
    if (r!=MOSQ_ERR_SUCCESS)
        fprintf(stderr, "MQTT Publish for info [%s] failed with %s\n", topic, mosquitto_strerror(r));

    return r;
}


void detect_from_video(YOLOv8_face &face, OpenFace &of, SelfieSegment &ss, int camera, string file="")
{
    bool run=true;
    bool embeddings=false;
    bool store=false;
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
    int frames=0;
    int f;
    bool haveface=false;
    int label=0;

    while (cap.read(frame) && run) {
        cv::Mat vec;
        cv::Mat scaled;

        frames++;

        tm.start();

        double scale = 1024.0f/frame.size().width;
        resize(frame, scaled, Size(), scale, scale, INTER_AREA);

        if (imageContrast!=33 || imageBrightness!=0) {
            scaled.convertTo(scaled, -1, (float)imageContrast/33.0, imageBrightness);
        }

        if (skip_frame==1 && (frames & 1)) {
            continue;
        } else {

            f=face.detect(scaled);

            if (f>0) {
                int i=face.getLargestFace();
                cv::Mat theFace=face.getFaceMat(i, scaled);
                focus.simulatedFocus=simulatedFocus;

                focus.isFocused(theFace, peaking);
                if (embeddings) {
                    vec=of.detect(theFace);
                    if (conn && embeddings && store) {
                        dump_face(vec, label);
                    }
                }

                //cv::Mat s2, ssm;
                //ssm=ss.detect(scaled);
                // imshow(kWinMask, ssm);

                haveface=true;

                auto n=face.getNosePosition(i);
                float conf=face.getFaceConfidence(i);

                mqtt_publish_info_topic_point(mqtt, mqtt_topic_prefix, "face", n, face.faceArea, conf);

                printf("Face size: %f (%f, %f) (%f)\n", face.faceArea, n.x, n.y, conf);

                face.drawPred(scaled, i);

            } else if (f==0 && haveface==true) {
                int r;
                const char *ja="{}";
                r=mosquitto_publish(mqtt, NULL, "video/0/facedetect/face", strlen(ja), ja, 0, false);
                haveface=false;
            }

            if (f==0 && peaking) {
                focus_peaking(scaled, focus.inFocus);
            }
        }

        imshow(kWinName, scaled);

        tm.stop();

        // printf("FPS: %f, Faces: (%d) %d\n", tm.getFPS(), f, face.faceCount);

        int key = waitKey(1);
        switch (key) {
        case 'q':
            run=false;
            break;
        case 's':
            if (f>0 && embeddings) {
                printf("Adding face with label: %d\n", label);
                of.store(vec, label);
            }
            break;
        case 'w':
            store=!store;
            printf("Embeddings store to database enabled: %d\n", store);
            break;
        case 'e':
            embeddings=!embeddings;
            printf("Embeddings enabled: %d\n", embeddings);
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
                cout << avgc << pavg << endl;
            }
            break;
        case 't':
            if (f>0 && pavg.rows>0)
                of.predict(pavg);
            break;
        case 'r':
            if (f>0 && vec.rows>0)
                of.predict(vec);
            break;
        case 'a':
            label++;
            printf("Label ID is now: %d\n", label);
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

void mqtt_log_callback(struct mosquitto *m, void *userdata, int level, const char *str)
{
    // fprintf(stderr, "[MQTT-%d] %s\n", level, str);
}

int connect_mqtt(void)
{
    int port = 1883;
    int keepalive = 120;
    bool clean_session = true;

    mqtt=mosquitto_new(mqtt_clientid, clean_session, NULL);
    mosquitto_log_callback_set(mqtt, mqtt_log_callback);

    printf("Connecting to MQTT...\n");

    if (mosquitto_connect(mqtt, mqtt_host, port, keepalive)) {
        fprintf(stderr, "Unable to connect.\n");
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

    while ((opt = getopt(argc, argv, "f:d:c:s")) != -1) {
        switch(opt) {
        case 'f':
            input=optarg;
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
        }
    }

    printf("DB: %s\n", dbopts);
    printf("Camera: %d, skip: %d\n", camera_id, skip_frame);
    connect_db(dbopts);

    mosquitto_lib_init();
    connect_mqtt();


    namedWindow(kWinName, WINDOW_NORMAL);
    namedWindow(kWinRoi, WINDOW_NORMAL);
    //namedWindow(kWinMask, WINDOW_NORMAL);

    createTrackbar("Focus:", kWinName, &simulatedFocus, 400);
    createTrackbar("Contrast:", kWinName, &imageContrast, 100);
    createTrackbar("Brightness:", kWinName, &imageBrightness, 100);

    p=cv::Mat(1, 128, CV_64F);

    detect_from_video(face, of, ss, camera_id, input ? input : "");

    destroyAllWindows();

    if (conn)
        PQfinish(conn);

    mosquitto_destroy(mqtt);
    mosquitto_lib_cleanup();

    return 0;
}
