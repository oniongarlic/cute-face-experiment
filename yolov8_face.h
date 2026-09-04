#ifndef YOLOV8_FACE_H
#define YOLOV8_FACE_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "moving_average.hpp"

typedef struct YoloFaceOptions {
    std::string model;
    bool use_gpu;
    float confThreshold;
    float nmsThreshold;
} YoloFaceOptions;

class YoloFace {
public:
    int idx;
    cv::Rect box;
    float confidence;
    float angle;
    float area;
    bool inside;
    std::vector<cv::Point> landmarks;
    std::vector<float> landmarkconf;
};

class YOLOv8_face
{
public:
    YOLOv8_face(const YoloFaceOptions &opts);
    int detect(cv::Mat& srcimg);
    cv::Mat frame;
    cv::Rect lgbox;
    double variance = 0.0;
    int faceCount=0;
    
    void drawPred(cv::Mat &frame, int faceIndex);
    void getRotatedFace(cv::Mat &output, int faceIndex);
    int getLargestFace();
    
    YoloFace getFace(int idx);
    cv::Mat getFaceMat(int idx, const cv::Mat &frame);
    std::vector<cv::Point> getFaceLandmarks(int idx);
    cv::Point2f getNosePosition(int faceIndex);
    cv::Rect getROI(int faceIndex);

    void getAlignedFace(cv::Mat &output, int faceIndex);
private:
    cv::Mat resize_image(const cv::Mat srcimg, int *newh, int *neww, int *padh, int *padw);
    
    const bool keep_ratio = true;
    const int inpWidth = 640;
    const int inpHeight = 640;
    const double inpArea=inpWidth*inpHeight;
    
    float confThreshold;
    float nmsThreshold;
    const int num_class = 1;
    const int reg_max = 16;
    cv::dnn::Net net;
    cv::Mat rot;
    
    float srcRatioh;
    float srcRatiow;
    
    // The final list of faces
    std::vector<YoloFace> faces;
    std::vector<int> faceindex;
    
    void softmax_(const float* x, float* y, int length);
    void generate_proposal(const cv::Mat &out, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw);
};

#endif // YOLOV8_FACE_H
