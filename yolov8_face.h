#ifndef YOLOV8_FACE_H
#define YOLOV8_FACE_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "moving_average.hpp"

class YOLOv8_face
{
public:
    YOLOv8_face(std::string modelpath, float confThreshold, float nmsThreshold);
    int detect(cv::Mat& frame);
    cv::Mat theFace;
    cv::Rect lgbox;
    double variance = 0.0;
    int faceCount=0;
    double faceArea=0; // normalized 0-1
    
    void drawPred(cv::Mat &frame, int faceIndex);
    void getRotatedFace(const cv::Mat& frame, cv::Mat &output, const cv::Rect &roi, const std::vector<cv::Point> landmark);
    int getLargestFace();
    
    cv::Rect getFace(int idx) { return boxes[faces[idx]]; }
    cv::Mat getFaceMat(int idx, const cv::Mat &frame);
    float getFaceConfidence(int idx) { return confidences[idx]; }
    std::vector<cv::Point> getFaceLandmarks(int idx) { return landmarks[faces[idx]]; };
    cv::Point2f getNosePosition(int faceIndex);
    cv::Rect getROI(int faceIndex);

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
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector< std::vector<cv::Point>> landmarks;
    std::vector<int> faces;
    
    void softmax_(const float* x, float* y, int length);
    void generate_proposal(const cv::Mat &out, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw);
    
    MovingAverage avg_angle;
};

#endif // YOLOV8_FACE_H
