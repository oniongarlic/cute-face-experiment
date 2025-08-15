#ifndef OPENFACE_H
#define OPENFACE_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

typedef struct OpenFaceOptions {
    std::string model;
    bool use_gpu;
} OpenFaceOptions;

class OpenFace
{
public:
    OpenFace(const OpenFaceOptions &opts);
	cv::Mat detect(cv::Mat &frame);
	void store(const cv::Mat &vec, int label);
	void train();
	void predict(cv::Mat &query);

private:
	cv::dnn::Net net;
	cv::Ptr<cv::ml::SVM> svm;
	cv::Mat trainingData;
	cv::Mat labels;
};

#endif
