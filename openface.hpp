#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

class OpenFace
{
public:
	OpenFace(std::string modelpath);
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
