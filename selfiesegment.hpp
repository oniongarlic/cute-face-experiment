#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

class SelfieSegment
{
public:
	SelfieSegment(std::string modelpath);
	cv::Mat detect(cv::Mat &frame);

private:
	cv::dnn::Net net;
};
