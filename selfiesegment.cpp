#include "selfiesegment.hpp"

SelfieSegment::SelfieSegment(std::string modelpath)
{
//net=cv::dnn::readNetFromTFLite(modelpath);
net=cv::dnn::readNetFromONNX("weights/model_float32.onnx");

//net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
//net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

cv::Mat SelfieSegment::detect(cv::Mat &frame)
{
std::vector<cv::Mat> outs;
cv::Mat resized;

#if 1
cv::Mat blob;
blob=cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(256,256), 0, true);
#else
cv::resize(frame, resized, cv::Size(256,256));
resized.convertTo(resized, CV_32F );
int sz[] = {1, 256, 256, 3};
cv::Mat blob(4, sz, CV_32F, resized.ptr<float>(0));
#endif

this->net.setInput(blob);

std::vector<cv::String> outNames = this->net.getUnconnectedOutLayersNames();

this->net.forward(outs, outNames);

std::cout << outs.size() << ":" << outs[0].elemSize() << outs[0].cols << std::endl;

return outs[0];
}
