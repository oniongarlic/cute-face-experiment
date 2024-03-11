#include "openface.hpp"

OpenFace::OpenFace(std::string modelpath)
{
net=cv::dnn::readNetFromTorch(modelpath);

net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

trainingData=cv::Mat(0, 128, CV_32F);
labels=cv::Mat(0, 1, CV_32SC1);

svm = cv::ml::SVM::create();
svm->setType(cv::ml::SVM::C_SVC);
svm->setKernel(cv::ml::SVM::RBF);
svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));
}

cv::Mat OpenFace::detect(cv::Mat &frame)
{
std::vector<cv::Mat> outs;
cv::Mat blob;

cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(96, 96), cv::Scalar(0, 0, 0), true, false);
this->net.setInput(blob);
this->net.forward(outs);

return outs[0];
}

void OpenFace::store(cv::Mat vec)
{
cv::Mat l=cv::Mat(1, 1, CV_32SC1, label);

// store it
trainingData.push_back(vec);
labels.push_back(l);
printf("Data[%d]: %d\n", label, labels.rows);
}

void OpenFace::train()
{
if (trainingData.empty()) {
	printf("No data to train with ! (Store some with with 's')\n");
	return;
}
svm->train(trainingData, cv::ml::ROW_SAMPLE, labels);
}

void OpenFace::predict(cv::Mat &query)
{
cv::Mat res;
float r;

r=svm->predict(query, res);

std::cout << r << res << std::endl;
}
