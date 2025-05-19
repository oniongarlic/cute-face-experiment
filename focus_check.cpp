#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "focus_check.hpp"

static void fftShift(const cv::Mat &input, cv::Mat &output)
{
int cx=input.cols/2;
int cy=input.rows/2;

cv::Mat iq0(input, cv::Rect(0, 0, cx, cy));
cv::Mat iq1(input, cv::Rect(cx, 0, cx, cy));
cv::Mat iq2(input, cv::Rect(0, cy, cx, cy));
cv::Mat iq3(input, cv::Rect(cx, cy, cx, cy));

cv::Mat dq0(output, cv::Rect(0, 0, cx, cy));
cv::Mat dq1(output, cv::Rect(cx, 0, cx, cy));
cv::Mat dq2(output, cv::Rect(0, cy, cx, cy));
cv::Mat dq3(output, cv::Rect(cx, cy, cx, cy));

iq3.copyTo(dq0);
iq0.copyTo(dq3);
iq2.copyTo(dq1);
iq1.copyTo(dq2);
}

bool FocusCheck::isFocused(cv::Mat &face, bool peaking)
{
cv::Mat iroig, lap, lapim;
cv::Mat edges, er, i32, r128, ci;
cv::Scalar emean, estddev;
double ev;
cv::Size sDft=cv::Size(128, 128);

if (simulatedFocus>0) {
	double s=(double)simulatedFocus/50.0;
	cv::GaussianBlur(face, face, cv::Size(7, 7), s, s);
}

cv::cvtColor(face, iroig, cv::COLOR_BGR2GRAY);

cv::resize(iroig, r128, sDft, 0, 0, cv::INTER_AREA);

//cv::equalizeHist(r128, r128);

r128.convertTo(i32, CV_32F);

cv::Mat planes[] = {i32, cv::Mat::zeros(sDft, CV_32F)};
cv::merge(planes, 2, ci);

cv::dft(i32, ci);
cv::split(ci, planes);

cv::Mat mag, mags=cv::Mat::zeros(sDft, CV_32F);
cv::magnitude(planes[0], planes[1], mag);
mag+=cv::Scalar::all(1);
cv::log(mag, mag);
cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
fftShift(mag, mags);

int cx=mags.cols/2;
int cy=mags.rows/2;
int wh=mags.rows/2/2;

cv::Mat c(mags, cv::Rect(cx-wh, cy-wh, cx, cy));
cv::Mat ch(mag, cv::Rect(cx-wh, cy-wh, cx, cy));

// cv::threshold(mag, mag, (float)peakThreshold/10.0, 1, 1);

cv::meanStdDev(mag, emean, estddev);
float hvariance = estddev.val[0] * estddev.val[0];

cv::meanStdDev(mags, emean, estddev);
variance = estddev.val[0] * estddev.val[0];

// printf("dftMean: %f (%f)-(%f)\n", emean[0], variance, hvariance);

inFocus=(variance<focusThreshold) ? true : false;

cv::imshow("dftRaw", mag);
cv::imshow("dftShift", mags);

cv::equalizeHist(iroig, iroig);
cv::Laplacian(iroig, lap, CV_32F, 3);

cv::Scalar mean, stddev; // 0:1st channel, 1:2nd channel and 2:3rd channel
cv::meanStdDev(lap, mean, stddev);
lvariance = stddev.val[0] * stddev.val[0];

// printf("lapVAR: %f\n", lvariance);

// Red edges
if (peaking) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::GaussianBlur(iroig, edges, cv::Size(3, 3), 1.5, 1.5);
    cv::Canny(edges, edges, 100, 300, 3, true);

#if 1
    cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(face, contours, -1, cv::Scalar(0,0,255), 1);
#else
    cv::cvtColor(edges, er, cv::COLOR_GRAY2BGR);
    er=er.mul(cv::Scalar(0, 0, 255), 1);
    cv::bitwise_or(face, er, face, edges);
#endif
}

return inFocus;
}
