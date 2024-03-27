#include <opencv2/opencv.hpp>

class FocusCheck
{
public:
bool isFocused(cv::Mat &face, bool peaking);
float focusThreshold=0.01;
float variance=0.0;
float lvariance=0.0;
int simulatedFocus=0;
bool inFocus=false;
};
