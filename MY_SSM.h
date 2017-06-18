#ifndef MY_SSM_H
#define MY_SSM_H
#include <cv.h>
#include <highgui.h>
#include <vector>
using namespace cv;
class MY_SSM
{
public:
    Mat covar, means;
    Mat eValues, eVectors;
    Mat P, b, cenC;
    double variation;
    int modes_num;
    MY_SSM(string modelPath);
    MY_SSM();
    void update(Mat dX);
    void copy(MY_SSM cy);
    void update_noCons(Mat dX);
    void draw();
    void drawImage(Mat img, string windowName, int pluse);
    Mat get();

};
#endif
