#ifndef MY_DNN_H
#define MY_DNN_H
#include <cv.h>
#include <highgui.h>
#include <vector>
using namespace cv;
class MY_DNN
{
    public:
        int layer_number;
        int layer_unit[100];
        int input_size;
        int class_num;
        Mat W[100];
        Mat b[100];
        Mat softmax;
        MY_DNN(string modelPath);
        Mat predAct(Mat image);
        Mat pred(Mat image);
};
#endif
