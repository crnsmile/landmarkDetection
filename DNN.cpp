#include "MY_DNN.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;
//int layer_number;
//int layer_unit[100];
//int input_size;
//int class_num;
//Mat W[100];
//Mat b[100];
//Mat softmax;

double MY_MAX(Mat X)
{
    double maxD = -1000000;
    for (int i = 0; i < X.rows; i++)
    {
        if (maxD < X.at<double>(i, 0))
            maxD = X.at<double>(i, 0);
    }
    return maxD;
}

double MY_SUM(Mat X)
{
    double sumD = 0;
    for (int i = 0; i < X.rows; i++)
        sumD += X.at<double>(i, 0);
    return sumD;
}

Mat MY_EXP(Mat X)
{
    Mat tem(X.rows, X.cols, CV_64F);
    for (int i = 0; i < X.rows; i++)
        tem.at<double>(i, 0) = exp(X.at<double>(i, 0));
    return tem;
}

Mat MY_SUB(Mat X, double maxD)
{
    Mat tem(X.rows, X.cols, CV_64F);
    for (int i = 0; i < X.rows; i++)
        tem.at<double>(i, 0) = X.at<double>(i, 0) - maxD;
    return tem;
}

Mat MY_DIV(Mat X, double sumD)
{
    Mat tem(X.rows, X.cols, CV_64F);
    for (int i = 0; i < X.rows; i++)
        tem.at<double>(i, 0) = X.at<double>(i, 0) / sumD;
    return tem;
}

Mat sigmoid(Mat Z)
{
    Mat tem(Z.rows, Z.cols, CV_64F);
    for (int i = 0; i < Z.rows; i++)
    {
        double val = Z.at<double>(i, 0);
        tem.at<double>(i, 0) = 1 / (1 + exp(-1 * val));
    }
    //1 . / (1 + exp(-x));
    return tem;
}


Mat MY_DNN::predAct(Mat image)
{
    vector<double> result;
    //std::cout <<"inner: "<< layer_number << std::endl;
    Mat X(input_size, 1, CV_64F);
    Mat image_roi = image.clone();
    Mat normal_image(image.rows, image.cols, CV_64F);
    cvtColor(image_roi, image_roi, CV_BGR2GRAY);
    resize(image_roi, image_roi, Size(32, 32), 0, 0, 1);
    image_roi.convertTo(normal_image, CV_64F);
    normalize(normal_image, normal_image, 1.0, 0.0, NORM_MINMAX);

    //cout << "see: " << normal_image.at<double>(15, 15) << endl;
    for (int r = 0; r < normal_image.rows; r++)
        for (int c = 0; c < normal_image.cols; c++)
            X.at<double>(r * normal_image.cols + c, 0) = normal_image.at<double>(r, c);

    for (int layer = 1; layer < layer_number - 1; layer++)
    {
        Mat Z = W[layer] * X + b[layer];
        Mat A = sigmoid(Z);
        X = A.clone();
    }
    //M = softmaxTheta * a{ n + 1 };
    //M = bsxfun(@minus, M, max(M, [], 1)); % prevent overflow
    //h = exp(M);
    //h = bsxfun(@rdivide, h, sum(h)); % normalize
    //[mm, pred] = max(h, [], 1);
    X = softmax * X;
    //X = MY_SUB(X, MY_MAX(X));
    //Mat H = MY_EXP(X);
    //H = MY_DIV(H, MY_SUM(H));

    //cout << "pre done" << endl;
    //return H;
    return X;
}

Mat MY_DNN::pred(Mat image)
{
    vector<double> result;
    //std::cout <<"inner: "<< layer_number << std::endl;
    Mat X(input_size, 1, CV_64F);
    Mat image_roi = image.clone();
    Mat normal_image(image.rows, image.cols, CV_64F);
    cvtColor(image_roi, image_roi, CV_BGR2GRAY);
    resize(image_roi, image_roi, Size(32, 32), 0, 0, 1);
    image_roi.convertTo(normal_image, CV_64F);
    //normal_image = normal_image / 256;
    normalize(normal_image, normal_image, 1.0, 0.0, NORM_MINMAX);

    //cout << "see: " << normal_image.at<double>(15, 15) << endl;
    for (int r = 0; r < normal_image.rows; r++)
        for (int c = 0; c < normal_image.cols; c++)
            X.at<double>(r * normal_image.cols + c, 0) = normal_image.at<double>(r, c);

    for (int layer = 1; layer < layer_number - 1; layer++)
    {
        Mat Z = W[layer] * X + b[layer];
        Mat A = sigmoid(Z);
        X = A.clone();
    }
    //M = softmaxTheta * a{ n + 1 };
    //M = bsxfun(@minus, M, max(M, [], 1)); % prevent overflow
    //h = exp(M);
    //h = bsxfun(@rdivide, h, sum(h)); % normalize
    //[mm, pred] = max(h, [], 1);
    X = softmax * X;
    X = MY_SUB(X, MY_MAX(X));
    Mat H = MY_EXP(X);
    H = MY_DIV(H, MY_SUM(H));

    //cout << "pre done" << endl;
    //return X;
    return H;
}

MY_DNN::MY_DNN(string modelPath)
{
    ifstream finn(modelPath.c_str());
    int n = 0;
    finn >> layer_number;
    for (int i = 0; i < layer_number; i++)
    {
        finn >> layer_unit[i];
        cout << layer_unit[i] << endl;
    }
    input_size = layer_unit[0];
    class_num = layer_unit[layer_number - 1];

    int input_units = layer_unit[layer_number - 2];
    int output_units = layer_unit[layer_number - 1];
    Mat softmaxPara(output_units, input_units, CV_64F);
    for (int j = 0; j < input_units; j++)
        for (int i = 0; i < output_units; i++)
        {
            n++;
            double tem;
            finn >> tem;
            softmaxPara.at<double>(i, j) = tem;
        }
    softmax = softmaxPara.clone();
    for (int layer = 1; layer < layer_number - 1; layer++)
    {
        int input_units = layer_unit[layer - 1];
        int output_units = layer_unit[layer];
        Mat paraW(output_units, input_units, CV_64F);
        Mat parab(output_units, 1, CV_64F);
        for (int j = 0; j < input_units; j++)
            for (int i = 0; i < output_units; i++)
            {
                n++;
                double tem;
                finn >> tem;
                paraW.at<double>(i, j) = tem;
            }
        for (int i = 0; i < output_units; i++)
        {
            n++;
            double tem;
            finn >> tem;
            parab.at<double>(i, 0) = tem;
        }
        W[layer] = paraW.clone();
        b[layer] = parab.clone();
    }
    //cout << "n: " << n << endl;
    finn.close();
}
