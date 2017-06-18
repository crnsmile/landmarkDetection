#include "MY_SSM.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;
using namespace cv;
// from pt0->pt1 and from pt0->pt2
double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-6);
}

Mat getCentre(Mat C)
{
    double mx = 0;
    double my = 0;
    for (int i = 0; i < C.rows / 2; i++)
    {
        mx += C.at<double>(i, 0);
        my += C.at<double>(i + C.rows / 2, 0);
    }
    mx /= C.rows / 2;
    my /= C.rows / 2;

    Mat centre(C.rows, 1, CV_64F);
    for (int i = 0; i < centre.rows / 2; i++)
    {
        centre.at<double>(i, 0) = mx;
        centre.at<double>(i + centre.rows / 2, 0) = my;
    }
    return centre;
}

Mat initC(double cenX, double cenY)
{
    Mat initCen(38, 1, CV_64F);
    for (int i = 0; i < 19; i++)
    {
        initCen.at<double>(i, 0) = cenX;
        initCen.at<double>(i + 19, 0) = cenY;
    }
    return initCen;
}

Mat constrain(Mat b, Mat Values)
{
    double con = 0.1;
    Mat tem = b.clone();
    for (int i = 0; i < tem.rows; i++)
    {
        if (tem.at<double>(i, 0) > con * sqrt(Values.at<double>(i, 0)))
        {
            tem.at<double>(i, 0) = con * sqrt(Values.at<double>(i, 0));
            //cout <<i<< " up" << endl;
        }
        else if (tem.at<double>(i, 0) < -1 * con * sqrt(Values.at<double>(i, 0)))
        {
            tem.at<double>(i, 0) = -1 * con * sqrt(Values.at<double>(i, 0));
            //cout <<i<< " down" << endl;
        }
    }
    return tem;
}

MY_SSM::MY_SSM(string modelPath)
{
    //Mat covar, means;
    //Mat eValues, eVectors;
    //Mat P, b;
    //double variation;
    //int modes_num;

    variation = 0.99;
    // 5, 17, 12
    // 6, 10, 11
    string pointPath = modelPath;
    int num_image = 400;
    char c;
    Mat matric(38, 400, CV_64F);
    double cenX = 0;
    double cenY = 0;
    for (int i = 1; i <= num_image; i++)
    {
        char ind[100] = "";
        sprintf(ind, "%d", i);
        string sInd = "00";
        if (i >= 10)
            sInd = "0";
        if (i >= 100)
            sInd = "";
        sInd = sInd + ind;
        string imagePoints = pointPath + sInd + ".txt";
        ifstream finn(imagePoints.c_str());

        double mx = 0;
        double my = 0;
        for (int j = 1; j <= 19; j++)
        {
            double x, y;
            finn >> x >> c >> y;
            matric.at<double>(j - 1, i - 1) = x;
            matric.at<double>(j - 1 + 19, i - 1) = y;
            mx += x;
            my += y;
        }
        mx /= 19;
        my /= 19;
        for (int j = 1; j <= 19; j++)
        {
            matric.at<double>(j - 1, i - 1) -= mx;
            matric.at<double>(j - 1 + 19, i - 1) -= my;
        }
        cenX += mx;
        cenY += my;

        //Point p1(matric.at<double>(4, i - 1), matric.at<double>(4 + 19, i - 1));
        //Point p2(matric.at<double>(16, i - 1), matric.at<double>(16 + 19, i - 1));
        //Point p0(matric.at<double>(11, i - 1), matric.at<double>(11 + 19, i - 1));
        //matric.at<double>(38, i - 1) = angle(p1, p2, p0)*1;
        //Point p11(matric.at<double>(5, i - 1), matric.at<double>(5 + 19, i - 1));
        //Point p22(matric.at<double>(9, i - 1), matric.at<double>(9 + 19, i - 1));
        //Point p00(matric.at<double>(10, i - 1), matric.at<double>(10 + 19, i - 1));
        //matric.at<double>(39, i - 1) = angle(p11, p22, p00)*1;


        cout << "load points " << i << endl;
        finn.close();
    }

    cenX /= num_image;
    cenY /= num_image;

    Mat covarMat, meansMat;
    Mat eValuesMat, eVectorsMat;
    calcCovarMatrix(matric, covarMat, meansMat, CV_COVAR_NORMAL | CV_COVAR_COLS);
    eigen(covarMat, eValuesMat, eVectorsMat);

    Scalar s = sum(eValuesMat);
    double totV = s[0];
    double V = 0;
    int k = 0;
    for (int i = 0; i < eValuesMat.rows; i++)
    {
        V += eValuesMat.at<double>(i, 0);
        if (V / totV > variation)
        {
            k = i;
            break;
        }
    }

    Mat tem(eValuesMat.rows, 1, CV_64FC1, Scalar(0));
    cout << "------" << endl;
    cout << eValuesMat << endl;
    cout << "-------" << endl;
    b = tem.clone();
    P = eVectorsMat.t();
    covar = covarMat.clone();
    means = meansMat.clone();
    eValues = eValuesMat.clone();
    eVectors = eVectorsMat.clone();
    modes_num = k;
    cenC = initC(cenX, cenY);
}

MY_SSM::MY_SSM()
{}

void MY_SSM::copy(MY_SSM cy)
{
    //Mat covar, means;
    covar = cy.covar.clone();
    means = cy.means.clone();

    //Mat eValues, eVectors;
    eValues = cy.eValues.clone();
    eVectors = cy.eVectors.clone();
    //Mat P, b;
    P = cy.P.clone();
    b = cy.b.clone();
    //double variation;
    variation = cy.variation;
    //int modes_num;
    modes_num = cy.modes_num;
    cenC = cy.cenC;
}

void MY_SSM::update(Mat landmarks)
{
    Mat genM = means + P*b;
    Mat cenLand = getCentre(landmarks);
    Mat norLand = landmarks - cenLand;
    Mat dX = norLand - genM;
    //Mat P_1 = P.inv();
    Mat P_1 = P.t();
    //cout << "dX" << dX.rows << " " << dX.cols << endl;
    //cout << "P_1" << P_1.rows << " " << P_1.cols << endl;
    Mat db = P_1*dX;
    //b = b + db;
    b = constrain(b + db, eValues);
    cenC = cenLand.clone();
}

void MY_SSM::update_noCons(Mat landmarks)
{
    Mat genM = means + P*b;
    Mat cenLand = getCentre(landmarks);
    Mat norLand = landmarks - cenLand;
    Mat dX = norLand - genM;
    //Mat P_1 = P.inv();
    Mat P_1 = P.t();
    //cout << "dX" << dX.rows << " " << dX.cols << endl;
    //cout << "P_1" << P_1.rows << " " << P_1.cols << endl;
    Mat db = P_1*dX;
    b = b + db;
    //b = constrain(b + db, eValues);
    cenC = cenLand.clone();
}

Mat MY_SSM::get()
{
    Mat genM = means + P*b + cenC;
    return genM;
}

void MY_SSM::draw()
{
    Mat showShape(2400, 1935, CV_8UC1, Scalar(255));
    namedWindow("show shape", CV_WINDOW_NORMAL);
    Mat genM = this->get();
    Point poi[25];
    for (int i = 0; i < 19; i++)
    {
        poi[i].x = genM.at<double>(i, 0);
        poi[i].y = genM.at<double>(i + 19, 0);
        circle(showShape, poi[i], 8, Scalar(0), CV_FILLED, CV_AA, 0);
        cout << poi[i] << endl;
    }
    line(showShape, poi[0], poi[3], Scalar(0), 2, 8, 0);
    line(showShape, poi[3], poi[18], Scalar(0), 2, 8, 0);
    line(showShape, poi[18], poi[9], Scalar(0), 2, 8, 0);
    line(showShape, poi[9], poi[18], Scalar(0), 2, 8, 0);
    line(showShape, poi[1], poi[0], Scalar(0), 2, 8, 0);
    line(showShape, poi[1], poi[2], Scalar(0), 2, 8, 0);
    line(showShape, poi[17], poi[2], Scalar(0), 2, 8, 0);
    line(showShape, poi[14], poi[12], Scalar(0), 2, 8, 0);
    line(showShape, poi[13], poi[12], Scalar(0), 2, 8, 0);
    line(showShape, poi[13], poi[15], Scalar(0), 2, 8, 0);
    line(showShape, poi[17], poi[4], Scalar(0), 2, 8, 0);
    line(showShape, poi[4], poi[11], Scalar(0), 2, 8, 0);
    line(showShape, poi[5], poi[10], Scalar(0), 2, 8, 0);
    line(showShape, poi[5], poi[6], Scalar(0), 2, 8, 0);
    line(showShape, poi[6], poi[8], Scalar(0), 2, 8, 0);
    line(showShape, poi[7], poi[8], Scalar(0), 2, 8, 0);
    line(showShape, poi[7], poi[9], Scalar(0), 2, 8, 0);
    line(showShape, poi[10], poi[9], Scalar(0), 2, 8, 0);
    line(showShape, poi[17], poi[16], Scalar(0), 2, 8, 0);
    line(showShape, poi[11], poi[16], Scalar(0), 2, 8, 0);
    imshow("show shape", showShape);
    waitKey(0);
}

void MY_SSM::drawImage(Mat img, string windowName, int pluse)
{
    //Mat showShape(2400, 1935, CV_8UC1, Scalar(255));
    Mat showShape = img.clone();
    namedWindow(windowName.c_str(), CV_WINDOW_NORMAL);
    Mat genM = this->get();
    Point poi[25];
    for (int i = 0; i < 19; i++)
    {
        poi[i].x = genM.at<double>(i, 0);
        poi[i].y = genM.at<double>(i + 19, 0);
        circle(showShape, poi[i], 8, Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
        //cout << poi[i] << endl;
    }
    line(showShape, poi[0], poi[3], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[3], poi[18], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[18], poi[9], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[9], poi[18], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[1], poi[0], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[1], poi[2], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[17], poi[2], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[14], poi[12], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[13], poi[12], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[13], poi[15], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[17], poi[4], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[4], poi[11], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[5], poi[10], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[5], poi[6], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[6], poi[8], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[7], poi[8], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[7], poi[9], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[10], poi[9], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[17], poi[16], Scalar(0, 255, 0, 0), 2, 8, 0);
    line(showShape, poi[11], poi[16], Scalar(0, 255, 0, 0), 2, 8, 0);
    imshow(windowName.c_str(), showShape);
    if (pluse) waitKey(0);
}
