#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include "MY_DNN.h"
#include "MY_SSM.h"
#include <queue>

using namespace std;
using namespace cv;


string pointPath = ".\\400_junior\\";
MY_DNN DN = MY_DNN("model_96.txt");
MY_DNN DNNG = MY_DNN("model_2classes_92.txt");
MY_SSM M1 = MY_SSM(pointPath);
const int stepSize = 10;
const int image_width = 2400;
const int image_heigth = 1935;
const int landmarkNumber = 19;
const double PThread = 0.9;
double possibility[image_width / stepSize + 1][image_heigth / stepSize + 1][landmarkNumber + 1] = { 0 };
double con[image_width / stepSize + 1][image_heigth / stepSize + 1][landmarkNumber + 1] = { 0 };
vector<Point>CP[landmarkNumber + 1];
time_t start, stop;
Mat img;
//Mat img = imread("381.bmp");

int hist[6] = { 0, 0, 0, 0, 0, 0 };
int phist[20][6] = { 0 };

struct Node
{
    int indLand;
    double  priority;
    Node(int land, double posi)
    {
        indLand = land;
        priority = posi;
    }
};

struct NodeCmp
{
    bool operator()(const Node &na, const Node &nb)
    {
        return na.priority > nb.priority;
    }
};

double MY_MAX1(Mat X)
{
    double maxD = -1000000;
    for (int i = 0; i < X.rows; i++)
    {
        if (maxD < X.at<double>(i, 0))
            maxD = X.at<double>(i, 0);
    }
    return maxD;
}

double Myabs(double a, double b)
{
    if (a > b) return a - b;
    else return b - a;
}

int MAX_ID(Mat X)
{
    double maxD = -1000000;
    int ID = 0;
    for (int i = 0; i < X.rows; i++)
    {
        if (maxD < X.at<double>(i, 0))
        {
            maxD = X.at<double>(i, 0);
            ID = i + 1;
        }
    }
    return ID;
}

Mat getCentrem(Mat C)
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


void on_mouse(int event, int x, int y, int flags, void* ustc)
{
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);

    if (event == CV_EVENT_LBUTTONDOWN)
    {
        Point pt = Point(x, y);
        Rect rect(x - 48, y - 48, 96, 96);
        Mat imgge = img.clone();
        Mat image_roi = imgge(rect);
        Mat H = DN.pred(image_roi);
        Mat HNG = DNNG.pred(image_roi);
        Mat HA = DN.predAct(image_roi);
        string flag = "Yes";
        Point poi;

        if (MAX_ID(HNG) == 2) flag = "No";
        rectangle(imgge, rect, Scalar(0, 255, 0, 0), 2, 8, 0);
        poi.x = x;
        poi.y = y;
        circle(imgge, poi, 3, Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
        imshow("test_demo", imgge);

        cout << "Predict landmark: " << MAX_ID(H) << " Possibility: " << MY_MAX1(H) << endl;

        //		cout << " Positive?: " << flag << endl;
        cout << endl;
        cout << endl;
        cout << endl;
        cout << endl;
        cout << endl;
        cout << endl;
        namedWindow("image_roi", CV_WINDOW_NORMAL);
        imshow("image_roi", image_roi);
        for (int i = 0; i < 19; i++)
            cout << i + 1 << " " << HA.at<double>(i, 0) << " " << H.at<double>(i, 0) << endl;
        cout << endl;
    }
}

void convolution(double possi[image_width / stepSize + 1][image_heigth / stepSize + 1][landmarkNumber + 1], double target[image_width / stepSize + 1][image_heigth / stepSize + 1][landmarkNumber + 1])
{
    int t = 0;
    int maskSize = 3;
    double mask5[5][5] = {
        { 1.0 / 273, 4.0 / 273, 7.0 / 273, 4.0 / 273, 1.0 / 273 },
        { 4.0 / 273, 16.0 / 273, 26.0 / 273, 16.0 / 273, 4.0 / 273 },
        { 7.0 / 273, 26.0 / 273, 41.0 / 273, 26.0 / 273, 7.0 / 273 },
        { 4.0 / 273, 16.0 / 273, 26.0 / 273, 16.0 / 273, 4.0 / 273 },
        { 1.0 / 273, 4.0 / 273, 7.0 / 273, 4.0 / 273, 1.0 / 273 }
    };
    /*double mask3[3][3] = {
    { 1.0 / 16, 2.0 / 16, 1.0 / 16 },
    { 2.0 / 16, 4.0 / 16, 2.0 / 16 },
    { 1.0 / 16, 2.0 / 16, 1.0 / 16 },
    };*/
    double mask3[3][3] = {
        { 0.00163118, 0.0371255, 0.00163118 },
        { 0.0371255, 0.844973, 0.0371255 },
        { 0.00163118, 0.0371255, 0.00163118 }
    };
    for (int land = 0; land < landmarkNumber; land++)
    {
        for (int x = 0; x < image_width / stepSize + 1; x++)
            for (int y = 0; y < image_heigth / stepSize + 1; y++)
            {
                double tem = 0;
                //if (possi[x][y][land] < PThread) continue;
                for (int ix = -1 * maskSize / 2; ix <= maskSize / 2; ix++)
                    for (int iy = -1 * maskSize / 2; iy <= maskSize / 2; iy++)
                    {
                        int xx = x + ix;
                        int yy = y + iy;
                        if (xx < 0 || xx >= image_width)continue;
                        if (yy < 0 || yy >= image_heigth)continue;
                        //if (possi[xx][yy][land] < PThread) continue;
                        tem += possi[xx][yy][land] * mask3[ix + maskSize / 2][iy + maskSize / 2];
                    }
                target[x][y][land] = tem;


                //target[x*(image_heigth / stepSize + 1)*(landmarkNumber + 1) + y*(landmarkNumber + 1)+land] = tem;
            }
    }

}

void genCondidate(double posi[image_width / stepSize + 1][image_heigth / stepSize + 1][landmarkNumber + 1], vector<Point>CP[landmarkNumber + 1])
{
    int t = 0;
    for (int land = 0; land < landmarkNumber; land++)
        for (int r = 0; r < image_width; r += stepSize)
            for (int c = 0; c < image_heigth; c += stepSize)
            {
                int ix = r / stepSize;
                int iy = c / stepSize;
                if (posi[ix][iy][land] > PThread)
                {
                    t++;
                    Point temP(r, c);
                    CP[land].push_back(temP);
                }
            }
    cout << "number of condidate: " << t << endl;
}

double countPosi(double possibility[image_width / stepSize + 1][image_heigth / stepSize + 1][landmarkNumber + 1], Mat Upoints)
{
    double out = 0;
    for (int land = 0; land < landmarkNumber; land++)
    {
        double x = Upoints.at<double>(land, 0);
        double y = Upoints.at<double>(land + 19, 0);
        int ix = int(x / stepSize);
        int iy = int(y / stepSize);
        if (possibility[iy][ix][land] <= 0)
            out += 0;
        else
            out += possibility[iy][ix][land];
    }
    //return 1 - out;
    return landmarkNumber - out;
}

int countDist(Mat Upoints, Mat landmarks)
{
    int out = 0;
    for (int land = 0; land < landmarkNumber; land++)
    {
        out += Myabs(Upoints.at<double>(land, 0), landmarks.at<double>(land, 0));
        out += Myabs(Upoints.at<double>(land + landmarkNumber, 0), landmarks.at<double>(land + landmarkNumber, 0));
    }
    return out;
}

double evaluation(Mat landmarkse, MY_SSM ssm, double possi[image_width / stepSize + 1][image_heigth / stepSize + 1][landmarkNumber + 1])
{

    double alpha = 0;
    MY_SSM ssmC;
    ssmC.copy(ssm);
    ssmC.update(landmarkse);
    //ssmC.update_noCons(landmarks);
    Mat Upoints = ssmC.get();
    int sumdiff = countDist(Upoints, landmarkse);
    //if (sumdiff != 0) cout << "error" << endl;
    //if (sumdiff[0] < THREAD) break;
    double posiItem = countPosi(possi, Upoints);
    double cost = alpha * sumdiff + (1 - alpha)*posiItem;
    //cout << "sumdiff: " << sumdiff << "posiItem£º " << posiItem << endl;
    return cost;
}

double Printevaluation(Mat landmarkse, MY_SSM ssm, double possi[image_width / stepSize + 1][image_heigth / stepSize + 1][landmarkNumber + 1])
{

    double alpha = 0;
    MY_SSM ssmC;
    ssmC.copy(ssm);
    ssmC.update(landmarkse);
    //ssmC.update_noCons(landmarks);
    Mat Upoints = ssmC.get();
    int sumdiff = countDist(Upoints, landmarkse);
    //if (sumdiff != 0) cout << "error" << endl;
    //if (sumdiff[0] < THREAD) break;
    double posiItem = countPosi(possi, Upoints);
    double cost = alpha * sumdiff + (1 - alpha)*posiItem;

    for (int land = 0; land < landmarkNumber; land++)
        cout << "diff: " << Myabs(Upoints.at<double>(land, 0), landmarkse.at<double>(land, 0)) << " " << Myabs(Upoints.at<double>(land + landmarkNumber, 0), landmarkse.at<double>(land + landmarkNumber, 0)) << endl;

    for (int land = 0; land < landmarkNumber; land++)
    {
        double x = Upoints.at<double>(land, 0);
        double y = Upoints.at<double>(land + 19, 0);
        int ix = int(x / stepSize);
        int iy = int(y / stepSize);
        cout << "posibility: " << land + 1 << " " << possi[iy][ix][land] << endl;
    }

    cout << "sumdiff: " << sumdiff << "posiItem£º " << posiItem << endl;
    return cost;
}

void preprocess(MY_DNN dnn, MY_DNN dnnNG, Mat img, int roiSize, int stepSize)
{
    cout << "preprocess" << endl;
    cout << img.rows << endl;
    Mat image = img.clone();
    Mat normal_image = img.clone();

    Mat Response_surface(image_width / stepSize + 1, image_heigth / stepSize + 1, CV_64F);
    for (int r = 0; r < image_width; r += stepSize)
        for (int c = 0; c < image_heigth; c += stepSize)
        {
            if (r % 100 == 0 && c % 1000 == 0)
                cout << r << " " << c << endl;
            double xx = r - roiSize / 2;
            double yy = c - roiSize / 2;
            if ((xx < 0) || (xx + roiSize) > image_width) continue;
            if ((yy < 0) || (yy + roiSize) > image_heigth) continue;

            Rect rect(yy, xx, roiSize, roiSize);
            Mat image_roi = image(rect);
            Mat HNG = dnnNG.pred(image_roi);
            if (MAX_ID(HNG) != 1) continue;
            Mat H = dnn.pred(image_roi);
            for (int i = 0; i < 19; i++)
            {
                possibility[r / stepSize][c / stepSize][i] = H.at<double>(i, 0);
                Response_surface.at<double>(r / stepSize, c / stepSize) = H.at<double>(i, 0);
                Point poi(c, r);

                //if ( (i == 0) || (i == 1) || i == 2 || i == 3 || i == 9 || i = 12 || i == 18)
                if (i == 9)
                    if (H.at<double>(i, 0) > PThread)
                        circle(normal_image, poi, 3, Scalar(0, 0, H.at<double>(i, 0) * 255, 0), CV_FILLED, CV_AA, 0);
            }
        }
    convolution(possibility, con);
    genCondidate(possibility, CP);
    for (int i = 0; i < 19; i++)
    {
        cout << "landmark: " << i + 1 << " " << CP[i].size() << endl;
    }
    //namedWindow("Response_surface", CV_WINDOW_NORMAL);
    //imshow("Response_surface", Response_surface);
    //namedWindow("visualization", CV_WINDOW_NORMAL);
    //imshow("visualization", normal_image);

    waitKey(0);
    cout << "end preprocess" << endl;
}

Mat readjust(MY_SSM ssm, int searchSize)
{
    int roiSize = 96;
    double stageMinCost;
    Mat landmarks;
    MY_SSM ssmstage;

    priority_queue<Node, vector<Node>, NodeCmp> que;

    ssmstage.copy(ssm);
    landmarks = ssmstage.get();
    for (int land = 0; land < landmarkNumber; land++)
    {
        double nowy = landmarks.at<double>(land, 0);
        double nowx = landmarks.at<double>(land + landmarkNumber, 0);
        double posi = possibility[int(nowx / stepSize)][int(nowy / stepSize)][land];
        que.push(Node(land, posi));
    }

    while (!que.empty())
    {
        Node now = que.top();
        que.pop();
        cout << now.priority << endl;
        int land = now.indLand;
        int len = CP[land].size();
        landmarks = ssmstage.get();
        double nowy = landmarks.at<double>(land, 0);
        double nowx = landmarks.at<double>(land + landmarkNumber, 0);
        //if (possibility[int(nowx) / stepSize][int(nowy) / stepSize][land] > 0.9)continue;
        stageMinCost = 10000000;
        for (int i = 0; i < len; i++)
        {
            Point temP = CP[land][i];
            if (Myabs(temP.x, nowx) > searchSize / 2 || Myabs(temP.y, nowy) > searchSize / 2) continue;

            Mat landmarks1 = landmarks.clone();
            landmarks1.at<double>(land, 0) = temP.y;
            landmarks1.at<double>(land + 19, 0) = temP.x;

            double cost = evaluation(landmarks1, ssmstage, possibility);
            if (cost < stageMinCost)
            {
                stageMinCost = cost;
                landmarks.at<double>(land, 0) = temP.y;
                landmarks.at<double>(land + 19, 0) = temP.x;
            }
        }
        ssmstage.update_noCons(landmarks);
    }
    landmarks = ssmstage.get();
    return landmarks;
}

Mat searchLandmarks(MY_DNN dnn, MY_DNN dnnNG, MY_SSM ssm, Mat img)
{
    start = time(NULL);
    Mat points = ssm.get();
    Mat image = img.clone();
    double searchSize = 600;
    int roiSize = 96;
    int steps = searchSize / stepSize;
    double THREAD = 0.2;
    double minCost = 10000001;
    double resultcost = 10000001;
    preprocess(dnn, dnnNG, img, roiSize, stepSize);
    //ssm.drawImage(image,"shape",1);
    int t = 0;
    int ctt = 0;
    Mat finalResult;
    while (true)
    {
        points = ssm.get();
        Mat landmarks = points.clone();
        t++;
        if (t <= 1)
        {
            Mat MaxP(points.rows / 2, 1, CV_64FC1, Scalar(-100));
            for (int land = 0; land < landmarkNumber; land++)
            {
                double x = points.at<double>(land, 0);
                double y = points.at<double>(land + 19, 0);
                for (int r = -1 * steps / 2; r < steps / 2; r++)
                    for (int c = -1 * steps / 2; c < steps / 2; c++)
                    {
                        double xx = x + r * stepSize;
                        double yy = y + c * stepSize;
                        int ix = int(xx / stepSize);
                        int iy = int(yy / stepSize);
                        double posi = possibility[iy][ix][land];
                        //double posi = getposi(con[iy][ix][land]);
                        if (posi < 0.5) continue;
                        if (posi > MaxP.at<double>(land, 0))
                        {
                            MaxP.at<double>(land, 0) = posi;
                            landmarks.at<double>(land, 0) = xx;
                            landmarks.at<double>(land + landmarkNumber, 0) = yy;
                        }
                    }
            }
            ssm.update(landmarks);
            //ssm.update_noCons(landmarks);
        }
        else
        {
            double stageMinCost = minCost;
            MY_SSM ssmstage;
            ssmstage.copy(ssm);
            for (int land = 0; land < landmarkNumber; land++)
            {
                int len = CP[land].size();
                landmarks = ssmstage.get();
                double nowy = landmarks.at<double>(land, 0);
                double nowx = landmarks.at<double>(land + landmarkNumber, 0);
                for (int i = 0; i < len; i++)
                {
                    Point temP = CP[land][i];
                    if (Myabs(temP.x, nowx) > searchSize / 2 || Myabs(temP.y, nowy) > searchSize / 2) continue;

                    Mat landmarks1 = landmarks.clone();
                    landmarks1.at<double>(land, 0) = temP.y;
                    landmarks1.at<double>(land + 19, 0) = temP.x;

                    double cost = evaluation(landmarks1, ssmstage, possibility);
                    if (cost < stageMinCost)
                    {
                        stageMinCost = cost;
                        landmarks.at<double>(land, 0) = temP.y;
                        landmarks.at<double>(land + 19, 0) = temP.x;
                    }
                }
                ssmstage.update(landmarks);
            }
            ssm.copy(ssmstage);
        }
        double nowcost = Printevaluation(ssm.get(), ssm, possibility);


        double kk = nowcost;
        if (Myabs(nowcost, minCost) < 1e-6)
        {
            if (resultcost > minCost)
            {
                resultcost = minCost;
                ctt = 0;
                finalResult = ssm.get();
            }
            else
                ctt++;

            if (ctt > 5) break;
            //if (resultcost < 0.2) break;
            MY_SSM ssmc;
            ssmc.copy(ssm);
            ssmc.update(readjust(ssm, searchSize));
            kk = evaluation(ssmc.get(), ssmc, possibility);
            //if (kk > nowcost) break;
            ssm.copy(ssmc);
            cout << nowcost << " ---> " << kk << endl;
        }
        minCost = kk;
        //cout << "minCost" << minCost << endl;

        //ssm.drawImage(img,"shape",1);

    }
    cout << "number of iterations: " << t << endl;
    stop = time(NULL);
    cout << "Use Time: " << stop - start << endl;
    //ssm.drawImage(img,"shape",1);
    return finalResult;
}

void cot()
{
    Mat RC(38, 400, CV_64F);
    for (int i = 1; i <= 400; i++)
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
        Mat TP(38, 1, CV_64F);
        for (int j = 1; j <= 19; j++)
        {
            double x, y;
            char c;
            finn >> x >> c >> y;
            TP.at<double>(j - 1, 0) = x;
            TP.at<double>(j - 1 + 19, 0) = y;
        }
        finn.close();
        Mat dX = TP - getCentrem(TP) - M1.means;
        Mat P_1 = M1.P.inv();
        Mat db = P_1*dX;
        //cout << "dp: " << db << endl;
        //waitKey(0);
        for (int j = 1; j <= 38; j++)
        {
            RC.at<double>(j - 1, i - 1) = db.at<double>(j - 1, 0);
        }
    }
    for (int j = 1; j <= 38; j++)
    {
        Mat   img = RC.row(j - 1);
        //cout << "size: " << img.size() << endl;
        //cout << "img: " << img << endl;
        Mat     mean;
        Mat     stddev;
        meanStdDev(img, mean, stddev);
        double       mean_pxl = mean.at<double>(0, 0);
        double       stddev_pxl = stddev.at<double>(0, 0);
        cout << "evalue: " << sqrt(M1.eValues.at<double>(j - 1, 0)) << " mean: " << mean_pxl << " dev: " << stddev_pxl << " partition: " << (3 * stddev_pxl) / sqrt(M1.eValues.at<double>(j - 1, 0)) << endl;
    }
}

double cLen(Point a, Point b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
void testPoint(Mat landmarks, string truePoint, Mat testImage)
{
    Mat trueP = landmarks.clone();
    ifstream finnT(truePoint.c_str());
    Point poi[25];
    for (int i = 0; i < 19; i++)
    {
        double x, y, xx, yy;
        char c;
        finnT >> xx >> c >> yy;
        x = landmarks.at<double>(i, 0);
        y = landmarks.at<double>(i + landmarkNumber, 0);
        trueP.at<double>(i, 0) = xx;
        trueP.at<double>(i + landmarkNumber, 0) = yy;
        poi[i].x = xx;
        poi[i].y = yy;
        //finn.getline(ls, 100);
        //cout <<"kkk "<< ls<<endl;
        Point pt(x, y);
        Point pt2(xx, yy);
        double len = cLen(pt, pt2) * 0.1;
        cout << "len: " << len << endl;
        if (len < 1)
        {
            hist[1]++;
            phist[i][1]++;
        }
        if (len < 2)
        {
            hist[2]++;
            phist[i][2]++;
        }
        if (len < 3)
        {
            hist[3]++;
            phist[i][3]++;
        }
        if (len < 4)
        {
            hist[4]++;
            phist[i][4]++;
        }
        if (len < 5)
        {
            hist[5]++;
            phist[i][5]++;
        }
        hist[0]++;
        circle(testImage, pt2, 8, Scalar(0, 255, 0, 0), CV_FILLED, CV_AA, 0);
        circle(testImage, pt, 8, Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
        line(testImage, pt, pt2, Scalar(255, 0, 0, 0), 3);
    }
    cout << "ground truth" << endl;
    double kk = Printevaluation(trueP, M1, possibility);
    for (int land = 0; land < landmarkNumber; land++)
    {
        double x = trueP.at<double>(land, 0);
        double y = trueP.at<double>(land + 19, 0);
        int ix = int(x / stepSize);
        int iy = int(y / stepSize);
        cout << "posibility: " << land + 1 << " " << possibility[iy][ix][land] << endl;
    }
    /*
    cvLine(scr, poi[0], poi[3], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[3], poi[18], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[18], poi[9], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[9], poi[18], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[1], poi[0], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[1], poi[2], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[17], poi[2], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[14], poi[12], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[13], poi[12], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[13], poi[15], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[17], poi[4], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[4], poi[11], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[5], poi[10], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[5], poi[6], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[6], poi[8], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[7], poi[8], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[7], poi[9], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[10], poi[9], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[17], poi[16], cvScalar(0, 255, 0, 0), 2, 8, 0);
    cvLine(scr, poi[11], poi[16], cvScalar(0, 255, 0, 0), 2, 8, 0);*/


    /*for (int i = 1; i <= 5; i++)
    cout << "<" << i << " " << hist[i] << " " << double(hist[i]) / hist[0] << endl;
    cout << ">5 " << hist[0] << " " << double(hist[0]) / hist[0] << endl;
    cout << endl;

    for (int j = 1; j <= 19; j++)
    {
    for (int i = 1; i <= 5; i++)
    cout << phist[j][i] << " ";
    cout << endl;
    }*/
    finnT.close();
}

void test(MY_DNN dnn, MY_DNN dnnNG, MY_SSM ssm)
{
    int i = 0;
    double totp[30];
    int k = 0;
    for (int i = 381; i <= 400; i++)
    {
        char ind[100] = "";
        sprintf(ind, "%d", i);
        string testPath = ind;
        testPath = testPath + ".bmp";
        string truePoint = ind;
        truePoint = truePoint + ".txt";

        memset(possibility, 0, sizeof(possibility));
        memset(con, 0, sizeof(con));
        for (int j = 0; j < landmarkNumber; j++)
            CP[j].clear();

        Mat testImage = imread(testPath.c_str());
        Mat landmarks = searchLandmarks(dnn, dnnNG, ssm, testImage);
        totp[k++] = evaluation(landmarks, ssm, possibility);
        testPoint(landmarks, truePoint, testImage);
        /*namedWindow("test_demo", CV_WINDOW_NORMAL);
        imshow("test_demo", testImage);
        waitKey(0);*/
        string outImage = "out";
        outImage = outImage + ind + ".bmp";
        imwrite(outImage.c_str(), testImage);
    }
    for (int j = 0; j < k; j++)
        cout << j << " " << totp[j] << endl;

    for (int j = 0; j < 19; j++)
    {
        for (int i = 1; i <= 5; i++)
            cout << phist[j][i] * 1.0 / 20 << " ";
        cout << endl;
    }
    for (int i = 1; i <= 5; i++)
        cout << "<" << i << " " << hist[i] << " " << hist[i] * 1.0 / hist[0] << endl;
}

Mat landmarkDetect(Mat inputImage)
{
    //convolution(possibility,con);
    //cout << "see see: " <<con[1][2][3]<< endl;

    cot();
    //namedWindow("test_demo", CV_WINDOW_NORMAL);
    //imshow("test_demo", img);
    //setMouseCallback("test_demo", on_mouse, 0);
    //test(DN, DNNG, M1);
    img = inputImage.clone();
    Mat landmarks;
    landmarks = searchLandmarks(DN, DNNG, M1, img);
    //waitKey(0);
    return landmarks;
}
