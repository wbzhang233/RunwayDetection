//
// Created by wbzhang on 2019/12/16.
//

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

Mat dst, cdst, cdstP;
int framecount;

int thresholdS=150;
void callbackOnHoughline(int,void *)
{
    Mat drawCdst=cdst.clone();
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(dst, lines, 1, CV_PI/180, thresholdS, 0, 0 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( drawCdst, pt1, pt2, Scalar(0,0,255), 1, LINE_AA);
    }
    imshow("Detected Lines (in red) - Standard Hough Line Transform", drawCdst);
    char str[50];
    sprintf(str,"hough-result/%d.png",framecount);
    cv::imwrite(str,drawCdst);
}

int thresholdP=50;
void callbackOnHoughlineP(int,void *)
{
    Mat drawCdstP=cdstP.clone();
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI/180, thresholdP, 50, 10 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        line( drawCdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
    }
    imshow("Detected Lines (in red) - Probabilistic Line Transform", drawCdstP);

}


int main(int argc, char** argv)
{
    const char* default_path = "/wbzhang/disk-d/Datasets/20200106plane/plane4";
    const char* pathname = argc >=2 ? argv[1] : default_path;
    framecount=0;
    while(framecount<290){
        char filename[50];
        sprintf(filename,"%s/frame%04d.jpg",pathname,framecount);
        cout<<filename<<endl;
        // Loads an image
        Mat src = cv::imread( filename, IMREAD_REDUCED_GRAYSCALE_2);

        // Check if image is loaded fine
        if(src.empty()){
            printf(" Error opening image\n");
            waitKey();
            return -1;
        }
        // Edge detection
        Canny(src, dst, 50, 200, 3);

        // Copy edges to the images that will display the results in BGR
        cvtColor(dst, cdst, COLOR_GRAY2BGR);
        cdstP = cdst.clone();

        // Standard Hough Line Transform
        const char* winname1="Detected Lines (in red) - Standard Hough Line Transform";
        namedWindow("Detected Lines (in red) - Standard Hough Line Transform",WINDOW_AUTOSIZE);
        createTrackbar("threshold:",winname1,&thresholdS,255,callbackOnHoughline);
        callbackOnHoughline(150,0);

        // Probabilistic Line Transform
        const char* winname2="Detected Lines (in red) - Probabilistic Line Transform";
        namedWindow(winname2,WINDOW_AUTOSIZE);
        createTrackbar("threshold",winname2,&thresholdP,255,callbackOnHoughlineP);
        callbackOnHoughlineP(50,0);

        // Show results
//        imshow("Source", src);
//        imwrite("src.png",src);
        // Wait and Exit
        waitKey(10);
        ++framecount;
    }
    cout<<"All Images DONE..."<<endl;
    waitKey();
    return 0;
}
