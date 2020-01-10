#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <map>
#include <set>

#define SAVE_RESULTS true
#define PI 3.141592653

using namespace std;
using namespace cv;

const char* result_winname="result";
const char* cluster_winname="clusterLines";
int framecount;
bool useRefine = false;
bool useCanny = true;
int scalepos=15;
int sizepos=0;
Mat image;
Mat src;

//10个颜色
Scalar colorTab[] =
{
    Scalar(0, 0, 255),
    Scalar(0, 255, 0),
    Scalar(255, 100, 100),
    Scalar(255, 0, 255),
    Scalar(0, 255, 255),
    Scalar(255, 0, 0),
    Scalar(255, 255, 0),
    Scalar(255, 0, 100),
    Scalar(100, 100, 100),
    Scalar(50, 125, 125)
};

//计算斜率
double getSlope(Vec4f line){
    return line(3)-line(1)/(line(2)-line(0) );
}

// 计算倾斜角 (-pi,pi)
double getTheta(Vec4f line){
    return atan( getSlope(line) );
}

//弧度转角度值，取整数
template <typename _T> _T radiusToAngle(_T theta){
    return theta*180/PI;
}

//判断k-means聚类是否有效
bool getFlag(Mat label,int K){
    bool flag=true;
    for (int i = 0; i < label.rows; ++i) {
        for (int j = 0; j < label.cols; ++j) {
            int ldata=label.at<int >(i*label.cols+j);
            if (ldata>K){
                flag = false;
            }
        }
    }
    return flag;
}


void lsdCallBack(int, void* usrdata)
{
    int prysize=pow(2,sizepos);
    Mat image_clone=Mat(Size(image.cols/prysize,image.rows/prysize),image.type());
    cout<<"result_size:"<<image_clone.cols/prysize<<","<<image_clone.rows<<endl;
    cv::resize(image,image_clone,image_clone.size(),BORDER_DEFAULT);

    Mat clusterLines;
    cv::resize(src,clusterLines,image_clone.size(),BORDER_DEFAULT);

    if (useCanny)
        Canny(image_clone, image_clone, 100, 200, 3);
    double scale=MIN((double)(scalepos+1)/20,1);
    cout<<"scale:"<<scale<<endl;

    //useRefine=!useRefine;
    //cout<<"Refine:"<<useRefine<<endl;

    Ptr<LineSegmentDetector> ls = useRefine ? createLineSegmentDetector(LSD_REFINE_STD,scale) : createLineSegmentDetector(LSD_REFINE_NONE,scale);

    vector<Vec4f> lines_std;

    double startTime = double(getTickCount());
    ls->detect(image_clone, lines_std);
    double duration_ms = (double(getTickCount()) - startTime) * 1000 / getTickFrequency();
    std::cout << "It took " << duration_ms << " ms." << std::endl<<std::endl;
    cout<<"lines_size:"<<lines_std.size()<<endl;

    /** 线段斜率聚类 **/
    // 1 将角度若干等分，并统计直方图
    // 2 寻找最大及最小的斜率值；将其氛围若干个阈值区间，统计每个阈值区间存在的个数（或者相当于统计直方图）
    //opencv k-means聚类算法接口
    Mat labelsP,labelsN;
    int K=3;
    if (lines_std.size() >= K){
        //vector<map<double,int> > slopeHist;
        // 将斜率按正负分成两个子集
        vector<double > vecSlopeP;
        vector<double > vecSlopeN;
        for (int i = 0; i < lines_std.size(); ++i) {
            double slope = getSlope(lines_std[i]);
            if (slope>=0){
                vecSlopeP.push_back(slope);
            }else{
                vecSlopeN.push_back(slope);
            }
        }

        // 将两个子集存入mat
        Mat sloptDataP = Mat(vecSlopeP.size(),1,CV_32FC1);
        Mat sloptDataN = Mat(vecSlopeN.size(),1,CV_32FC1);

        vector<double>::iterator iterP=vecSlopeP.begin();
        while (iterP!=vecSlopeP.end()){
            float slope = (double) *iterP;
            float *pdata = sloptDataP.ptr<float>(0);
            *pdata=slope;
            ++iterP;
        }

        vector<double>::iterator iterN=vecSlopeN.begin();
        while (iterN!=vecSlopeN.end()){
            float slope = (float) *iterN;
            float *pdata = sloptDataN.ptr<float>(0);
            *pdata=slope;
            ++iterN;
        }

        // K-Means方法聚类
        cv::kmeans(sloptDataP,K,labelsP,TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 0), 5, KMEANS_PP_CENTERS);
        cv::kmeans(sloptDataN,K,labelsN,TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 0), 5, KMEANS_PP_CENTERS);

        //如果labels中由任何数越界，表示聚类失败，返回
        assert(getFlag(labelsN,K) && getFlag(labelsP,K));
        if ( getFlag(labelsN,K) && getFlag(labelsP,K) ){
            cout<<"labelsP:"<<labelsP<<endl;
            //在原图中按照类别颜色绘制线段
            for (int i = 0; i < lines_std.size();++i)
            {
                Point startPt(lines_std[i](0),lines_std[i](1));
                Point endPt(lines_std[i](2),lines_std[i](3));
                if (labelsP.at<int>(i)<10){
                    line(clusterLines, startPt, endPt, colorTab[labelsP.at<int>(i)]);//标记像素点的类别，颜色区分
                }
            }
        }
        imshow(cluster_winname,clusterLines);
        char str[50];
        sprintf(str,"clusterLines/%d.png",framecount);
        cv::imwrite(str,clusterLines);

        /** 保存线段检测结果
        // Show found lines
        if (useCanny)
            image_clone = Scalar(0, 0, 255);

        ls->drawSegments(image_clone, lines_std);
        imshow(result_winname, image_clone);

        char scaleStr[10];
        sprintf(scaleStr,"%d-%.2f-",framecount,scale);
        string save_str="lsd-results/"+string(scaleStr)+"-"+to_string(prysize)+".jpg";
    //    save_str+=string(scaleStr);
    //    stringstream ss;
    //    ss<<scaleStr;
    //    save_str+=ss.str();
    //    save_str+= "-"+to_string(prysize)+".jpg";
        cv::imwrite(save_str,image_clone);**/
        }
}

int main(int argc,const char* argv[])
{
    //cout<<argc<<endl<<argv[0]<<"2"<<argv[1]<<endl;
    string default_path("/wbzhang/disk-d/Datasets/20200106plane/plane4");
    string root_path("/clusterLine2");
    string pathname = argc >=3 ? string(argv[1]) : default_path;
    string savepath = argc >=3 ? string(argv[1])+string(argv[2]) : default_path+root_path;
    framecount=0;
    while(framecount<290) {
        char filename[50];
        sprintf(filename, "%s/frame%04d.jpg", pathname.c_str(), framecount);
        cout << filename << endl;
        // Loads an image
        src= cv::imread(filename, IMREAD_REDUCED_COLOR_2);
        image = cv::imread(filename, IMREAD_REDUCED_GRAYSCALE_2);

        if (image.empty()) {
            cout << "Unable to load Image" << endl;
            return 1;
        }

/*        vector<Mat> pyrImages;
        buildPyramid(image, pyrImages, 4);
        for (auto img = pyrImages.begin(); img < pyrImages.end(); ++img) {
            cv::imwrite("pyr/" + to_string(img - pyrImages.begin()) + "pyr.png", *img);
        }*/
//        Mat pyr16 = imread("result/0.80--16.jpg");
//        Mat re16;
//        pyrUp(pyr16, re16, pyr16.size() * 2);//1/8
//        pyrUp(re16, re16, re16.size() * 2);//1/4
//        pyrUp(re16, re16, re16.size() * 2);//1/2
//        pyrUp(re16, re16, re16.size() * 2);//1
//        Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
//        filter2D(re16, re16, re16.depth(), kernel);
//        namedWindow("re16", WINDOW_AUTOSIZE);
//        cv::imshow("re16", re16);
        /**highgui for tuning**/
        namedWindow(result_winname, WINDOW_AUTOSIZE);
        imshow(result_winname, image);
        createTrackbar("scale", result_winname, &scalepos, 20, lsdCallBack);
        createTrackbar("size", result_winname, &sizepos, 4, lsdCallBack);
        lsdCallBack(0, 0);

        waitKey(10);
        ++framecount;
    }
    destroyAllWindows();
    return 0;
}
