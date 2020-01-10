//
// Created by wbzhang on 2020/1/6.
//


#include "../include/utility.h"

#define LOOP true //未能读到图片后重新读取第一张图片

const char* result_winname="result";
const char* cluster_winname="clusterLines";
int framecount;
bool useRefine = false;
bool useCanny = true;
Mat image;
Mat src;


int main(int argc, char* argv[])
{
    string default_path = "/wbzhang/disk-d/Datasets/20200106plane/plane4";
    string default_savepath = "Results/clusterLine2";
    string path = argc >=3 ? argv[1] : default_path;
    string savepath = argc >=3 ? string(argv[1])+string(argv[2]) : default_path+default_savepath;
    framecount=1;

    while(framecount<290) {
        char filename[50];
        sprintf(filename, "%s/frame%04d.jpg", path.c_str(), framecount);
        //sprintf(filename, "%s","remote_macau.png");

        cout << filename << endl;
        // Loads an image
        src= cv::imread(filename, IMREAD_REDUCED_COLOR_2);
        image = cv::imread(filename, IMREAD_REDUCED_GRAYSCALE_2);

        if(image.empty()){
            cout << "Unable to load Image" << endl;
            if(LOOP){
                framecount=0;
                continue;
            }else return 1;
        }

        int prysize=1;
        Mat image_clone=Mat(Size(image.cols/prysize,image.rows/prysize),image.type());
        cout<<"result_size:"<<image_clone.cols/prysize<<","<<image_clone.rows<<endl;
        cv::resize(image,image_clone,image_clone.size(),BORDER_DEFAULT);

        Mat clusterLines;
        cv::resize(src,clusterLines,image_clone.size(),BORDER_DEFAULT);


        // lsd　线段检测
        if (useCanny)
            Canny(image_clone, image_clone, 100, 200, 3);
        double scale=0.8;
        //cout<<"scale:"<<scale<<endl;
        cout<<"Refine:"<<useRefine<<endl;

        Ptr<cv::LineSegmentDetector> ls = useRefine ? createLineSegmentDetector(LSD_REFINE_STD,scale) : createLineSegmentDetector(LSD_REFINE_NONE,scale);

        vector<Vec4f> lines_std;

        double startTime = double(getTickCount());
        ls->detect(image_clone, lines_std);
        double duration_ms = (double(getTickCount()) - startTime) * 1000 / getTickFrequency();
        std::cout << "It took " << duration_ms << " ms." << std::endl;
        cout<<"lines_size:"<<lines_std.size()<<endl;

        // 绘制lsd　检测出的所有线段
        Mat drawLines=src.clone();
        ls->drawSegments(drawLines,lines_std);
        imshow("drawLines",drawLines);
        char txString[50];
        sprintf(txString,"Results/drawLines/%d.png",framecount);
        cv::imwrite(txString,drawLines);

        //保存检测出的线段信息
        if(framecount==278) {
            writeLinesData(lines_std,"/home/wbzhang/Code/RunwayDetection/data/lines278.txt");
        }

        /** [1] 根据直方图来判断跑道线的范围
        vector<double > vecSlope;
        for (int i = 0; i < lines_std.size(); ++i) {
            float slope = getTheta(lines_std[i]);
            vecSlope.push_back(slope);
        }
        //获取直方图
        map<int,int> hist=getAngleHist(vecSlope);
        //保存直方图
        if(framecount==278){
            writeHistMap(hist,"/home/wbzhang/Code/RunwayDetection/data/hist278.txt");
        }

        //获取直方图中值最大的三个数，并用前三种颜色进行绘制
        int array[181];
        for (int k = 0; k < sizeof(array)/4; ++k) {
            array[k]=0;
        }
        map<int ,int >::iterator histIter;
        for (int j = 0; j < 181; ++j) {
            histIter=hist.find(j);
            array[j]=histIter->second;
        }
        quicksort<int >(array,0,181);

        // 查找三个类对应的key，即角度范围存在最多的几个角度范围
        int maxkey;
        for(map<int,int>::iterator it=hist.begin();it!=hist.end();++it){
            if(it->second==array[180]){
               maxkey=it->first;//假设最多的角度范围恰巧仅有一个，仅仅返回第一个
               it=hist.end();
            }
        }

        //对类别最多的线段集合　按位置k-means聚类，求最大类的中心
        vector<Vec4f> mostLines;
        Mat posedata=Mat::zeros(hist[maxkey],2,CV_32FC1);
        int pcount=0;
        for (int l = 0; l < lines_std.size(); ++l) {
            if(round(getTheta(lines_std[l]) ) == maxkey){
                mostLines.push_back(lines_std[l]);
                posedata.at<float>(pcount,0)=( lines_std[l](0)+lines_std[l](2) )/2;
                posedata.at<float>(pcount,1)=( lines_std[l](1)+lines_std[l](3) )/2;
                pcount++;
            }
        }
        Mat labels=Mat::zeros(hist[maxkey],1,CV_8UC1);
        cv::kmeans(posedata,2,labels,TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 1, 0), 5, KMEANS_PP_CENTERS);

        //随后从原始的弧度矢量中进行筛选，符合此三个范围则进行相应的绘制
        for (int i = 0; i < mostLines.size();++i)
        {
            Point startPt(mostLines[i](0),mostLines[i](1));
            Point endPt(mostLines[i](2),mostLines[i](3));
                line(clusterLines, startPt, endPt, colorTab[labels.at<int>(i)]);//标记像素点的类别，颜色区分
                char text[50];
                sprintf(text,"%.2d",getTheta(lines_std[i]));
                putText(clusterLines,text,startPt,0.5,FONT_HERSHEY_SIMPLEX ,colorTab[labels.at<int>(i)]);
        }
        imshow(cluster_winname,clusterLines);
        char namestring[50];
        sprintf(namestring,"Results/Mostlines/%d.png",framecount);
        cv::imwrite(namestring,clusterLines);
        cout<<"write results done..."<<endl<<endl;**/

        /** [2] 线段斜率 k-means聚类
        // 1 将角度若干等分，并统计直方图
        // 2 寻找最大及最小的斜率值；将其氛围若干个阈值区间，统计每个阈值区间存在的个数

        //opencv k-means聚类算法
        int K=5;
        Mat labels=Mat::zeros(lines_std.size(),1,CV_8UC1);
        for (int j = 0; j < labels.rows; ++j) {
            for (int i = 0; i < labels.cols; ++i) {
                labels.at<int>(j,i)=255;
            }
        }

        if (lines_std.size() > K){
            Mat slopeData = Mat::zeros(lines_std.size(),1,CV_32FC1);
            //vector<map<double,int> > slopeHist;
            vector<double > vecSlope;
            for (int i = 0; i < lines_std.size(); ++i) {
                float slope = getTheta(lines_std[i]);
                vecSlope.push_back(slope);
//                float *pdata0 = slopeData.ptr<float>(0);
//                *pdata0=lines_std[i](2)/2+lines_std[i](0)/2;
//                float *pdata1 = slopeData.ptr<float>(1);
//                *pdata1=lines_std[i](3)/2+lines_std[i](1)/2;
                float *pdata2 = slopeData.ptr<float>(0);
                *pdata2=slope;
            }

            //获取直方图
            map<int,int> hist=getAngleHist(vecSlope);
            if(framecount==200){
                //保存直方图
                fstream fp1("/home/wbzhang/Code/RunwayDetection/data/hist.txt",ios::app| ios::in | ios::out);
                if(!fp1){
                    cout << "open file failed" << endl;
                    return 0;
                }
                map<int,int>::iterator iter=hist.begin();
                while(iter!=hist.end()){
                    fp1<<iter->first<<" "<<iter->second<<endl;
                    ++iter;
                }
                fp1.close();
                cout<<"hist data saved..."<<endl;
            }

            //double startTimeK = double(getTickCount());
            cv::kmeans(slopeData,K,labels,TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 0), 5, KMEANS_PP_CENTERS);
            //std::cout << "Kmeans took " <<(double(getTickCount()) - startTimeK) * 1000 / getTickFrequency()<< " ms." << std::endl;

            cout<<"labels"<<labels<<endl;
            assert(getFlag(labels,K) );
            if(getFlag(labels,K)){
                //在原图中按照类别颜色绘制线段
                for (int i = 0; i < lines_std.size();++i)
                {
                    Point startPt(lines_std[i](0),lines_std[i](1));
                    Point endPt(lines_std[i](2),lines_std[i](3));
                    //cout<<"labels:"<<labels.at<int>(i)<<endl;
                    if (labels.at<int>(i)<10) {
                        line(clusterLines, startPt, endPt, colorTab[labels.at<int>(i)]);//标记像素点的类别，颜色区分
                    }
                }
            }
        }
        imshow(cluster_winname,clusterLines);
        char str[50];
        sprintf(str,"%s/%d.png",savepath.c_str(),framecount);
        cv::imwrite(str,clusterLines);
        cout<<"saved results..."<<endl<<std::endl;**/

        waitKey(10);
        ++framecount;
    }
    destroyAllWindows();
    return 0;
}
