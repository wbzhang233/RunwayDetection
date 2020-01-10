// 通用的一些函数
//　文件写入、快速排序等等
// Created by wbzhang on 2020/1/10.
//

#ifndef RUNWAYDETECTION_UTILITY_H
#define RUNWAYDETECTION_UTILITY_H

#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <map>
#include <fstream>

#define PI 3.141592653

using namespace std;
using namespace cv;

//10个颜色表
Scalar colorTab[]={
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

// 计算斜率
double getSlope(Vec4f line){
    return ( line(3)-line(1) ) /( line(2)-line(0) );
}

//　计算角度
template <typename _T> double getTheta(_T line){
    double theta=atan(getSlope(line));
    if(theta<0){
        return (theta+PI)*180/PI;
    }else return  theta*180/PI;
}

//判断k-means聚类是否有效,初值设为负一，看是否存在
bool getFlag(Mat label,int K){
    bool flag=true;
    for (int i = 0; i < label.rows; ++i) {
        for (int j = 0; j < label.cols; ++j) {
            int ldata=label.at<int >(i*label.cols+j);
            if (ldata==-1){
                flag = false;
            }
        }
    }
    return flag;
}

//统计角度出现的直方图
// 输入为弧度，(-pi/2,pi/2)
map<int,int>  getAngleHist(vector<double > vecSlope){
    // 初始化直方图map
    map<int,int> hist;
    for (int j = 0; j < 181; ++j) {
        hist.insert(pair<int,int>(j,0));
    }

    //设置map
    map<int,int>::iterator iter;
    for (int i = 0; i < vecSlope.size(); ++i) {
        int angle = round(vecSlope[i]);
        //查找map的key，并给其value加一
        iter = hist.find(angle);
        //添加断言
        assert(iter!=hist.end());
        if(iter!=hist.end()){
            iter->second++;
        }
    }
    return hist;
}

int getHistValue(map<int,int>::iterator iterator){
    return iterator->second;
}

// 快速排序
template<typename T>
void quicksort(T data[], int first, int last)
{
    int lower = first + 1;
    int upper = last;
    swap(data[first], data[(first + last) / 2]);
    T bound = data[first];
    while (lower <= upper)
    {
        while (data[lower] < bound)
            lower++;
        while (data[upper] > bound)
            upper--;
        if (lower < upper)
            swap(data[lower++], data[upper--]);
        else lower++;
    }
    swap(data[upper], data[first]);
    if (first < upper - 1)
        quicksort(data, first, upper - 1);
    if (upper + 1 < last)
        quicksort(data, upper + 1, last);
}

// 快速排序
template<class T>
void quicksort(T data[], int n)
{
    int i, max;
    if (n < 2)
        return;
    for (i = 1, max = 0; i < n; i++)
        if (data[max] < data[i])
            max = i;
    swap(data[n - 1], data[max]);
    quicksort(data, 0, n - 2);  //
}

//　打印数组
void PrintArray(int array[], int len)
{
    for (int i = 0; i < len; i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;
}

bool writeHistMap(map<int,int> hist,char* filename){
    fstream fp1(filename,ios::app| ios::in | ios::out);
    if(!fp1){
        cout << "open file failed" << endl;
        return 0;
    }
    map<int,int>::iterator iterh=hist.begin();
    while(iterh!=hist.end()){
        fp1<<iterh->first<<" "<<iterh->second<<endl;
        ++iterh;
    }
    fp1.close();
    cout<<"hist data saved..."<<endl;
    return true;
}

bool writeLinesData(vector<Vec4f> lines_std,char* filename){
    fstream fp(filename, ios::app | ios::in | ios::out);
    if (!fp) {
        cout << "open file failed" << endl;
        return 0;
    }
    vector<Vec4f>::iterator iter = lines_std.begin();
    while (iter != lines_std.end()) {
        Vec4f line = *iter;
        fp << line(0) / 2 + line(2) / 2 << "  " << line(1) / 2 + line(3) / 2 << "   " << getTheta(line) << endl;
        ++iter;
    }
    fp.close();
    cout << "lines data saved..." << endl;
    return true;
}


#endif //RUNWAYDETECTION_UTILITY_H
