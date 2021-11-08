#pragma once
#include "utils.h"

/*提取轮廓*/
template <typename T>
void ExtractContour(Mat &srcImg, vector<vector<T>> &contours, float lowVal, float highVal, int mode);

/*计算轮廓的重心*/
template <typename T1, typename T2>
void GetContourGravity(vector<T1> &contour, T2 &gravity);
template <typename T1, typename T2>
void GetIdxContourGravity(vector<T1>& contour, vector<int>& idxes, T2& gravity);

/*平移轮廓*/
template <typename T1, typename T2>
void TranContour(vector<T1> &contour, T2 &gravity);

/*获得最长轮廓*/
template <typename T>
void GetMaxLenContuor(vector<vector<T>> &contours, int &maxLenIndex);

/*获得最短轮廓*/
template <typename T>
void GetMinLenContuor(vector<vector<T>> &contours, int &minLenIndex);

/*根据长度选择轮廓*/
template <typename T>
void SelContourLen(vector<vector<T>> &contours, vector<vector<T>> &selContours, int minLen, int maxLen);

/*选择包围面积最大的轮廓*/
template <typename T>
void GetMaxAreaContour(vector<vector<T>> &contours, int &maxIndex);

/*选择包围面积最小的轮廓*/
template <typename T>
void GetMinAreaContour(vector<vector<T>> &contours, int &minIndex);

/*根据面积选择轮廓*/
template <typename T>
void SelContourArea(vector<vector<T>> &contours, vector<vector<T>> &selContours, int minArea, int maxArea);

/*填充轮廓*/
template <typename T>
void FillContour(Mat &srcImg, vector<T> &contour, Scalar color = Scalar(255));

/*多边形近似轮廓*/
template <typename T>
void PolyFitContour(vector<T> &contour, vector<T> &poly, double distThres = 5.0);

/*合并轮廓*/
template <typename T>
void MergeContour(vector<vector<T>> &contours, vector<T> &contour);

/*轮廓平滑*/
template <typename T1, typename T2>
void Img_SmoothContour(vector<T1>& srcContour, vector<T2>& dstContour, int size, double thresVal);


