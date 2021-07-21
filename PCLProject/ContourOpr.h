#pragma once
#include "OPENCV_UTILS.h"

/*提取轮廓*/
void ExtractContour(Mat &srcImg, vector<vector<Point>> &contours, float lowVal, float highVal, int mode);

/*计算轮廓的重心*/
void GetContourGraviry(vector<Point2f> &contour, Point2f &gravity);

/*平移轮廓*/
void TranContour(vector<Point2f> &contour, Point2f &gravity);

/*获得最长轮廓*/
void GetMaxLenContuor(vector<vector<Point>> &contours, int &maxLenIndex);

/*获得最短轮廓*/
void GetMinLenContuor(vector<vector<Point>> &contours, int &minLenIndex);

/*根据长度选择轮廓*/
void SelContourLen(vector<vector<Point>> &contours, vector<vector<Point>> &selContours, int minLen, int maxLen);

/*选择包围面积最大的轮廓*/
void GetMaxAreaContour(vector<vector<Point>> &contours, int &maxIndex);

/*选择包围面积最小的轮廓*/
void GetMinAreaContour(vector<vector<Point>> &contours, int &minIndex);

/*根据面积选择轮廓*/
void SelContourArea(vector<vector<Point>> &contours, vector<vector<Point>> &selContours, int minArea, int maxArea);

/*填充轮廓*/
void FillContour(Mat &srcImg, vector<Point> &contour, Scalar color = Scalar(255));

/*多边形近似轮廓*/
void PolyFitContour(vector<Point> &contour, vector<Point> &poly, double distThres = 5.0);

/*合并轮廓*/
void MergeContour(vector<vector<Point>> &contours, vector<Point> &contour);


