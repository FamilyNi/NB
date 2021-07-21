#pragma once
#include "OPENCV_UTILS.h"

/*��ȡ����*/
void ExtractContour(Mat &srcImg, vector<vector<Point>> &contours, float lowVal, float highVal, int mode);

/*��������������*/
void GetContourGraviry(vector<Point2f> &contour, Point2f &gravity);

/*ƽ������*/
void TranContour(vector<Point2f> &contour, Point2f &gravity);

/*��������*/
void GetMaxLenContuor(vector<vector<Point>> &contours, int &maxLenIndex);

/*����������*/
void GetMinLenContuor(vector<vector<Point>> &contours, int &minLenIndex);

/*���ݳ���ѡ������*/
void SelContourLen(vector<vector<Point>> &contours, vector<vector<Point>> &selContours, int minLen, int maxLen);

/*ѡ���Χ�����������*/
void GetMaxAreaContour(vector<vector<Point>> &contours, int &maxIndex);

/*ѡ���Χ�����С������*/
void GetMinAreaContour(vector<vector<Point>> &contours, int &minIndex);

/*�������ѡ������*/
void SelContourArea(vector<vector<Point>> &contours, vector<vector<Point>> &selContours, int minArea, int maxArea);

/*�������*/
void FillContour(Mat &srcImg, vector<Point> &contour, Scalar color = Scalar(255));

/*����ν�������*/
void PolyFitContour(vector<Point> &contour, vector<Point> &poly, double distThres = 5.0);

/*�ϲ�����*/
void MergeContour(vector<vector<Point>> &contours, vector<Point> &contour);


