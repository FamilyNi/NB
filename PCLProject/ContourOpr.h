#pragma once
#include "utils.h"

/*��ȡ����*/
template <typename T>
void ExtractContour(Mat &srcImg, vector<vector<T>> &contours, float lowVal, float highVal, int mode);

/*��������������*/
template <typename T1, typename T2>
void GetContourGravity(vector<T1> &contour, T2 &gravity);
template <typename T1, typename T2>
void GetIdxContourGravity(vector<T1>& contour, vector<int>& idxes, T2& gravity);

/*ƽ������*/
template <typename T1, typename T2>
void TranContour(vector<T1> &contour, T2 &gravity);

/*��������*/
template <typename T>
void GetMaxLenContuor(vector<vector<T>> &contours, int &maxLenIndex);

/*����������*/
template <typename T>
void GetMinLenContuor(vector<vector<T>> &contours, int &minLenIndex);

/*���ݳ���ѡ������*/
template <typename T>
void SelContourLen(vector<vector<T>> &contours, vector<vector<T>> &selContours, int minLen, int maxLen);

/*ѡ���Χ�����������*/
template <typename T>
void GetMaxAreaContour(vector<vector<T>> &contours, int &maxIndex);

/*ѡ���Χ�����С������*/
template <typename T>
void GetMinAreaContour(vector<vector<T>> &contours, int &minIndex);

/*�������ѡ������*/
template <typename T>
void SelContourArea(vector<vector<T>> &contours, vector<vector<T>> &selContours, int minArea, int maxArea);

/*�������*/
template <typename T>
void FillContour(Mat &srcImg, vector<T> &contour, Scalar color = Scalar(255));

/*����ν�������*/
template <typename T>
void PolyFitContour(vector<T> &contour, vector<T> &poly, double distThres = 5.0);

/*�ϲ�����*/
template <typename T>
void MergeContour(vector<vector<T>> &contours, vector<T> &contour);

/*����ƽ��*/
template <typename T1, typename T2>
void Img_SmoothContour(vector<T1>& srcContour, vector<T2>& dstContour, int size, double thresVal);


