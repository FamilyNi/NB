#include "ContourOpr.h"

//��ȡ����======================================================================
void ExtractContour(Mat &srcImg, vector<vector<Point>> &contours, float lowVal, float highVal, int mode)
{
	Mat dstImg = Mat(srcImg.size(), srcImg.type(), Scalar(0));
	if (mode == 0)
		threshold(srcImg, dstImg, lowVal, 255, THRESH_BINARY);
	if (mode == 1)
		threshold(srcImg, dstImg, 0, 255, THRESH_OTSU);
	if (mode == 2)
		Canny(srcImg, dstImg, lowVal, highVal);
	contours.resize(0);
	findContours(dstImg, contours, RETR_LIST, CHAIN_APPROX_NONE);
}
//==============================================================================

//��������������================================================================
void GetContourGravity(vector<Point2f> &contour, Point2f &gravity)
{
	int len = contour.size();
	if (len == 0)
		return;
	float sum_x = 0.0f, sum_y = 0.0f;
	for (int i = 0; i < len; ++i)
	{
		sum_x += contour[i].x;
		sum_y += contour[i].y;
	}
	gravity.x = sum_x / len;
	gravity.y = sum_y / len;
}
//==============================================================================

//ƽ������======================================================================
void TranContour(vector<Point2f> &contour, Point2f &gravity)
{
	for (int i = 0; i < contour.size(); ++i)
	{
		contour[i].x += gravity.x;
		contour[i].y += gravity.y;
	}
}
//==============================================================================

//��������==================================================================
void GetMaxLenContuor(vector<vector<Point>> &contours, int &maxLenIndex)
{
	int len = contours.size();
	if (len == 0)
		return;
	int maxLen = contours[0].size();
	for (int i = 1; i < len; ++i)
	{
		if (maxLen < contours[i].size())
		{
			maxLen = contours[i].size();
			maxLenIndex = i;
		}
	}
}
//==============================================================================

//����������==================================================================
void GetMinLenContuor(vector<vector<Point>> &contours, int &minLenIndex)
{
	int len = contours.size();
	if (len == 0)
		return;
	int maxLen = contours[0].size();
	for (int i = 1; i < len; ++i)
	{
		if (maxLen > contours[i].size())
		{
			maxLen = contours[i].size();
			minLenIndex = i;
		}
	}
}
//==============================================================================

//���ݳ���ѡ������==============================================================
void SelContourLen(vector<vector<Point>> &contours, vector<vector<Point>> &selContours, int minLen, int maxLen)
{
	selContours.resize(0);
	if (contours.size() == 0)
		return;
	for (int i = 0; i < contours.size(); ++i)
	{
		if (contours[i].size() > minLen && contours[i].size() < maxLen)
			selContours.push_back(contours[i]);
	}
}
//==============================================================================

//ѡ���Χ�����������========================================================
void GetMaxAreaContour(vector<vector<Point>> &contours, int &maxIndex)
{
	int len = contours.size();
	if (len == 0)
		return;
	double maxArea = contourArea(contours[0]);
	for (int i = 1; i < len; ++i)
	{
		double area = contourArea(contours[i]);
		if (maxArea < area)
		{
			maxArea = area;
			maxIndex = i;
		}
	}
}
//==============================================================================

//ѡ���Χ�����С������========================================================
void GetMinAreaContour(vector<vector<Point>> &contours, int &minIndex)
{
	int len = contours.size();
	if (len == 0)
		return;
	double minArea = contourArea(contours[0]);
	for (int i = 1; i < len; ++i)
	{
		double area = contourArea(contours[i]);
		if (minArea > area)
		{
			minArea = area;
			minIndex = i;
		}
	}
}
//==============================================================================

//�������ѡ������==============================================================
void SelContourArea(vector<vector<Point>> &contours, vector<vector<Point>> &selContours, int minArea, int maxArea)
{
	selContours.resize(0);
	if (contours.size() == 0)
		return;
	for (int i = 0; i < contours.size(); ++i)
	{
		double area = contourArea(contours[i]);
		if (area > minArea && area < maxArea)
			selContours.push_back(contours[i]);
	}
}
//==============================================================================

//�������======================================================================
void FillContour(Mat &srcImg, vector<Point> &contour, Scalar color)
{
	if (srcImg.empty() || contour.size() == 0)
		return;
	fillPoly(srcImg, contour, color);
}
//==============================================================================

//����ν�������================================================================
void PolyFitContour(vector<Point> &contour, vector<Point> &poly, double distThres)
{
	poly.resize(0);
	if (contour.size() == 0)
		return;
	approxPolyDP(contour, poly, distThres, false);
}
//==============================================================================

//�ϲ�����======================================================================
void MergeContour(vector<vector<Point>> &contours, vector<Point> &contour)
{
	contour.resize(0);
	if (contours.size() == 0)
		return;
	for (int i = 0; i < contours.size(); i++)
	{
		contour.insert(contour.end(), contours[i].begin(), contours[i].end());
	}
}
//==============================================================================