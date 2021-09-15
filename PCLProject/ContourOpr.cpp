#include "ContourOpr.h"
#include <opencv2/flann.hpp>

//提取轮廓======================================================================
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

//计算轮廓的重心================================================================
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

//平移轮廓======================================================================
void TranContour(vector<Point2f> &contour, Point2f &gravity)
{
	for (int i = 0; i < contour.size(); ++i)
	{
		contour[i].x += gravity.x;
		contour[i].y += gravity.y;
	}
}
//==============================================================================

//获得最长轮廓==================================================================
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

//获得最短轮廓==================================================================
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

//根据长度选择轮廓==============================================================
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

//选择包围面积最大的轮廓========================================================
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

//选择包围面积最小的轮廓========================================================
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

//根据面积选择轮廓==============================================================
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

//填充轮廓======================================================================
void FillContour(Mat &srcImg, vector<Point> &contour, Scalar color)
{
	if (srcImg.empty() || contour.size() == 0)
		return;
	fillPoly(srcImg, contour, color);
}
//==============================================================================

//多边形近似轮廓================================================================
void PolyFitContour(vector<Point> &contour, vector<Point> &poly, double distThres)
{
	poly.resize(0);
	if (contour.size() == 0)
		return;
	approxPolyDP(contour, poly, distThres, false);
}
//==============================================================================

//合并轮廓======================================================================
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

//轮廓平滑======================================================================
void SmoothContour(vector<cv::Point2f>& srcContour, vector<cv::Point2f>& dstContour, int size, double thresVal)
{
	dstContour.resize(srcContour.size());
	cv::Mat source = cv::Mat(srcContour).reshape(1);
	cv::flann::KDTreeIndexParams indexParams(2);
	cv::flann::Index kdtree(source, indexParams);
	for (int i = 0; i < srcContour.size(); ++i)
	{
		vector<float> vecQuery(2);//存放查询点
		vecQuery[0] = srcContour[i].x; //查询点x坐标
		vecQuery[1] = srcContour[i].y; //查询点y坐标
		vector<int> vecIndex(size);//存放返回的点索引
		vector<float> vecDist(size);//存放距离
		cv::flann::SearchParams params(32);//设置knnSearch搜索参数
		kdtree.knnSearch(vecQuery, vecIndex, vecDist, size, params);
		vector<cv::Point2f> pts_(size);
		for (int j = 0; j < size; ++j)
		{
			pts_[j] = srcContour[vecIndex[j]];
		}
		cv::Vec4d line;
		cv::fitLine(pts_, line, cv::DIST_L2, 0, 0.01, 0.01);
		float scale = srcContour[i].x * line[0] + srcContour[i].y * line[1] - (line[2] * line[0] + line[3] * line[1]);
		cv::Point2f v_p;
		v_p.x = line[2] + scale * line[0]; v_p.y = line[3] + scale * line[1];
		double dist = std::powf(v_p.x - srcContour[i].x, 2) + std::powf(v_p.y - srcContour[i].y, 2);
		if (dist > thresVal)
			dstContour[i] = v_p;
		else
			dstContour[i] = srcContour[i];
	}
}
//==============================================================================