#include "ContourOpr.h"
#include <opencv2/flann.hpp>
#include "MathOpr.h"

//提取轮廓======================================================================
template <typename T>
void ExtractContour(Mat &srcImg, vector<vector<T>> &contours, float lowVal, float highVal, int mode)
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
template <typename T1, typename T2>
void GetContourGravity(vector<T1> &contour, T2 &gravity)
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
template <typename T1, typename T2>
void GetIdxContourGravity(vector<T1>& contour, vector<int>& idxes, T2& gravity)
{
	int len = idxes.size();
	if (len == 0)
		return;
	float sum_x = 0.0f, sum_y = 0.0f;
	for (int i = 0; i < len; ++i)
	{
		sum_x += contour[idxes[i]].x;
		sum_y += contour[idxes[i]].y;
	}
	gravity.x = sum_x / len;
	gravity.y = sum_y / len;
}
//==============================================================================

//平移轮廓======================================================================
template <typename T1, typename T2>
void TranContour(vector<T1> &contour, T2 &gravity)
{
	for (int i = 0; i < contour.size(); ++i)
	{
		contour[i].x += gravity.x;
		contour[i].y += gravity.y;
	}
}
//==============================================================================

//获得最长轮廓==================================================================
template <typename T>
void GetMaxLenContuor(vector<vector<T>> &contours, int &maxLenIndex)
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
template <typename T>
void GetMinLenContuor(vector<vector<T>> &contours, int &minLenIndex)
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
template <typename T>
void SelContourLen(vector<vector<T>> &contours, vector<vector<T>> &selContours, int minLen, int maxLen)
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
template <typename T>
void GetMaxAreaContour(vector<vector<T>> &contours, int &maxIndex)
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
template <typename T>
void GetMinAreaContour(vector<vector<T>> &contours, int &minIndex)
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
template <typename T>
void SelContourArea(vector<vector<T>> &contours, vector<vector<T>> &selContours, int minArea, int maxArea)
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
template <typename T>
void FillContour(Mat &srcImg, vector<T> &contour, Scalar color)
{
	if (srcImg.empty() || contour.size() == 0)
		return;
	fillPoly(srcImg, contour, color);
}
//==============================================================================

//多边形近似轮廓================================================================
template <typename T>
void PolyFitContour(vector<T> &contour, vector<T> &poly, double distThres)
{
	poly.resize(0);
	if (contour.size() == 0)
		return;
	approxPolyDP(contour, poly, distThres, false);
}
//==============================================================================

//合并轮廓======================================================================
template <typename T>
void MergeContour(vector<vector<T>> &contours, vector<T> &contour)
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
template <typename T1, typename T2>
void Img_SmoothContour(vector<T1>& srcContour, vector<T2>& dstContour, int size, double thresVal)
{
	int pts_num = srcContour.size();
	dstContour.resize(pts_num);
	cv::Mat source(cv::Size(2, pts_num), CV_32FC1, cv::Scalar(0));
	for (int i = 0; i < pts_num; ++i)
	{
		float *pSource = source.ptr<float>(i);
		pSource[0] = srcContour[i].x; pSource[1] = srcContour[i].y;
	}
	cv::flann::KDTreeIndexParams indexParams(2);
	cv::flann::Index kdtree(source, indexParams);

	vector<T1> pts_(size);
	vector<float> vecQuery(2);//存放查询点
	vector<int> vecIndex(size);//存放返回的点索引
	vector<float> vecDist(size);//存放距离
	cv::Vec4d line;
	for (int i = 0; i < srcContour.size(); ++i)
	{
		vecQuery[0] = srcContour[i].x; //查询点x坐标
		vecQuery[1] = srcContour[i].y; //查询点y坐标
		kdtree.knnSearch(vecQuery, vecIndex, vecDist, size);
		for (int j = 0; j < size; ++j)
		{
			pts_[j] = srcContour[vecIndex[j]];
		}
		cv::fitLine(pts_, line, cv::DIST_L2, 0, 0.01, 0.01);
		T2 v_p;
		Img_PtProjLinePt(srcContour[i], line, v_p);
		double dist = std::powf(v_p.x - srcContour[i].x, 2) + std::powf(v_p.y - srcContour[i].y, 2);
		dstContour[i] = dist > thresVal ? v_p : srcContour[i];
	}
}
//==============================================================================