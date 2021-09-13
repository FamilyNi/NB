#pragma once
#include "utils.h"
#include "OpenCV_Utils.h"

/*两点计算直线：
	pt1、pt2：[in]平面上的两个点
	line：[out]输出的直线---[0]：x方向上法向量、
				[1]：y方向的法向量、[2]：c值
*/
template <typename T>
void Img_TwoPtsComputeLine(T& pt1, T& pt2, cv::Vec3d& line);

/*随机一致采样算法计算直线
	pts：[in]平面上的点簇
	line：[out]输出的圆---[0]：x方向上法向量、
				[1]：y方向的法向量、[2]：c值
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
template <typename T>
void Img_RANSACComputeLine(vector<T>& pts, cv::Vec3d& line, vector<T>& inlinerPts, double thres);

/*Turkey直线拟合*/
void Img_TurkeyFitLine(vector<cv::Point>& pts, cv::Vec3d& line, int k, double thres);

void LineTest();
