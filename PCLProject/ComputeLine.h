#pragma once
#include "utils.h"
#include "OpenCV_Utils.h"

/*说明：
	直线方程： a * x + b * y + c = 0
	pts：[in]平面上的点簇
	line：---[0]：a、[1]：b、[2]：c
*/

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

/*最小二乘法拟合直线：
	weights：[in]权重
	line：[out]
*/
template <typename T>
void Img_OLSFitLine(vector<T>& pts, vector<double>& weights, cv::Vec3d& line);

/*Huber计算权重：
	line：[in]
	weights：[out]权重
*/
template <typename T>
void Img_HuberLineWeights(vector<T>& pts, cv::Vec3d& line, vector<double>& weights);

/*Turkey计算权重：
	line：[in]
	weights：[out]权重
*/
template <typename T>
void Img_TurkeyLineWeights(vector<T>& pts, cv::Vec3d& line, vector<double>& weights);

/*直线拟合*/
template <typename T>
void Img_FitLine(vector<T>& pts, cv::Vec3d& line, int k, NB_MODEL_FIT_METHOD method);

void LineTest();
