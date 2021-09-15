#pragma once
#include "utils.h"

/*说明：
	直线方程： x = a * t + x0; y = b * t + y0; z = c * t + z0
	pts：[in]空间中的点簇
	line：---[0]：a、[1]：b、[2]：c、[3]：x0、[4]：y0、[5]：z0
*/

/*空间两点求直线：
	pt1、pt2：[in]空间中的任意两点
	line：[out]
*/
template <typename T>
void PC_OLSFit3DLine(T& pt1, T& pt2, cv::Vec6d& line);

/*最小二乘法拟合空间直线：
	weights：[in]权重
	line：[out]
*/
template <typename T>
void PC_OLSFit3DLine(vector<T>& pts, vector<double>& weights, cv::Vec6d& line);

/*Huber计算权重：
	line：[in]
	weights：[out]权重
*/
template <typename T>
void PC_Huber3DLineWeights(vector<T>& pts, cv::Vec6d& line, vector<double>& weights);

/*Turkey计算权重：
	line：[in]
	weights：[out]权重
*/
template <typename T>
void PC_Turkey3DLineWeights(vector<T>& pts, cv::Vec6d& line, vector<double>& weights);

/*空间直线拟合：
	line：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
template <typename T>
void PC_Fit3DLine(vector<T>& pts, cv::Vec6d& line, int k, NB_MODEL_FIT_METHOD method);


void PC_3DLineTest();