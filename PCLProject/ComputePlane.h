#pragma once
#include "utils.h"

/*说明：
	平面方程：a * x + b * y + c * z + d = 0
	pts：[in]空间中的点簇
	plane：平面---[0]：a、[1]：b、[2]：c、[3]：d
*/

/*三点计算平面：
	pt1、pt2、pt3：[in]空间中的三个点
	plane：[out]
*/
template <typename T>
void PC_ThreePtsComputePlane(T& pt1, T& pt2, T& pt3, cv::Vec4d& plane);

/*最小二乘法拟合平面：
	weights：[in]权重
	plane：[out]
*/
template <typename T>
void PC_OLSFitPlane(vector<T>& pts, vector<double>& weights, cv::Vec4d& plane);

/*huber计算权重：
	plane：[in]
	weights：[out]权重
*/
template <typename T>
void PC_HuberPlaneWeights(vector<T>& pts, cv::Vec4d& plane, vector<double>& weights);

/*Turkey计算权重：
	plane：[in]
	weights：[out]权重
*/
template <typename T>
void PC_TurkeyPlaneWeights(vector<T>& pts, cv::Vec4d& plane, vector<double>& weights);

/*平面拟合：
	plane：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
template <typename T>
void PC_FitPlane(vector<T>& pts, cv::Vec4d& plane, int k, NB_MODEL_FIT_METHOD method);


void PC_PlaneTest();