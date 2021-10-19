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
template <typename T1, typename T2>
void PC_ThreePtsComputePlane(T1& pt1, T1& pt2, T1& pt3, T2& plane);

/*随机一致采样算法计算平面：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
template <typename T1, typename T2>
void PC_RANSACComputePlane(vector<T1>& pts, T2& plane, vector<T1>& inlinerPts, double thres);

/*最小二乘法拟合平面：
	weights：[in]权重
	plane：[out]
*/
template <typename T1, typename T2>
void PC_OLSFitPlane(vector<T1>& pts, vector<double>& weights, T2& plane);

/*huber计算权重：
	plane：[in]
	weights：[out]权重
*/
template <typename T1, typename T2>
void PC_HuberPlaneWeights(vector<T1>& pts, T2& plane, vector<double>& weights);

/*Tukey计算权重：
	plane：[in]
	weights：[out]权重
*/
template <typename T1, typename T2>
void PC_TukeyPlaneWeights(vector<T1>& pts, T2& plane, vector<double>& weights);

/*平面拟合：
	plane：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
template <typename T1, typename T2>
void PC_FitPlane(vector<T1>& pts,T2& plane, int k, NB_MODEL_FIT_METHOD method);

void PC_PlaneTest();