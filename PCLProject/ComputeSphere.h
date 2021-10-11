#pragma once
#include "utils.h"

/*说明：
	球方程：(x - a)^2 + (y - b)^2 + (z - c)^2 = r^2
	拟合方程：x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
	pts：[in]空间中的点簇
	sphere：圆---[0]：a、[1]：b、[2]：c、[3]：r
*/

/*点到园的距离*/
template <typename T1, typename T2>
void PC_PtToShpereDist(T1& pt, T2& sphere, double& dist);

/*四点计算球：
	sphere：[out]
*/
template <typename T1, typename T2>
void PC_FourPtsComputeSphere(vector<T1>& pts, T2& sphere);

/*随机一致采样算法计算球：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
template <typename T1, typename T2>
void PC_RANSACComputeSphere(vector<T1>& pts, T2& sphere, vector<T1>& inlinerPts, double thres);

/*最小二乘法拟合球：
	weights：[in]权重
	sphere：[out]输出球
*/
template <typename T1, typename T2>
void PC_OLSFitSphere(vector<T1>& pts, vector<double>& weights, T2& sphere);

/*Huber计算权重：
	sphere：[in]
	weights：[out]权重
*/
template <typename T1, typename T2>
void PC_HuberSphereWeights(vector<T1>& pts, T2& sphere, vector<double>& weights);

/*Tukey计算权重：
	sphere：[in]
	weights：[out]权重
*/
template <typename T1, typename T2>
void PC_TukeySphereWeights(vector<T1>& pts, T2& sphere, vector<double>& weights);

/*拟合球：
	sphere：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
template <typename T1, typename T2>
void PC_FitSphere(vector<T1>& pts, T2& sphere, int k, NB_MODEL_FIT_METHOD method);

void PC_SphereTest();