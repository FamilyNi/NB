#pragma once
#include "utils.h"

/*说明：
	球方程：(x - a)^2 + (y - b)^2 + (z - c)^2 = r^2
	拟合方程：x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
	pts：[in]空间中的点簇
	sphere：圆---[0]：a、[1]：b、[2]：c、[3]：r
*/

/*四点计算球：
	sphere：[out]
*/
template <typename T>
void PC_FourPtsComputeSphere(vector<T>& pts, cv::Vec4d& sphere);

/*最小二乘法拟合球：
	weights：[in]权重
	sphere：[out]
*/
template <typename T>
void PC_OLSFitSphere(vector<T>& pts, vector<double>& weights, cv::Vec4d& sphere);

/*huber计算权重：
	sphere：[in]
	weights：[out]权重
*/
template <typename T>
void PC_HuberSphereWeights(vector<T>& pts, cv::Vec4d& sphere, vector<double>& weights);

/*Turkey计算权重：
	sphere：[in]
	weights：[out]权重
*/
template <typename T>
void PC_TurkeySphereWeights(vector<T>& pts, cv::Vec4d& sphere, vector<double>& weights);

/*拟合球：
	sphere：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
template <typename T>
void PC_FitSphere(vector<T>& pts, cv::Vec4d& sphere, int k, NB_MODEL_FIT_METHOD method);


void PC_SphereTest();