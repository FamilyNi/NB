#pragma once
#include "utils.h"

/*说明：
	拟合方程：a * x^2 + b * x * y + c * y * y + d * x + e * y + f = 0
	pts：[in]平面上的点簇
	ellipse：圆---[0]：a、[1]：b、[2]：c、[3]：d、[4]：e、[5]：f
*/

/*椭圆方程标准化*/
template <typename T>
void Img_EllipseNormalization(T& ellipse_, T& normEllipse);

/*点到椭圆的距离--超简单版，不建议采用*/
template <typename T1, typename T2>
void Img_PtsToEllipseDist(T1& pt, T2& ellipse, double& dist);

/*六点求椭圆*/
template <typename T1, typename T2>
void Img_SixPtsComputeEllipse(vector<T1>& pts, T2& ellipse);

/*随机一致采样算法计算椭圆圆：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
template <typename T1, typename T2>
void Img_RANSACComputeCircle(vector<T1>& pts, T2& ellipse, vector<T1>& inlinerPts, double thres);

/*最小二乘法拟合椭圆：
	weights：[in]权重
	ellipse：[out]
*/
template <typename T1, typename T2>
void Img_OLSFitEllipse(vector<T1>& pts, vector<double>& weights, T2& ellipse);

/*huber计算权重：
	ellipse：[in]
	weights：[out]权重
*/
template <typename T1, typename T2>
void Img_HuberEllipseWeights(vector<T1>& pts, T2& ellipse, vector<double>& weights);

/*Turkey计算权重：
	ellipse：[in]
	weights：[out]权重
*/
template <typename T1, typename T2>
void Img_TukeyEllipseWeights(vector<T1>& pts, T2& ellipse, vector<double>& weights);

/*拟合椭圆：
	ellipse：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
template <typename T1, typename T2>
void Img_FitEllipse(vector<T1>& pts, T2& ellipse, int k, NB_MODEL_FIT_METHOD method);


void Img_EllipseTest();
