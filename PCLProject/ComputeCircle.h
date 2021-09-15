#pragma once
#include "utils.h"
#include "OpenCV_Utils.h"

/*说明：
	园方程：(x - a)^2 + (y - b)^2 = r^2
	拟合方程：x^2 + y^2 + A * x + B * y + C = 0
	pts：[in]平面上的点簇
	circle：圆---[0]：a、[1]：b、[2]：r
*/

/*三点求圆：
	pt1、pt2、pt3：[in]平面不共线的三个点
	circle：[out]输出的圆---[0]：x圆心、[1]：y圆心、[2]：半径
*/
template <typename T>
void Img_ThreePtsComputeCicle(T& pt1, T& pt2, T& pt3, cv::Vec3d& circle);

/*随机一致采样算法计算圆：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
template <typename T>
void Img_RANSACComputeCircle(vector<T>& pts, cv::Vec3d& circle, vector<T>& inlinerPts, double thres);

/*最小二乘法拟合园：
	weights：[in]权重
	circle：[out]
*/
template <typename T>
void Img_OLSFitCircle(vector<T>& pts, vector<double>& weights, cv::Vec3d& circle);

/*huber计算权重：
	circle：[in]
	weights：[out]权重
*/
template <typename T>
void Img_HuberCircleWeights(vector<T>& pts, cv::Vec3d& circle, vector<double>& weights);

/*Turkey计算权重：
	circle：[in]
	weights：[out]权重
*/
template <typename T>
void Img_TurkeyCircleWeights(vector<T>& pts, cv::Vec3d& circle, vector<double>& weights);

/*拟合园
	circle：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
template <typename T>
void Img_FitCircle(vector<T>& pts, cv::Vec3d& circle, int k, NB_MODEL_FIT_METHOD method);

void CircleTest();