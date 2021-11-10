#pragma once
#include "../BaseOprFile/utils.h"
#include "../BaseOprFile/OpenCV_Utils.h"

/*说明：
	园方程：(x - a)^2 + (y - b)^2 = r^2
	拟合方程：x^2 + y^2 + A * x + B * y + C = 0
	pts：[in]平面上的点簇
*/

/*随机一致采样算法计算圆：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
void Img_RANSACFitCircle(NB_Array2D pts, Circle2D& circle, vector<int>& inliners, double thres);

/*最小二乘法拟合园：
	weights：[in]权重
	circle：[out]
*/
void Img_OLSFitCircle(NB_Array2D pts, vector<double>& weights, Circle2D& circle);

/*huber计算权重：
	circle：[in]
	weights：[out]权重
*/
void Img_HuberCircleWeights(NB_Array2D pts, Circle2D& circle, vector<double>& weights);

/*Tukey计算权重：
	circle：[in]
	weights：[out]权重
*/
void Img_TukeyCircleWeights(NB_Array2D pts, Circle2D& circle, vector<double>& weights);

/*拟合园
	circle：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、Tukey
*/
void Img_FitCircle(NB_Array2D pts, Circle2D& circle, int k, NB_MODEL_FIT_METHOD method);

//二维圆拟合测试
void Img_FitCircleTest();
