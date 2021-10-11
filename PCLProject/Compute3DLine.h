#pragma once
#include "utils.h"

/*说明：
	直线方程： x = a * t + x0; y = b * t + y0; z = c * t + z0
	pts：[in]空间中的点簇
	line：---[0]：a、[1]：b、[2]：c、[3]：x0、[4]：y0、[5]：z0
*/

/*两点求线*/
template <typename T1, typename T2>
void PC_TwoPtsComputeLine(T1& pt1, T1& pt2, T2& line);

/*随机一致采样算法计算空间直线：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
template <typename T1, typename T2>
void PC_RANSACComputeLine(vector<T1>& pts, T2& line, vector<T1>& inlinerPts, double thres);

/*最小二乘法拟合空间直线：
	weights：[in]权重
	line：[out]
*/
template <typename T1, typename T2>
void PC_OLSFit3DLine(vector<T1>& pts, vector<double>& weights, T2& line);

/*Huber计算权重：
	line：[in]
	weights：[out]权重
*/
template <typename T1, typename T2>
void PC_Huber3DLineWeights(vector<T1>& pts, T2& line, vector<double>& weights);

/*Tukey计算权重：
	line：[in]
	weights：[out]权重
*/
template <typename T1, typename T2>
void PC_Tukey3DLineWeights(vector<T1>& pts, T2& line, vector<double>& weights);

/*空间直线拟合：
	line：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
template <typename T1, typename T2>
void PC_Fit3DLine(vector<T1>& pts, T2& line, int k, NB_MODEL_FIT_METHOD method);


void PC_3DLineTest();