#pragma once
#include "utils.h"

/*说明：
	直线方程： x = a * t + x0; y = b * t + y0; z = c * t + z0
	pts：[in]空间中的点簇
*/

/*随机一致采样算法计算空间直线：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
void PC_RANSACFitLine(NB_Array3D pts, Line3D& line, vector<int>& inlinerPts, double thres);

/*最小二乘法拟合空间直线：
	weights：[in]权重
	line：[out]
*/
//template <typename T1, typename T2>
void PC_OLSFit3DLine(NB_Array3D pts, vector<double>& weights, Line3D& line);

/*Huber计算权重：
	line：[in]
	weights：[out]权重
*/
//template <typename T1, typename T2>
void PC_Huber3DLineWeights(NB_Array3D pts, Line3D& line, vector<double>& weights);

/*Tukey计算权重：
	line：[in]
	weights：[out]权重
*/
//template <typename T1, typename T2>
void PC_Tukey3DLineWeights(NB_Array3D pts, Line3D& line, vector<double>& weights);

/*空间直线拟合：
	line：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
//template <typename T1, typename T2>
void PC_Fit3DLine(NB_Array3D pts, Line3D& line, int k, NB_MODEL_FIT_METHOD method);

//空间三位直线拟合测试
void PC_FitLineTest();
