#pragma once
#include "../BaseOprFile/utils.h"

/*说明：
	平面方程：a * x + b * y + c * z + d = 0
	pts：[in]空间中的点簇
*/

/*随机一致采样算法计算平面：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
void PC_RANSACFitPlane(NB_Array3D pts, Plane3D& plane, vector<int>& inliners, double thres);

/*最小二乘法拟合平面：
	weights：[in]权重
	plane：[out]
*/
void PC_OLSFitPlane(NB_Array3D pts, vector<double>& weights, Plane3D& plane);

/*huber计算权重：
	plane：[in]
	weights：[out]权重
*/
void PC_HuberPlaneWeights(NB_Array3D pts, Plane3D& plane, vector<double>& weights);

/*Tukey计算权重：
	plane：[in]
	weights：[out]权重
*/
void PC_TukeyPlaneWeights(NB_Array3D pts, Plane3D& plane, vector<double>& weights);

/*平面拟合：
	plane：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
void PC_FitPlane(NB_Array3D pts, Plane3D& plane, int k, NB_MODEL_FIT_METHOD method);

//空间平面拟合测试
void PC_FitPlaneTest();
