#pragma once
#include "../BaseOprFile/utils.h"

/*说明：
	空间园表述：一般方程：x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
				法线方向：(x - a)*vx + (y - b)*vy + (z - c)*vz = 0
				vx、vy、vz为圆所在平面的法向量
	pts：[in]空间中的点簇
*/

/*随机一致采样算法计算圆：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
void PC_RANSACFitCircle(NB_Array3D pts, Circle3D& circle, vector<int>& inliners, double thres);

/*最小二乘法拟合空间空间圆：
	weights：[in]权重
	circle：[out]
*/
void PC_OLSFit3DCircle(NB_Array3D pts, vector<double>& weights, Circle3D& circle);

/*最小二乘法拟合圆：
	circle：[out]输出圆
	weights：[in]权重
*/
void PC_HuberCircleWeights(NB_Array3D pts, Circle3D& circle, vector<double>& weights);

/*Tukey计算权重：
	circle：[out]输出圆
	weights：[in]权重
*/
void PC_TukeyCircleWeights(NB_Array3D pts, Circle3D& circle, vector<double>& weights);

/*拟合圆：
	circle：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、tukey
*/
void PC_FitCircle(NB_Array3D pts, Circle3D& circle, int k, NB_MODEL_FIT_METHOD method);

//空间三维圆拟合测试
void PC_FitCircleTest();

