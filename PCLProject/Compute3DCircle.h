#pragma once
#include "utils.h"

/*说明：
	空间园表述：一般方程：x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
				法线方向：(x - a)*vx + (y - b)*vy + (z - c)*vz = 0
				vx、vy、vz为圆所在平面的法向量
	pts：[in]空间中的点簇
	circle：---[0]：圆心x、[1]：圆心y、[2]：圆心z、[3]：半径、[4]：vx、[5]：vy、[6]：vz
*/

/*点到圆的距离*/
template <typename T1, typename T2>
void PC_PtToCircleDist(T1& pt, T2& circle, double& dist);

/*三点计算园：
	pt1、pt2、pt3：[in]空间中的三个点
	circle：[out]
*/
template <typename T1, typename T2>
void PC_ThreePtsComputeCircle(T1& pt1, T1& pt2, T1& pt3, T2& circle);

/*随机一致采样算法计算圆：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
template <typename T1, typename T2>
void PC_RANSACComputeCircle(vector<T1>& pts, T2& circle, vector<T1>& inlinerPts, double thres);

/*最小二乘法拟合空间空间园：
	weights：[in]权重
	circle：[out]
*/
template <typename T1, typename T2>
void PC_OLSFit3DCircle(vector<T1>& pts, vector<double>& weights, T2& circle);

/*最小二乘法拟合圆：
	circle：[out]输出圆
	weights：[in]权重
*/
template <typename T1, typename T2>
void PC_HuberCircleWeights(vector<T1>& pts, T2& circle, vector<double>& weights);

/*Tukey计算权重：
	circle：[out]输出圆
	weights：[in]权重
*/
template <typename T1, typename T2>
void PC_TukeyCircleWeights(vector<T1>& pts, T2& circle, vector<double>& weights);

void PC_CircleTest();
