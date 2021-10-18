#pragma once
#include "utils.h"

/*说明：
	空间园表述：一般方程：x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
				法线方向：(x - a)*vx + (y - b)*vy + (z - c)*vz = 0
	pts：[in]空间中的点簇
	circle：---[0]：a、[1]：b、[2]：c、[3]：x0、[4]：y0、[5]：z0
*/

/*点到圆的距离*/
template <typename T1, typename T2>
void PC_PtToCircleDist(T1& pt, T2& circle, double& dist);

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
