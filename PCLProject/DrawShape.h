#pragma once
#include "utils.h"

/*绘制球:
	sphere：[out]输出的球
	center：[in]球心
	raduis：[in]半径
	step：[in]角度步长
*/
void DrawSphere(PC_XYZ::Ptr& sphere, P_XYZ& center, double raduis, double step);

/*绘制椭球面：
	ellipsoid：[out]输出的椭球面
	center：[in]椭球的中心位置
	a、b、c：[in]分别为x、y、z轴的轴长
	step：[in]角度步长
*/
void DrawEllipsoid(PC_XYZ::Ptr& ellipsoid, P_XYZ& center, double a, double b, double c, double step);