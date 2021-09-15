#pragma once
#include "utils.h"
#include "MathOpr.h"

/*绘制平面：
	linePC：[in]输出直线
	min_x、max_x：[in] x方向的最大值与最小值
	min_y、max_y：[in] y方向的最大值与最小值
	min_z、max_z：[in] z方向的最大值与最小值
	line：[in]直线参数
	step：[in]步长
*/
void PC_DrawLine(PC_XYZ::Ptr& linePC, double min_x, double max_x, double min_y,
	double max_y, double min_z, double max_z, cv::Vec6d& line, double step);

/*绘制平面：
	planePC：[in]输出平面
	min_x、max_x：[in] x方向的最大值与最小值
	min_y、max_y：[in] y方向的最大值与最小值
	min_z、max_z：[in] z方向的最大值与最小值
	plane：[in]平面参数
	step：[in]步长
*/
void PC_DrawPlane(PC_XYZ::Ptr& planePC, double min_x, double max_x, double min_y, 
	double max_y, double min_z, double max_z, cv::Vec4d& plane, double step);


/*绘制球:
	spherePC：[out]输出的球
	center：[in]球心
	raduis：[in]半径
	step：[in]角度步长
*/
void PC_DrawSphere(PC_XYZ::Ptr& spherePC, P_XYZ& center, double raduis, double step);

/*绘制椭球面：
	ellipsoidPC：[out]输出的椭球面
	center：[in]椭球的中心位置
	a、b、c：[in]分别为x、y、z轴的轴长
	step：[in]角度步长
*/
void PC_DrawEllipsoid(PC_XYZ::Ptr& ellipsoidPC, P_XYZ& center, double a, double b, double c, double step);

/*添加噪声：
	srcPC：[in]原始点云
	noisePC：[out]噪声点云
	range：[in]噪声大小
	step：[in]点云步长
*/
void PC_AddNoise(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& noisePC, int range, int step);