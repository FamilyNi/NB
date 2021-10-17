#pragma once
#include "utils.h"

/*形状变换：
	pc：[in--out]输入输出形状
	shape：[in]形状参数
	mode：[in] 0--表示直线、1--表示平面
*/
void PC_ShapeTrans(PC_XYZ::Ptr& pc, cv::Vec6d& shape, cv::Point3d& vec);

/*绘制直线：
	linePC：[out]输出直线
	length：[in]直线长度
	line：[in]直线参数
	step：[in]步长
*/
void PC_DrawLine(PC_XYZ::Ptr& linePC, cv::Vec6d& line, double length, double step);

/*绘制平面：
	planePC：[out]输出平面
	length：[in]平面长
	width：[in]平面宽
	plane：[in]平面参数
	step：[in]步长
*/
void PC_DrawPlane(PC_XYZ::Ptr& planePC, cv::Vec6d& plane, double length, double width, double step);

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

/*绘制椭圆：
	ellipseImg：[out]输出的椭圆
	center：[in]椭圆的中心位置
	a、b：[in]分别为x、y方向上轴长
	rotAng：[in]旋转角度
	step：[in]角度步长
*/
void Img_DrawEllipse(Mat& ellipseImg, cv::Point2d& center, double rotAng, double a, double b, double step);

/*绘制立方体（空心）：
	rectPC：[out]输出平面
	rect：[in]平面参数
	step：[in]步长
*/
void PC_DrawCube(PC_XYZ::Ptr& rectPC, cv::Vec6d& cube, double step);

/*添加噪声：
	srcPC：[in]原始点云
	noisePC：[out]噪声点云
	range：[in]噪声大小
	step：[in]点云步长
*/
void PC_AddNoise(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& noisePC, int range, int step);

//测试程序
void DrawShapeTest();