#pragma once
#include "utils.h"

/*平面两线求点：
	line1、line2：[in]平面两直线---[0]：a、[1]：b、[2]：c
	pt：[out]输出的点
*/
template <typename T>
void Img_LineComputePt(cv::Vec3d& line1, cv::Vec3d& line2, T& pt);

/*三面共点：
	plane1、plane2、plane3：[in]空间中的三个平面---[0]：a、[1]：b、[2]：c、[3]：d
	pt：[out]输出的点
*/
template <typename T>
void PC_PlaneComputePt(cv::Vec4d& plane1, cv::Vec4d& plane2, cv::Vec4d& plane3, T& pt);

/*空间两线距离最近的点：
	line1、line2：[in]空间两直线---[0]：a、[1]：b、[2]：c----方向向量
					[3]：d、[4]：e、[5]：f---直线上的点
	pt1、pt2：[out]输出的点
	dist：[out]输出两直线的最短距离的平方
*/
template <typename T>
void PC_LineNearestPt(Vec6d& line1, Vec6d& line2, T& pt1, T& pt2, double& dist);
