#pragma once
#include "utils.h"
#include "OpenCV_Utils.h"

/*三点求圆：
	pt1、pt2、pt3：[in]平面不共线的三个点
	circle：[out]输出的圆---[0]：x圆心、[1]：y圆心、[2]：半径
*/
template <typename T>
void Img_ThreePointComputeCicle(T& pt1, T& pt2, T& pt3, cv::Vec3d& circle);

/*随机一致采样算法计算圆：
	pts：[in]平面上的点簇
	circle：[out]输出的圆---[0]：x圆心、[1]：y圆心、[2]：半径
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
template <typename T>
void Img_RANSACComputeCircle(vector<T>& pts, cv::Vec3d& circle, vector<T>& inlinerPts, double thres);

void CircleTest();