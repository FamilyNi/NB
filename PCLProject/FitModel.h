#pragma once
#include "utils.h"

/*RANSAC拟合平面:
	inliers：平面点云索引
	thresValue：局内点的判定阈值
*/
void PC_RandomFitPlane(PC_XYZ::Ptr &srcPC, vector<int> &inliers, double thresValue = 0.01);

/*最小二乘法拟合平面:Ax + By + Cz + D = 0
	srcPC：[in]输入点云
	sphere：[out]拟合的平面
*/
void PC_OLSFitPlane(PC_XYZ::Ptr &srcPC, Plane3D &plane);

//基于权重的平面拟合
void FitPlaneBaseOnWeight(PC_XYZ::Ptr &srcPC, P_N &normal, uint iter_k);

/*随机一致采样拟合球:
	srcPC：[in]输入点云
	sphere：[out]拟合的球
*/
void PC_RandomFitSphere(PC_XYZ::Ptr &srcPC, double thresValue);

/*最小二乘法拟合球:x^2 + y^2 + z^2 + Ax + By + Cz + D = 0
	srcPC：[in]输入点云
	sphere：[out]拟合的球
*/
void PC_OLSFitSphere(PC_XYZ::Ptr& srcPC, Sphere& sphere);

//平面拟合测试程序
void PC_FitPlaneTest();
