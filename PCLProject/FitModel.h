#pragma once
#include "utils.h"

/*约定：
	平面方程：Ax + By + Cz - D = 0
	srcPC：输入点云---输入
	plane：拟合平面----输出
*/

/*RANSAC拟合平面:
	inliers：平面点云索引
	thresValue：局内点的判定阈值
*/
void PC_RandomFitPlane(PC_XYZ::Ptr &srcPC, vector<int> &inliers, double thresValue = 0.01);

/*最小二乘法拟合平面*/
void PC_OLSFitPlane(PC_XYZ::Ptr &srcPC, Plane3D &plane);

/*两步拟合平面：
	第一步：采用RANSAC进行粗拟合
	第二步：采用最小二乘法进行精确拟合
*/
void PC_FitPlane(PC_XYZ::Ptr &srcPC, Plane3D &plane, float thresVal);

//基于权重的平面拟合
void FitPlaneBaseOnWeight(PC_XYZ::Ptr &srcPC, P_N &normal, uint iter_k);

//平面拟合测试程序
void PC_FitPlaneTest();
