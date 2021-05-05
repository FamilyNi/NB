#pragma once
#include "utils.h"

//计算点云的重心
void ComputePCGravity(PC_XYZ::Ptr &srcPC, P_XYZ &gravity);

//将点云投影到XY平面
void PC_ProjectToXY(PC_XYZ::Ptr &srcPC, cv::Mat &xyPlane);

//计算点云的法向量
void ComputePCNormal(PC_XYZ::Ptr &srcPC, PC_N::Ptr &normals, float radius);

//计算协方差矩阵
void ComputeCovMat(PC_XYZ::Ptr &pc, cv::Mat &covMat, P_XYZ &gravity);

//计算点云的深度图
void ComputePCRangeImg();