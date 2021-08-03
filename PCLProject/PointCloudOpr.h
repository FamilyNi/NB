#pragma once
#include "utils.h"

//计算点云的最小包围盒
void PC_ComputeOBB(const PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& obb);

//提取点云
void PC_ExtractPC(const PC_XYZ::Ptr& srcPC, vector<int>& indexes, PC_XYZ::Ptr& dstPC);

//点云直线投影平滑
void PC_LineProjSmooth(const PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, int size, float thresVal);

//计算点云的重心
void PC_GetPCGravity(PC_XYZ::Ptr &srcPC, P_XYZ &gravity);

//沿法线方向缩放点云
void PC_ScalePCBaseNormal(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &normal_p, PC_XYZ::Ptr &dstPC, float scale);

//将点云投影到XY二维平面
void PC_ProjectToXY(PC_XYZ::Ptr &srcPC, cv::Mat &xyPlane);

//计算点云的法向量
void PC_ComputePCNormal(PC_XYZ::Ptr &srcPC, PC_N::Ptr &normals, float radius);

//向量叉乘
void PC_MulCrossCalNormal(P_XYZ &p_1, P_XYZ &p_2, P_XYZ &normal_p);

//计算点云的协方差矩阵
void PC_ComputeCovMat(PC_XYZ::Ptr &pc, Mat& covMat, P_XYZ& gravity);
