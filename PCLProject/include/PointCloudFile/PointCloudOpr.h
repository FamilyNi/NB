#pragma once
#include "../BaseOprFile/utils.h"

//计算点云的最小包围盒
void PC_ComputeOBB(const PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& obb);

/*提取点云：
	indexes：[in]点云索引
	dstPC：[out]输出的点云
*/
void PC_ExtractPC(const PC_XYZ::Ptr& srcPC, vector<int>& indexes, PC_XYZ::Ptr& dstPC);

/*点云直线投影平滑：
	dstPC：[out]输出的点云
	size：[in]点云最近邻点的个数
	thresVal：[in]点云最近邻点的个数
*/
void PC_LineProjSmooth(const PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, int size, double thresVal);

//计算点云的重心
void PC_GetPCGravity(PC_XYZ::Ptr &srcPC, P_XYZ &gravity);

//将点云投影到XY二维平面
void PC_ProjectToXY(PC_XYZ::Ptr &srcPC, cv::Mat &xyPlane);

//计算点云的法向量
void PC_ComputePCNormal(PC_XYZ::Ptr &srcPC, PC_N::Ptr &normals, float radius);


