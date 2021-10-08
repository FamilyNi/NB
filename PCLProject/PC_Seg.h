#pragma once
#include "utils.h"
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/features/boundary.h>
#include <pcl/segmentation/extract_clusters.h>

//随机一致采样模型分割
void PC_RANSACSeg(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, int mode, float thresVal);

/*基于欧式距离分割方法*/
void PC_EuclideanSeg(PC_XYZ::Ptr &srcPC, std::vector<P_IDX> clusters, float distThresVal);

//区域生长
void PC_RegionGrowing(PC_XYZ::Ptr &srcPC, std::vector<vector<uint>> &indexs, float radius);

/*DBSCAN分割：
	indexs：[out]输出的点云索引
	radius：[in]搜索的邻域半径
	n：[in]邻域点的个数---用来判断该点是否为核心点
	minGroup：[in]最小族点的个数
	maxGroup：[in]最大族点的个数
*/
void PC_DBSCANSeg(PC_XYZ::Ptr& srcPC, vector<vector<int>>& indexs, 
	double radius, int n, int minGroup, int maxGroup);

//Different Of Normal分割
void DONSeg(PC_XYZ::Ptr &srcPC, float large_r, float small_r, float thresVal);

/*根据平面分割：
	plane：[in]参考平面
	index：[out]输出的点云索引
	thresVal：[in]点到平面的距离
	orit：[in]方向---0表示取平面上方的点、1表示取平面下方的点
*/
void PC_SegBaseOnPlane(PC_XYZ::Ptr& srcPC, Plane3D& plane, vector<int>& index, double thresVal, int orit);

/*点云分割测试程序*/
void PC_SegTest();