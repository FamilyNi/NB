#pragma once
#include "utils.h"
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/features/boundary.h>
#include <pcl/segmentation/extract_clusters.h>

/*随机采样一致性的点云分割:
	inliers：分割的点云索引
	mode：分割模型
	thresVal：距离阈值
*/
//区域生长
void PC_RegionGrowing(PC_XYZ::Ptr &srcPC, std::vector<vector<uint>> &indexs, float radius);

//DBSCAN分割
void DBSCANSeg(PC_XYZ::Ptr &srcPC, vector<vector<uint>> &indexs, float radius, uint p_number);

/*点云分割测试程序*/
void PC_SegTest();