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

//DBSCAN分割
void DBSCANSeg(PC_XYZ::Ptr &srcPC, vector<vector<uint>> &indexs, float radius, int p_number);

//Different Of Normal分割
void DONSeg(PC_XYZ::Ptr &srcPC, float large_r, float small_r, float thresVal);

/*点云分割测试程序*/
void PC_SegTest();