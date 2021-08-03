#pragma once
#include "utils.h"
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/features/boundary.h>
#include <pcl/segmentation/extract_clusters.h>

//���һ�²���ģ�ͷָ�
void PC_RANSACSeg(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, int mode, float thresVal);

/*����ŷʽ����ָ��*/
void PC_EuclideanSeg(PC_XYZ::Ptr &srcPC, std::vector<P_IDX> clusters, float distThresVal);

//��������
void PC_RegionGrowing(PC_XYZ::Ptr &srcPC, std::vector<vector<uint>> &indexs, float radius);

//DBSCAN�ָ�
void DBSCANSeg(PC_XYZ::Ptr &srcPC, vector<vector<uint>> &indexs, float radius, int p_number);

//Different Of Normal�ָ�
void DONSeg(PC_XYZ::Ptr &srcPC, float large_r, float small_r, float thresVal);

/*���Ʒָ���Գ���*/
void PC_SegTest();