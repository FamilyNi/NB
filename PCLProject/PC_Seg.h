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

/*DBSCAN�ָ
	indexs��[out]����ĵ�������
	radius��[in]����������뾶
	n��[in]�����ĸ���---�����жϸõ��Ƿ�Ϊ���ĵ�
	minGroup��[in]��С���ĸ���
	maxGroup��[in]������ĸ���
*/
void PC_DBSCANSeg(PC_XYZ::Ptr& srcPC, vector<vector<int>>& indexs, 
	double radius, int n, int minGroup, int maxGroup);

//Different Of Normal�ָ�
void DONSeg(PC_XYZ::Ptr &srcPC, float large_r, float small_r, float thresVal);

/*����ƽ��ָ
	plane��[in]�ο�ƽ��
	index��[out]����ĵ�������
	thresVal��[in]�㵽ƽ��ľ���
	orit��[in]����---0��ʾȡƽ���Ϸ��ĵ㡢1��ʾȡƽ���·��ĵ�
*/
void PC_SegBaseOnPlane(PC_XYZ::Ptr& srcPC, Plane3D& plane, vector<int>& index, double thresVal, int orit);

/*���Ʒָ���Գ���*/
void PC_SegTest();