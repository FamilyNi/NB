#pragma once
#include "utils.h"
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/features/boundary.h>
#include <pcl/segmentation/extract_clusters.h>

/*�������һ���Եĵ��Ʒָ�:
	inliers���ָ�ĵ�������
	mode���ָ�ģ��
	thresVal��������ֵ
*/
//��������
void PC_RegionGrowing(PC_XYZ::Ptr &srcPC, std::vector<vector<uint>> &indexs, float radius);

//DBSCAN�ָ�
void DBSCANSeg(PC_XYZ::Ptr &srcPC, vector<vector<uint>> &indexs, float radius, uint p_number);

/*���Ʒָ���Գ���*/
void PC_SegTest();