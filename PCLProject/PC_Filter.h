#pragma once
#include "utils.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/bilateral.h> 
#include <pcl/filters/fast_bilateral.h>  
#include <pcl/filters/median_filter.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/filters/morphological_filter.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/surface/mls.h>

/*Լ����
	srcPC��ԭ����
    dstPC���������������
*/

//���Ƶ������˲�-----�²���
int PC_VoxelGrid(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, float leafSize);

//ֱͨ�˲�----����ָ��������ָ������ĵ���
void PC_PassFilter(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, const string mode, double minVal, double maxVal);

//�����ܶ�Լ���Ķ�̬��׼����ֵ��Ⱥ����ģ��
void NeighbourMove(PC_XYZ::Ptr &srcPC, vector<uint> &index_p, uint k, float outcoef, float incoef);

//���Գ���
void PC_FitlerTest();