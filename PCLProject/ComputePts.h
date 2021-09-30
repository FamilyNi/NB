#pragma once
#include "utils.h"

/*ƽ��������㣺
	line1��line2��[in]ƽ����ֱ��---[0]��a��[1]��b��[2]��c
	pt��[out]����ĵ�
*/
template <typename T>
void Img_LineComputePt(cv::Vec3d& line1, cv::Vec3d& line2, T& pt);

/*���湲�㣺
	plane1��plane2��plane3��[in]�ռ��е�����ƽ��---[0]��a��[1]��b��[2]��c��[3]��d
	pt��[out]����ĵ�
*/
template <typename T>
void PC_PlaneComputePt(cv::Vec4d& plane1, cv::Vec4d& plane2, cv::Vec4d& plane3, T& pt);

/*�ռ����߾�������ĵ㣺
	line1��line2��[in]�ռ���ֱ��---[0]��a��[1]��b��[2]��c----��������
					[3]��d��[4]��e��[5]��f---ֱ���ϵĵ�
	pt1��pt2��[out]����ĵ�
	dist��[out]�����ֱ�ߵ���̾����ƽ��
*/
template <typename T>
void PC_LineNearestPt(Vec6d& line1, Vec6d& line2, T& pt1, T& pt2, double& dist);
