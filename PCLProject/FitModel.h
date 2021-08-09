#pragma once
#include "utils.h"

/*RANSAC���ƽ��:
	inliers��ƽ���������
	thresValue�����ڵ���ж���ֵ
*/
void PC_RandomFitPlane(PC_XYZ::Ptr &srcPC, vector<int> &inliers, double thresValue = 0.01);

/*��С���˷����ƽ��:Ax + By + Cz + D = 0
	srcPC��[in]�������
	sphere��[out]��ϵ�ƽ��
*/
void PC_OLSFitPlane(PC_XYZ::Ptr &srcPC, Plane3D &plane);

//����Ȩ�ص�ƽ�����
void FitPlaneBaseOnWeight(PC_XYZ::Ptr &srcPC, P_N &normal, uint iter_k);

/*���һ�²��������:
	srcPC��[in]�������
	sphere��[out]��ϵ���
*/
void PC_RandomFitSphere(PC_XYZ::Ptr &srcPC, double thresValue);

/*��С���˷������:x^2 + y^2 + z^2 + Ax + By + Cz + D = 0
	srcPC��[in]�������
	sphere��[out]��ϵ���
*/
void PC_OLSFitSphere(PC_XYZ::Ptr& srcPC, Sphere& sphere);

//ƽ����ϲ��Գ���
void PC_FitPlaneTest();
