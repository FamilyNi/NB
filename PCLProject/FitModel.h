#pragma once
#include "utils.h"

/*Լ����
	ƽ�淽�̣�Ax + By + Cz - D = 0
	srcPC���������---����
	plane�����ƽ��----���
*/

/*RANSAC���ƽ��:
	inliers��ƽ���������
	thresValue�����ڵ���ж���ֵ
*/
void PC_RandomFitPlane(PC_XYZ::Ptr &srcPC, vector<int> &inliers, double thresValue = 0.01);

/*��С���˷����ƽ��*/
void PC_OLSFitPlane(PC_XYZ::Ptr &srcPC, Plane3D &plane);

//����Ȩ�ص�ƽ�����
void FitPlaneBaseOnWeight(PC_XYZ::Ptr &srcPC, P_N &normal, uint iter_k);

//�����
void PC_RandomFitSphere(PC_XYZ::Ptr &srcPC, double thresValue);

//�ĵ������
void ComputeSphere(vector<P_XYZ>& pts, double* pSphere);

//��С���˷������
void PC_OLSFitSphere(PC_XYZ::Ptr& srcPC, Sphere& sphere);
void PC_OLSFitSphere_(PC_XYZ::Ptr& srcPC, Sphere& sphere);

//ƽ����ϲ��Գ���
void PC_FitPlaneTest();
