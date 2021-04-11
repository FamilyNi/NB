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

/*�������ƽ�棺
	��һ��������RANSAC���д����
	�ڶ�����������С���˷����о�ȷ���
*/
void PC_FitPlane(PC_XYZ::Ptr &srcPC, Plane3D &plane, float thresVal);

//����Ȩ�ص�ƽ�����
void FitPlaneBaseOnWeight(PC_XYZ::Ptr &srcPC, P_N &normal, uint iter_k);

//ƽ����ϲ��Գ���
void PC_FitPlaneTest();
