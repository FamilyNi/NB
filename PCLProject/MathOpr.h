#pragma once
#include "utils.h"

//ʮ����ת������
void DecToBin(const int dec_num, vector<bool>& bin);

//������תʮ����
void BinToDec(const vector<bool>& bin, int& dec_num);

//������ת������
void BinToGrayCode(const vector<bool>& bin, vector<bool>& grayCode);

//������ת������
void GrayCodeToBin(const vector<bool>& grayCode, vector<bool>& bin);

/*��ά�������:
	vec1��vec2��[in]��ʾ����1��2
	vec��[out]��˺�Ľ��
*/
void VecCross_PC(P_XYZ& vec1, P_XYZ& vec2, P_XYZ& vec);

/*������һ����
	p��[in--out]����һ������
*/
void Normal_PC(P_XYZ& p);

/*�ռ����߾�������ĵ�:
	line1��[in]ֱ��1---ǰ��������Ϊ����������������Ϊ�ռ��
	line2��[in]ֱ��2---ǰ��������Ϊ����������������Ϊ�ռ��
	pt1��[out]ֱ��1�ϵĵ�
	pt2��[out]ֱ��2�ϵĵ�
	����Ϊpt1��pt2�����ƽ��
*/
float SpaceLineNearestPt(Vec6f& line1, Vec6f& line2, P_XYZ& pt1, P_XYZ& pt2);

//�ĵ������
void ComputeSphere(vector<P_XYZ>& pts, Sphere& pSphere);

/*������ռ��:
	plane1��plane2��plane3��[in]��ʾ����ƽ��
	point��[out]�����ཻ�ĵ�
*/
void ComputePtBasePlanes(Plane3D plane1, Plane3D plane2, Plane3D plane3, P_XYZ& point);

/*��������:
	pts1��pts2��pts3��[in]����������
	plane��[out]����õ���ƽ��
	ע������ƽ��ķ�����a��b��c�����ǹ�һ����
*/
void ComputePlane(P_XYZ& pt1, P_XYZ& pt2, P_XYZ& pt3, Plane3D& plane);