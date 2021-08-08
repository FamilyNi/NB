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

/*�ռ����߾�������ĵ�:
	line1��[in]ֱ��1---ǰ��������Ϊ����������������Ϊ�ռ��
	line2��[in]ֱ��2---ǰ��������Ϊ����������������Ϊ�ռ��
	pt1��[out]ֱ��1�ϵĵ�
	pt2��[out]ֱ��2�ϵĵ�
	����Ϊpt1��pt2�����ƽ��
*/
float SpaceLineNearestPt(Vec6f& line1, Vec6f& line2, P_XYZ& pt1, P_XYZ& pt2);

//�ĵ������
void ComputeSphere(vector<P_XYZ>& pts, double* pSphere);;