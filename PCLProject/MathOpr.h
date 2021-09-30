#pragma once
#include "utils.h"

//�㵽ƽ��ľ���
void PC_PtToPlaneDist(P_XYZ& pt, cv::Vec4d& plane, double& dist);

//������һ��
template <typename T>
void PC_VecNormal(T& p);

//�㵽ƽ���ͶӰ��
void PC_PtProjPlanePt(P_XYZ& pt, cv::Vec4d& plane, P_XYZ& projPt);

//�ռ�㵽�ռ�ֱ�ߵľ���
void PC_PtToLineDist(P_XYZ& pt, cv::Vec6d& line, double& dist);

//�ռ�㵽�ռ�ֱ�ߵ�ͶӰ
void PC_PtProjLinePt(P_XYZ& pt, cv::Vec6d& line, P_XYZ& projPt);

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
template <typename T>
void VecCross_PC(T& vec1, T& vec2, T& vec);

/*��������֮��ľ���--��ά*/
template <typename T>
void Img_ComputePPDist(T& pt1, T& pt2, double& dist);