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

/*��ά������ˣ�===========================================================
	vec1��vec2��[in]��ʾ����1��2
	vec��[out]��˺�Ľ��
*/
template <typename T>
void PC_VecCross(T& vec1, T& vec2, T& vec, bool isNormal)
{
	vec.x = vec1.y * vec2.z - vec1.z * vec2.y;
	vec.y = vec1.z * vec2.x - vec1.x * vec2.z;
	vec.z = vec1.x * vec2.y - vec1.y * vec2.x;
	if (isNormal)
	{
		double norm_ = 1.0 / std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
		vec.x *= norm_; vec.y *= norm_; vec.z *= norm_;
	}
}
//=========================================================================

/*��������֮��ľ���--��ά*/
template <typename T>
void Img_ComputePPDist(T& pt1, T& pt2, double& dist);

/*�޸����˹��ʽ��=========================================================
	rotAxis��[in]��ת��
	rotAng��[in]������֮��ļн�
	rotMat��[out]��ת����
*/
template <typename T>
void RodriguesFormula(T& rotAxis, double rotAng, cv::Mat& rotMat)
{
	if (rotMat.size() != cv::Size(3, 3))
		rotMat = cv::Mat(cv::Size(3, 3), CV_64FC1, cv::Scalar(0.0));
	double cosVal = std::cos(rotAng);
	double conVal_ = 1 - cosVal;
	double sinVal = std::sin(rotAng);
	double* pRotMat = rotMat.ptr<double>();

	pRotMat[0] = cosVal + rotAxis.x * rotAxis.x * conVal_;
	pRotMat[1] = rotAxis.x * rotAxis.y * conVal_ - rotAxis.z * sinVal;
	pRotMat[2] = rotAxis.x * rotAxis.z * conVal_ + rotAxis.y * sinVal;

	pRotMat[3] = rotAxis.y * rotAxis.x * conVal_ + rotAxis.z * sinVal;
	pRotMat[4] = cosVal + rotAxis.y * rotAxis.y * conVal_;
	pRotMat[5] = rotAxis.y * rotAxis.z * conVal_ - rotAxis.x * sinVal;

	pRotMat[6] = rotAxis.z * rotAxis.x * conVal_ - rotAxis.y * sinVal;
	pRotMat[7] = rotAxis.z * rotAxis.y * conVal_ + rotAxis.x * sinVal;
	pRotMat[8] = cosVal + rotAxis.z * rotAxis.z * conVal_;
}
//=========================================================================
