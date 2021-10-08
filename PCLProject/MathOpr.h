#pragma once
#include "utils.h"

//点到平面的距离
void PC_PtToPlaneDist(P_XYZ& pt, cv::Vec4d& plane, double& dist);

//向量归一化
template <typename T>
void PC_VecNormal(T& p);

//点到平面的投影点
void PC_PtProjPlanePt(P_XYZ& pt, cv::Vec4d& plane, P_XYZ& projPt);

//空间点到空间直线的距离
void PC_PtToLineDist(P_XYZ& pt, cv::Vec6d& line, double& dist);

//空间点到空间直线的投影
void PC_PtProjLinePt(P_XYZ& pt, cv::Vec6d& line, P_XYZ& projPt);

//十进制转二进制
void DecToBin(const int dec_num, vector<bool>& bin);

//二进制转十进制
void BinToDec(const vector<bool>& bin, int& dec_num);

//二进制转格雷码
void BinToGrayCode(const vector<bool>& bin, vector<bool>& grayCode);

//格雷码转二进制
void GrayCodeToBin(const vector<bool>& grayCode, vector<bool>& bin);

/*三维向量叉乘：===========================================================
	vec1、vec2：[in]表示向量1、2
	vec：[out]叉乘后的结果
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

/*计算两点之间的距离--二维*/
template <typename T>
void Img_ComputePPDist(T& pt1, T& pt2, double& dist);

/*罗格里格斯公式：=========================================================
	rotAxis：[in]旋转轴
	rotAng：[in]两向量之间的夹角
	rotMat：[out]旋转矩阵
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
