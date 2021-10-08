#include "MathOpr.h"

//点到平面的距离==============================================================
void PC_PtToPlaneDist(P_XYZ& pt, cv::Vec4d& plane, double& dist)
{
	dist = abs(pt.x * plane[0] + pt.y * plane[1] + pt.z * plane[2] + plane[3]);
}
//============================================================================

//向量归一化==================================================================
template <typename T>
void PC_VecNormal(T& p)
{
	float norm_ = 1 / std::max(std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z), 1e-8f);
	p.x *= norm_; p.y *= norm_; p.z *= norm_;
}
//============================================================================

//点到平面的投影点============================================================
void PC_PtProjPlanePt(P_XYZ& pt, cv::Vec4d& plane, P_XYZ& projPt)
{
	float dist = pt.x * plane[0] + pt.y * plane[1] + pt.z * plane[2] + plane[3];
	projPt = { pt.x - float(dist * plane[0]), float(pt.x - dist * plane[1]),float(pt.x - dist * plane[2]) };
}
//===========================================================================

//空间点到空间直线的距离=====================================================
void PC_PtToLineDist(P_XYZ& pt, cv::Vec6d& line, double& dist)
{
	double scale = pt.x * line[0] + pt.y * line[1] + pt.z * line[2] -
		(line[3] * line[0] + line[4] * line[1] + line[5] * line[2]);
	double diff_x = line[3] + scale * line[0] - pt.x;
	double diff_y = line[4] + scale * line[1] - pt.y;
	double diff_z = line[5] + scale * line[2] - pt.z;
	dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
}
//===========================================================================

//空间点到空间直线的投影=====================================================
void PC_PtProjLinePt(P_XYZ& pt, cv::Vec6d& line, P_XYZ& projPt)
{
	float scale = pt.x * line[0] + pt.y * line[1] + pt.z * line[2] -
		(line[3] * line[0] + line[4] * line[1] + line[5] * line[2]);
	projPt.x = line[3] + scale * line[0];
	projPt.y = line[4] + scale * line[1]; 
	projPt.z = line[5] + scale * line[2];
}
//==========================================================================

//十进制转二进制============================================================
void DecToBin(const int dec_num, vector<bool>& bin)
{
	int a = dec_num;
	int index = 0;
	int length = bin.size() - 1;
	while (a != 0)
	{
		if (index > length)
			break;
		bin[index] = a % 2;
		a /= 2;
		index++;
	}
}
//==========================================================================

//二进制转十进制============================================================
void BinToDec(const vector<bool>& bin, int& dec_num)
{
	dec_num = 0;
	for (size_t i = 0; i < bin.size(); ++i)
	{
		dec_num += bin[i] * std::pow(2, (int)i);
	}
}
//==========================================================================

//二进制转格雷码============================================================
void BinToGrayCode(const vector<bool>& bin, vector<bool>& grayCode)
{
	int len = bin.size();
	grayCode.resize(len);
	for (int i = 0; i < len - 1; ++i)
	{
		grayCode[i] = bin[i] ^ bin[i + 1];
	}
	grayCode[len - 1] = bin[len - 1];
}
//==========================================================================

//格雷码转二进制============================================================
void GrayCodeToBin(const vector<bool>& grayCode, vector<bool>& bin)
{
	int len = grayCode.size();
	bin.resize(len);
	bin[len - 1] = grayCode[len - 1];
	for (int i = len - 2; i > -1; --i)
	{
		bin[i] = grayCode[i] ^ bin[i + 1];
	}
}
//==========================================================================

//计算点间距===============================================================
template <typename T>
void Img_ComputePPDist(T& pt1, T& pt2, double& dist)
{
	double diff_x = pt1.x - pt2.x;
	double diff_y = pt1.y - pt2.y;
	return std::sqrt(max(diff_x * diff_x + diff_y * diff_y, EPS));
}
//=========================================================================

