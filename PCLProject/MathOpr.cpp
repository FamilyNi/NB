#include "MathOpr.h"

//点到平面的距离==============================================================
template <typename T1, typename T2>
void PC_PtToPlaneDist(T1& pt, T2& plane, double& dist)
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
template <typename T1, typename T2>
void PC_PtProjPlanePt(T1& pt, T2& plane, T1& projPt)
{
	float dist = pt.x * plane[0] + pt.y * plane[1] + pt.z * plane[2] + plane[3];
	projPt = { pt.x - float(dist * plane[0]), float(pt.x - dist * plane[1]),float(pt.x - dist * plane[2]) };
}
//===========================================================================

//空间点到空间直线的距离=====================================================
template <typename T1, typename T2>
void PC_PtToLineDist(T1& pt, T2& line, double& dist)
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
template <typename T1, typename T2>
void PC_PtProjLinePt(T1& pt, T2& line, T1& projPt)
{
	float scale = pt.x * line[0] + pt.y * line[1] + pt.z * line[2] -
		(line[3] * line[0] + line[4] * line[1] + line[5] * line[2]);
	projPt.x = line[3] + scale * line[0];
	projPt.y = line[4] + scale * line[1]; 
	projPt.z = line[5] + scale * line[2];
}
//===========================================================================

//三维向量叉乘===============================================================
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

//计算点间距===============================================================
template <typename T>
void Img_ComputePPDist(T& pt1, T& pt2, double& dist)
{
	double diff_x = pt1.x - pt2.x;
	double diff_y = pt1.y - pt2.y;
	return std::sqrt(max(diff_x * diff_x + diff_y * diff_y, EPS));
}
//=========================================================================

//罗格里格斯公式===========================================================
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
