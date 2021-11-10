#pragma once
#include "utils.h"

//点到平面的距离==============================================================
template <typename T1>
void PC_PtToPlaneDist(T1& pt, Plane3D& plane, double& dist)
{
	dist = abs(pt.x * plane.a + pt.y * plane.b + pt.z * plane.c + plane.d);
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
template <typename T1, typename T2, typename T3>
void PC_PtProjPlanePt(T1& pt, T2& plane, T3& projPt)
{
	float dist = pt.x * plane[0] + pt.y * plane[1] + pt.z * plane[2] + plane[3];
	projPt.x = pt.x - dist * plane[0];
	projPt.y = pt.y - dist * plane[1];
	projPt.z = pt.z - dist * plane[2];
}
//===========================================================================

//空间点到空间直线的距离=====================================================
template <typename T>
void PC_PtToLineDist(T& pt, Line3D& line, double& dist)
{
	double scale = pt.x * line.a + pt.y * line.b + pt.z * line.c -
		(line.x * line.a + line.y * line.b + line.z * line.c);
	double diff_x = line.x + scale * line.a - pt.x;
	double diff_y = line.y + scale * line.b - pt.y;
	double diff_z = line.z + scale * line.c - pt.z;
	dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
}
//===========================================================================

//空间点到空间直线的投影=====================================================
template <typename T1, typename T2, typename T3>
void PC_PtProjLinePt(T1& pt, T2& line, T3& projPt)
{
	float scale = pt.x * line[0] + pt.y * line[1] + pt.z * line[2] -
		(line[3] * line[0] + line[4] * line[1] + line[5] * line[2]);
	projPt.x = line[3] + scale * line[0];
	projPt.y = line[4] + scale * line[1];
	projPt.z = line[5] + scale * line[2];
}
//===========================================================================

//三维向量叉乘===============================================================
template <typename T1, typename T2, typename T3>
void PC_VecCross(T1& vec1, T2& vec2, T3& vec, bool isNormal)
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
template <typename T1, typename T2>
void Img_ComputePPDist(T1& pt1, T2& pt2, double& dist)
{
	double diff_x = pt1.x - pt2.x;
	double diff_y = pt1.y - pt2.y;
	return std::sqrt(max(diff_x * diff_x + diff_y * diff_y, EPS));
}
//=========================================================================

//平面上点到直线的投影=====================================================
template <typename T1, typename T2, typename T3>
void Img_PtProjLinePt(T1& pt, T2& line, T3& projPt)
{
	float scale = pt.x * line[0] + pt.y * line[1] - (line[2] * line[0] + line[3] * line[1]);
	projPt.x = line[2] + scale * line[0]; projPt.y = line[3] + scale * line[1];
}
//=========================================================================

//点到圆或者球的距离=======================================================
template <typename T1, typename T2>
void PC_PtToCircleDist(T1& pt, T2& circle, double& dist)
{
	double diff_x = pt.x - circle.x;
	double diff_y = pt.y - circle.y;
	double diff_z = pt.z - circle.z;
	dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
	dist = abs(dist - circle.r);
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


