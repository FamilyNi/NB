#include "ComputePts.h"

//平面两线求点===============================================================================
template <typename T>
void Img_LineComputePt(cv::Vec3d& line1, cv::Vec3d& line2, T& pt)
{
	if (abs(line1[0] * line2[0] + line1[1] * line2[1]) > 1.0 - EPS)
		return;
	pt.x = (line1[1] * line2[2] - line2[1] * line1[2]) /
		(line2[1] * line1[0] - line1[1] * line2[0]);
	pt.y = (line1[0] * line2[2] - line2[0] * line1[2]) /
		(line2[0] * line1[1] - line1[0] * line2[1]);
}
//===========================================================================================

//三面共点===================================================================================
template <typename T>
void PC_PlaneComputePt(cv::Vec4d& plane1, cv::Vec4d& plane2, cv::Vec4d& plane3, T& pt)
{
	double cosVal1 = plane1[0] * plane2[0] + plane1[1] * plane2[1] + plane1[2] * plane2[2];
	double cosVal2 = plane1[0] * plane3[0] + plane1[1] * plane3[1] + plane1[2] * plane3[2];
	double cosVal3 = plane2[0] * plane3[0] + plane2[1] * plane3[1] + plane2[2] * plane3[2];
	double eps_ = 1 - EPS;
	if (abs(cosVal1) > eps_ && abs(cosVal2) > eps_ && abs(cosVal3) > eps_)
		return;

	double e1 = plane2[2] * plane1[0] - plane1[2] * plane2[0];
	double f1 = plane2[2] * plane1[1] - plane1[2] * plane2[1];
	double g1 = -plane2[2] * plane1[3] + plane1[2] * plane2[3];

	double e2 = plane3[2] * plane2[0] - plane2[2] * plane3[0];
	double f2 = plane3[2] * plane2[1] - plane2[2] * plane3[1];
	double g2 = -plane3[2] * plane2[3] + plane2[2] * plane3[3];

	pt.y = (e2 * g1 - e1 * g2) / (e2 * f1 - e1 * f2);
	pt.x = (g1 - f1 * pt.y) / e1;
	pt.z = -(plane1[3] + plane1[0] * pt.x + plane1[1] * pt.y) / plane1[2];
}
//===========================================================================================

//空间两线距离最近的点=======================================================================
template <typename T>
void PC_LineNearestPt(Vec6d& line1, Vec6d& line2, T& pt1, T& pt2, double& dist)
{
	Mat A(cv::Size(2, 2), CV_32FC1, cv::Scalar(0));
	Mat B(cv::Size(1, 2), CV_32FC1, cv::Scalar(0));
	float* pA = A.ptr<float>();
	pA[0] = line1[0] * line1[0] + line1[1] * line1[1] + line1[2] * line1[2];
	pA[1] = -(line1[0] * line2[0] + line1[1] * line2[1] + line1[2] * line2[2]);
	pA[2] = pA[1];
	pA[3] = line2[0] * line2[0] + line2[1] * line2[1] + line2[2] * line2[2];

	float* pB = B.ptr<float>();
	float A1 = line1[0] * line1[3] + line1[1] * line1[4] + line1[2] * line1[5];
	float A2 = line1[0] * line2[3] + line1[1] * line2[4] + line1[2] * line2[5];
	float B1 = line2[0] * line2[3] + line2[1] * line2[4] + line2[2] * line2[5];
	float B2 = line2[0] * line1[3] + line2[1] * line1[4] + line2[2] * line1[5];
	pB[0] = A2 - A1;
	pB[1] = B2 - B1;
	Mat t = (A.inv()) * B;
	float* pT = t.ptr<float>(0);

	pt1.x = line1[0] * pT[0] + line1[3];
	pt1.y = line1[1] * pT[0] + line1[4];
	pt1.z = line1[2] * pT[0] + line1[5];

	pt2.x = line2[0] * pT[1] + line2[3];
	pt2.y = line2[1] * pT[1] + line2[4];
	pt2.z = line2[2] * pT[1] + line2[5];

	double diff_x = pt2.x - pt1.x;
	double diff_y = pt2.y - pt1.y;
	double diff_z = pt2.z - pt1.z;
	dist = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
}
//===========================================================================================