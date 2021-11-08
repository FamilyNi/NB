#pragma once
#include "utils.h"
#include "OpenCV_Utils.h"

//两点计算直线==================================================================================
template <typename T>
void Img_TwoPtsComputeLine(T& pt1, T& pt2, Line2D& line)
{
	double vec_x = pt2.x - pt1.x;
	double vec_y = pt2.y - pt1.y;
	double norm_ = 1 / std::sqrt(vec_x * vec_x + vec_y * vec_y);
	line.a = -vec_y * norm_;
	line.b = vec_x * norm_;
	line.c = -(line.a * pt1.x + line.b * pt1.y);
}
//==============================================================================================

//三点求圆======================================================================================
template <typename T>
void Img_ThreePtsComputeCicle(T& pt1, T& pt2, T& pt3, Circle2D& circle)
{
	double B21 = pt2.x * pt2.x + pt2.y * pt2.y - (pt1.x * pt1.x + pt1.y * pt1.y);
	double B32 = pt3.x * pt3.x + pt3.y * pt3.y - (pt2.x * pt2.x + pt2.y * pt2.y);

	double X21 = pt2.x - pt1.x;
	double Y21 = pt2.y - pt1.y;
	double X32 = pt3.x - pt2.x;
	double Y32 = pt3.y - pt2.y;

	circle.x = 0.5 * (B21 * Y32 - B32 * Y21) / (X21 * Y32 - X32 * Y21);
	circle.y = 0.5 * (B21 * X32 - B32 * X21) / (Y21 * X32 - Y32 * X21);

	double diff_x = pt1.x - circle.x;
	double diff_y = pt1.y - circle.y;
	circle.r = std::sqrt(diff_x * diff_x + diff_y * diff_y);
}
//==============================================================================================

//六点求椭圆====================================================================================
template <typename T1, typename T2>
void Img_SixPtsComputeEllipse(vector<T1>& pts, T2& ellipse)
{
	if (pts.size() < 6)
		return;
	Mat coeffMat(cv::Size(6, 6), CV_64FC1, cv::Scalar(1.0f));
	double* pCoeff = coeffMat.ptr<double>();
	for (int i = 0; i < pts.size(); ++i)
	{
		int idx = 6 * i;
		double x = pts[i].x, y = pts[i].y;
		pCoeff[idx] = x * x; pCoeff[idx + 1] = x * y; pCoeff[idx + 2] = y * y;
		pCoeff[idx+3] = x; pCoeff[idx + 4] = y;
	}
	//cv::Mat res;
	cv::Vec6d res;
	SVD::solveZ(coeffMat, res);
	for (int i = 0; i < 6; ++i)
		ellipse[i] = res[i];
}
//==============================================================================================

//两点求线======================================================================================
template <typename T>
void PC_TwoPtsComputeLine(T& pt1, T& pt2, Line3D& line)
{
	double diff_x = pt2.x - pt1.x;
	double diff_y = pt2.y - pt1.y;
	double diff_z = pt2.z - pt1.z;
	double norm_ = 1.0 / std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
	line.a = diff_x * norm_; line.b = diff_y * norm_; line.c = diff_z * norm_;
	line.x = pt1.x; line.y = pt1.y; line.z = pt1.z;
}
//==============================================================================================

//三点计算平面==================================================================================
template <typename T>
void PC_ThreePtsComputePlane(T& pt1, T& pt2, T& pt3, Plane3D& plane)
{
	P_XYZ nor_1(pt1.x - pt2.x, pt1.y - pt2.y, pt1.z - pt2.z);
	P_XYZ nor_2(pt1.x - pt3.x, pt1.y - pt3.y, pt1.z - pt3.z);
	P_XYZ norm(0, 0, 0);
	PC_VecCross(nor_1, nor_2, norm, true);
	if (abs(norm.x) < EPS && abs(norm.y) < EPS && abs(norm.z) < EPS)
		return;
	plane.a = norm.x; plane.b = norm.y; plane.c = norm.z;
	plane.d = -(plane.a * pt1.x + plane.b * pt1.y + plane.c * pt1.z);
}
//==============================================================================================

//四点计算球====================================================================================
template <typename T>
void PC_FourPtsComputeSphere(vector<T>& pts, Sphere3D& sphere)
{
	if (pts.size() != 4)
		return;
	cv::Mat XYZ(cv::Size(3, 3), CV_64FC1, cv::Scalar(0));
	double* pXYZ = XYZ.ptr<double>();
	cv::Mat m(cv::Size(1, 3), CV_64FC1, cv::Scalar(0));
	double* pM = m.ptr<double>();
	for (int i = 0; i < pts.size() - 1; ++i)
	{
		int idx = 3 * i;
		pXYZ[idx] = pts[i].x - pts[i + 1].x;
		pXYZ[idx + 1] = pts[i].y - pts[i + 1].y;
		pXYZ[idx + 2] = pts[i].z - pts[i + 1].z;

		double pt0_d = pts[i].x * pts[i].x + pts[i].y * pts[i].y + pts[i].z * pts[i].z;
		double pt1_d = pts[i + 1].x * pts[i + 1].x + pts[i + 1].y * pts[i + 1].y + pts[i + 1].z * pts[i + 1].z;
		pM[i] = (pt0_d - pt1_d) / 2.0;
	}

	cv::Mat center = (XYZ.inv()) * m;
	sphere.x = center.ptr<double>(0)[0];
	sphere.y = center.ptr<double>(0)[1];
	sphere.z = center.ptr<double>(0)[2];
	double diff_x = pts[0].x - sphere.x;
	double diff_y = pts[0].y - sphere.y;
	double diff_z = pts[0].z - sphere.z;
	sphere.r = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
}
//==============================================================================================
