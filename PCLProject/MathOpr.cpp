#include "MathOpr.h"

//十进制转二进制=====================================================
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
//===================================================================

//二进制转十进制=====================================================
void BinToDec(const vector<bool>& bin, int& dec_num)
{
	dec_num = 0;
	for (size_t i = 0; i < bin.size(); ++i)
	{
		dec_num += bin[i] * std::pow(2, (int)i);
	}
}
//===================================================================

//二进制转格雷码=====================================================
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
//===================================================================

//格雷码转二进制=====================================================
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
//===================================================================

//三维向量叉乘=======================================================
void VecCross_PC(P_XYZ& vec1, P_XYZ& vec2, P_XYZ& vec)
{
	vec.x = vec1.y * vec2.z - vec1.z * vec2.y;
	vec.y = vec1.z * vec2.x - vec1.x * vec2.z;
	vec.z = vec1.x * vec2.y - vec1.y * vec2.x;
}
//===================================================================

//向量归一化=========================================================
void Normal_PC(P_XYZ& p)
{
	float norm_ = 1 / std::max(std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z), 1e-8f);
	p.x *= norm_; p.y *= norm_; p.z *= norm_;
}
//==================================================================

//空间两线距离最近的点===============================================
float SpaceLineNearestPt(Vec6f& line1, Vec6f& line2, P_XYZ& pt1, P_XYZ& pt2)
{
	Mat A(cv::Size(2,2), CV_32FC1, cv::Scalar(0));
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

	float diff_x = pt2.x - pt1.x;
	float diff_y = pt2.y - pt1.y;
	float diff_z = pt2.z - pt1.z;
	return diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
}
//===================================================================

//四点计算球=========================================================
void ComputeSphere(vector<P_XYZ>& pts, Sphere& sphere)
{
	if (pts.size() < 4)
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
	sphere.c_x = center.ptr<double>(0)[0];
	sphere.c_y = center.ptr<double>(0)[1];
	sphere.c_z = center.ptr<double>(0)[2];
	double diff_x = pts[0].x - sphere.c_x;
	double diff_y = pts[0].y - sphere.c_y;
	double diff_z = pts[0].z - sphere.c_z;
	sphere.r = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
	return;
}
//===================================================================

//三面求点===========================================================
void ComputePtBasePlanes(Plane3D plane1, Plane3D plane2, Plane3D plane3, P_XYZ& point)
{
	double cosVal1 = plane1.a * plane2.a + plane1.b * plane2.b + plane1.c * plane2.c;
	double cosVal2 = plane1.a * plane3.a + plane1.b * plane3.b + plane1.c * plane3.c;
	double cosVal3 = plane2.a * plane3.a + plane2.b * plane3.b + plane2.c * plane3.c;
	double eps_ = 1 - EPS;
	if (abs(cosVal1) > eps_ && abs(cosVal2) > eps_ && abs(cosVal3) > eps_)
		return;

	double e1 = plane2.c * plane1.a - plane1.c * plane2.a;
	double f1 = plane2.c * plane1.b - plane1.c * plane2.b;
	double g1 = -plane2.c * plane1.d + plane1.c * plane2.d;

	double e2 = plane3.c * plane2.a - plane2.c * plane3.a;
	double f2 = plane3.c * plane2.b - plane2.c * plane3.b;
	double g2 = -plane3.c * plane2.d + plane2.c * plane3.d;

	point.y = (e2 * g1 - e1 * g2) / (e2 * f1 - e1 * f2);
	point.x = (g1 - f1 * point.y) / e1;
	point.z = -(plane1.d + plane1.a * point.x + plane1.b * point.y) / plane1.c;
}
//===================================================================

//三点求面===========================================================
void ComputePlane(P_XYZ& pt1, P_XYZ& pt2, P_XYZ& pt3, Plane3D& plane)
{
	P_XYZ nor_1(pt1.x - pt2.x, pt1.y - pt2.y, pt1.z - pt2.z);
	P_XYZ nor_2(pt1.x - pt3.x, pt1.y - pt3.y, pt1.z - pt3.z);
	P_XYZ norm(0,0,0);
	VecCross_PC(nor_1, nor_2, norm);
	if (abs(norm.x) < EPS && abs(norm.y) < EPS && abs(norm.z) < EPS)
		return;
	Normal_PC(norm);
	plane.a = norm.x; plane.b = norm.y; plane.c = norm.z;
	plane.d = -(plane.a * pt1.x + plane.b * pt1.y + plane.c * pt1.z);
}
//===================================================================