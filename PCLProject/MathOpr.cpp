#include "MathOpr.h"

//点到平面的距离
double PC_PtToPlaneDist(P_XYZ& pt, cv::Vec4d& plane)
{
	return abs(pt.x * plane[0] + pt.y * plane[1] + pt.z * plane[2] + plane[3]);
}

//向量归一化
template <typename T>
void PC_VecNormal(T& p)
{
	float norm_ = 1 / std::max(std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z), 1e-8f);
	p.x *= norm_; p.y *= norm_; p.z *= norm_;
}

//点到平面的投影点
void PC_PtProjPlanePt(P_XYZ& pt, cv::Vec4d& plane, P_XYZ& projPt)
{
	float dist = pt.x * plane[0] + pt.y * plane[1] + pt.z * plane[2] + plane[3];
	projPt = { pt.x - float(dist * plane[0]), float(pt.x - dist * plane[1]),float(pt.x - dist * plane[2]) };
}

//空间点到空间直线的距离
double PC_3DPtTo3DLineDist(P_XYZ& pt, cv::Vec6d& line)
{
	double scale = pt.x * line[0] + pt.y * line[1] + pt.z * line[2] -
		(line[3] * line[0] + line[4] * line[1] + line[5] * line[2]);
	double diff_x = line[3] + scale * line[0] - pt.x;
	double diff_y = line[4] + scale * line[1] - pt.y;
	double diff_z = line[5] + scale * line[2] - pt.z;
	return std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
}

//空间点到空间直线的投影
void PC_3DPtProjLinePt(P_XYZ& pt, cv::Vec6d& line, P_XYZ& projPt)
{
	float scale = pt.x * line[0] + pt.y * line[1] + pt.z * line[2] -
		(line[3] * line[0] + line[4] * line[1] + line[5] * line[2]);
	projPt.x = line[3] + scale * line[0];
	projPt.y = line[4] + scale * line[1]; 
	projPt.z = line[5] + scale * line[2];
}

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

/*计算两点之间的距离--二维*/
template <typename T1, typename T2>
double Img_ComputePPDist(T1& pt1, T2& pt2)
{
	double diff_x = pt1.x - pt2.x;
	double diff_y = pt1.y - pt2.y;
	return std::sqrt(max(diff_x * diff_x + diff_y * diff_y, EPS));
}