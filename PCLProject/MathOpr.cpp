#include "MathOpr.h"

//ʮ����ת������=====================================================
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

//������תʮ����=====================================================
void BinToDec(const vector<bool>& bin, int& dec_num)
{
	dec_num = 0;
	for (size_t i = 0; i < bin.size(); ++i)
	{
		dec_num += bin[i] * std::pow(2, (int)i);
	}
}
//===================================================================

//������ת������=====================================================
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

//������ת������=====================================================
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

//�ռ����߾�������ĵ�===============================================
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

//�ĵ������=========================================================
void ComputeSphere(vector<P_XYZ>& pts, double* pSphere)
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
	pSphere[0] = center.ptr<double>(0)[0];
	pSphere[1] = center.ptr<double>(0)[1];
	pSphere[2] = center.ptr<double>(0)[2];
	double diff_x = pts[0].x - pSphere[0];
	double diff_y = pts[0].y - pSphere[1];
	double diff_z = pts[0].z - pSphere[2];
	pSphere[3] = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
	return;
}
//===================================================================