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