#include "GrayCode.h"
#include "include/BaseOprFile/MathOpr.h"

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

//产生格雷码图像====================================================
void GenGrayCodeImg(vector<Mat>& grayCodeImgs)
{
	grayCodeImgs.resize(4);
	int imgW = 912;
	int imgH = 1040;
	for (int i = 0; i < grayCodeImgs.size(); ++i)
		grayCodeImgs[i] = Mat(Size(imgW, imgH), CV_8UC1, Scalar(0));
	int step = imgW / 16;
	for (int i = 0; i < 16; ++i)
	{
		vector<bool> bin(4, 0);
		DecToBin(i, bin);
		vector<bool> grayCode(4, 0);
		BinToGrayCode(bin, grayCode);
		for (int j = 0; j < grayCodeImgs.size(); ++j)
		{
			if (grayCode[j])
				grayCodeImgs[j].colRange(i * step, (i + 1) * step) = 255;
		}
	}
}
//==================================================================

//产生光栅图像======================================================
void GenGatingImg(vector<Mat>& phaseImgs)
{
	if (phaseImgs.size() != 0)
		phaseImgs.resize(0);
	phaseImgs.resize(3);
	int imgW = 912;
	int imgH = 1040;
	float feq = 16.0f;
	float lamda = imgW / feq;
	for (int i = 0; i < phaseImgs.size(); ++i)
	{
		phaseImgs[i] = Mat(Size(imgW, imgH), CV_8UC1, Scalar(0));
		if (phaseImgs[i].empty())
			return;
	}
	for (int i = 0; i < phaseImgs.size(); ++i)
	{
		Mat& phaseImg = phaseImgs[i];
		float phaseOff = i * CV_2PI / 3 - CV_PI / 2.0f;
		for (int i = 0; i < imgW; ++i)
		{
			phaseImg.colRange(i, (i + 1)) = (std::sinf(CV_2PI * i / lamda + phaseOff) + 1) * 100;
		}
	}
}
//==================================================================

//计算相位主值======================================================
void ComputePhasePriVal(vector<Mat>& phaseImgs, Mat& phasePriVal)
{
	if (phaseImgs.size() != 3)
		return;
	int r = phaseImgs[0].rows;
	int c = phaseImgs[0].cols;
	for (int i = 0; i < phaseImgs.size(); ++i)
	{
		if (phaseImgs[i].cols != c || phaseImgs[i].rows != r)
			return;
	}
	phasePriVal = Mat(r, c, CV_32FC1, Scalar(0));
	vector<uchar*> vp_PhaseImg(phaseImgs.size(), NULL);
	for (int i = 0; i < phaseImgs.size(); ++i)
	{
		vp_PhaseImg[i] = phaseImgs[i].ptr<uchar>(0);
	}
	float* pPhasePriVal = phasePriVal.ptr<float>(0);
	vector<float> I(phaseImgs.size(), 0.0f);
	for (int y = 0; y < r; ++y)
	{
		for (int x = 0; x < c; ++x)
		{
			for (int i = 0; i < phaseImgs.size(); ++i)
			{
				I[i] = (float)(vp_PhaseImg[i][x]);
			}
			float sinx = 1.7351 * (I[1] - I[2]);
			float cosx = 2.0f * I[0] - (I[1] + I[2]);
			pPhasePriVal[x] = -std::atan2(sinx, cosx) + CV_PI;
		}
		pPhasePriVal += c;
		for (int i = 0; i < phaseImgs.size(); ++i)
		{
			vp_PhaseImg[i] += c;
		}
	}
}
//==================================================================

//解包裹相位========================================================
void GrayCodeWarpPhase(vector<Mat>& grayCodeImg, Mat& phaseImg, Mat& warpPhaseImg)
{
	if (grayCodeImg.size() != 4)
		return;
	int r = phaseImg.rows;
	int c = phaseImg.cols;
	for (int i = 0; i < grayCodeImg.size(); ++i)
	{
		if (grayCodeImg[i].cols != c || grayCodeImg[i].rows != r)
			return;
	}
	warpPhaseImg = Mat(cv::Size(c, r), CV_32FC1, Scalar(0));
	vector<uchar*> vp_GrayCode(grayCodeImg.size(), NULL);
	for (int i = 0; i < grayCodeImg.size(); ++i)
	{
		vp_GrayCode[i] = grayCodeImg[i].ptr<uchar>(0);
	}
	float* pPhaseImg = phaseImg.ptr<float>(0);
	float* pWarpPhaseImg = warpPhaseImg.ptr<float>(0);
	for (int y = 0; y < r; ++y)
	{
		for (int x = 0; x < c; ++x)
		{
			vector<bool> grayCode(grayCodeImg.size(), 0);
			vector<bool> bin(grayCodeImg.size(), 0);
			for (int i = 0; i < grayCodeImg.size(); ++i)
			{
				grayCode[i] = vp_GrayCode[i][x];
			}
			//解码
			GrayCodeToBin(grayCode, bin);
			int dec_num = 0;
			BinToDec(bin, dec_num);
			pWarpPhaseImg[x] = (dec_num * CV_2PI + pPhaseImg[x]);
		}
		pPhaseImg += c;
		pWarpPhaseImg += c;
		for (int i = 0; i < vp_GrayCode.size(); ++i)
		{
			vp_GrayCode[i] += c;
		}
	}
}
//==================================================================