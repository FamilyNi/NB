#include "GrayCode.h"

void GenGrayCodeImg()
{
	int imgW = 912;
	int imgH = 1040;
	cv::Mat grayCodeImg1(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	cv::Mat grayCodeImg2(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	cv::Mat grayCodeImg3(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	cv::Mat grayCodeImg4(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	int step = imgW / 16;
	for (int i = 0; i < 16; ++i)
	{
		vector<bool> bin(4, 0);
		DecToBin(i, bin);
		vector<bool> grayCode(4, 0);
		BinToGrayCode(bin, grayCode);
		if (grayCode[0])
			grayCodeImg1.colRange(i * step, (i + 1) * step) = 255;
		if (grayCode[1])
			grayCodeImg2.colRange(i * step, (i + 1) * step) = 255;
		if (grayCode[2])
			grayCodeImg3.colRange(i * step, (i + 1) * step) = 255;
		if (grayCode[3])
			grayCodeImg4.colRange(i * step, (i + 1) * step) = 255;
	}
	int ttttt = 0;
}

void GenGatingImg()
{
	int imgW = 912;
	int imgH = 1040;
	cv::Mat gatingImg1(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	cv::Mat gatingImg2(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	cv::Mat gatingImg3(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	float feq = 16.0f;
	float lamda = imgW / feq;
	for (int i = 0; i < imgW; ++i)
	{
		gatingImg1.colRange(i, (i + 1)) = (std::sinf(CV_2PI * i / lamda) + 1) * 100;
		gatingImg2.colRange(i, (i + 1)) = (std::sinf(CV_2PI * i / lamda + lamda * CV_PI) + 1) * 100;
		gatingImg3.colRange(i, (i + 1)) = (std::sinf(CV_2PI * i / lamda + lamda * CV_2PI) + 1) * 100;
	}
	int ttttt = 0;
}

void GrayCodeWarpPhase(vector<cv::Mat>& grayCodeImg, cv::Mat& phaseImg, cv::Mat& warpPhaseImg)
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
	warpPhaseImg = cv::Mat(cv::Size(c, r), CV_32FC1, cv::Scalar(0));
	vector<uchar*> vp_GrayCode(grayCodeImg.size());
	for (int i = 0; i < grayCodeImg.size(); ++i)
	{
		vp_GrayCode[i] = grayCodeImg[i].ptr<uchar>(0);
	}
	float* pPhaseImg = phaseImg.ptr<float>(0);
	float* pWarpPhaseImg = warpPhaseImg.ptr<float>(0);
	vector<bool> grayCode(grayCodeImg.size(), 0);
	vector<bool> bin(grayCodeImg.size(), 0);
	for (int y = 0; y < r; ++y)
	{
		int offset_y = y * c;
		pPhaseImg += offset_y;
		pWarpPhaseImg += offset_y;
		for (int i = 0; i < vp_GrayCode.size(); ++i)
		{
			vp_GrayCode[i] += offset_y;
		}
		for (int x = 0; x < c; ++x)
		{
			for (int i = 0; i < grayCodeImg.size(); ++i)
			{
				grayCode[i] = vp_GrayCode[x];
			}
			//½âÂë
			GrayCodeToBin(grayCode, bin);
			int dec_num = 0;
			BinToDec(bin, dec_num);
			pWarpPhaseImg[x] = dec_num * CV_2PI + std::atanf(pPhaseImg[x]);
		}
	}
}