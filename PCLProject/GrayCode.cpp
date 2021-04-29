#include "GrayCode.h"

void GenGrayCodeImg(vector<cv::Mat>& grayCodeImg)
{
	grayCodeImg.resize(4);
	int imgW = 912;
	int imgH = 1040;
	for (int i = 0; i < grayCodeImg.size(); ++i)
		grayCodeImg[i] = cv::Mat(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	//cv::Mat grayCodeImg1(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	//cv::Mat grayCodeImg2(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	//cv::Mat grayCodeImg3(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	//cv::Mat grayCodeImg4(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	int step = imgW / 16;
	for (int i = 0; i < 16; ++i)
	{
		vector<bool> bin(4, 0);
		DecToBin(i, bin);
		vector<bool> grayCode(4, 0);
		BinToGrayCode(bin, grayCode);
		for (int j = 0; j < grayCodeImg.size(); ++j)
		{
			if (grayCode[j])
				grayCodeImg[j].colRange(i * step, (i + 1) * step) = 255;
		}
		//if (grayCode[0])
		//	grayCodeImg1.colRange(i * step, (i + 1) * step) = 255;
		//if (grayCode[1])
		//	grayCodeImg2.colRange(i * step, (i + 1) * step) = 255;
		//if (grayCode[2])
		//	grayCodeImg3.colRange(i * step, (i + 1) * step) = 255;
		//if (grayCode[3])
		//	grayCodeImg4.colRange(i * step, (i + 1) * step) = 255;
	}
	cv::Mat grayCodeImg1 = grayCodeImg[0];
	cv::Mat grayCodeImg2 = grayCodeImg[1];
	cv::Mat grayCodeImg3 = grayCodeImg[2];
	cv::Mat grayCodeImg4 = grayCodeImg[3];
	int ttttt = 0;
}

void GenGatingImg(vector<cv::Mat>& phaseImgs)
{
	if (phaseImgs.size() != 0)
		phaseImgs.resize(0);
	phaseImgs.resize(3);
	int imgW = 912;
	int imgH = 1040;
	float feq = 8.0f;
	float lamda = imgW / feq;
	for (int i = 0; i < phaseImgs.size(); ++i)
	{
		phaseImgs[i] = cv::Mat(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
		if (phaseImgs[i].empty())
			return;
	}
	for (int i = 0; i < phaseImgs.size(); ++i)
	{
		cv::Mat& phaseImg = phaseImgs[i];
		float phaseOff = i * CV_2PI / 3;
		for (int i = 0; i < imgW; ++i)
		{
			phaseImg.colRange(i, (i + 1)) = (std::cosf(CV_2PI * i / lamda + phaseOff) + 1) * 100;
		}
	}
	cv::Mat gatingImg1 = phaseImgs[0];
	cv::Mat gatingImg2 = phaseImgs[1];
	cv::Mat gatingImg3 = phaseImgs[2];
	int ttttt = 0;
}

void ComputePhasePriVal(vector<cv::Mat>& phaseImgs, cv::Mat& phasePriVal)
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
	phasePriVal = cv::Mat(r, c, CV_32FC1, cv::Scalar(0));
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
			//float phase = cv::fastAtan2(sinx, cosx);
			pPhasePriVal[x] = -std::atan2(sinx, cosx);
		}
		pPhasePriVal += c;
		for (int i = 0; i < phaseImgs.size(); ++i)
		{
			vp_PhaseImg[i] += c;
		}
	}
	int ttttttt = 0;
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
			//½âÂë
			GrayCodeToBin(grayCode, bin);
			int dec_num = 0;
			BinToDec(bin, dec_num);
			pWarpPhaseImg[x] = dec_num * CV_2PI + pPhaseImg[x];
		}
		pPhaseImg += c;
		pWarpPhaseImg += c;
		for (int i = 0; i < vp_GrayCode.size(); ++i)
		{
			vp_GrayCode[i] += c;
		}
	}
	int ttttttt = 0;
}