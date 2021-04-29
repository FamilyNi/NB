// PCLProject.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "utils.h"
#include "PC_Filter.h"
#include "FitModel.h"
#include "PC_Seg.h"
#include "JC_Calibrate.h"
#include "PPFMatch.h"
#include "GrayCode.h"

int main()
{
	vector<cv::Mat> grayCodeImg;
	GenGrayCodeImg(grayCodeImg);

	vector<cv::Mat> phaseImgs;
	GenGatingImg(phaseImgs);

	cv::Mat phasePriVal;
	ComputePhasePriVal(phaseImgs, phasePriVal);

	cv::Mat warpPhaseImg;
	GrayCodeWarpPhase(grayCodeImg, phasePriVal, warpPhaseImg);

	int dec_num = 13;
	vector<bool> bin(4, 0);
	DecToBin(dec_num, bin);
	vector<bool> grayCode(4, 0);
	BinToGrayCode(bin, grayCode);
	int dec_num_ = 0;
	BinToDec(bin, dec_num_);

	vector<bool> bin_;
	GrayCodeToBin(grayCode, bin_);

	bitset<4> t;
	bitset<4> t1;
	int shijinzhi = 15;
	int a = shijinzhi;
	int index = 0;
	while (a != 0)
	{
		t[index] = a % 2;
		a /= 2;
		index++;
	}

	for (int i = 0; i < t.size() - 1; ++i)
	{
		t1[i] = t[i] ^ t[i + 1];
	}
	t1[3] = t[3];
	TestProgram();
	return(0);
}
