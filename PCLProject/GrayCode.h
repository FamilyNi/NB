#pragma once
#include "include/BaseOprFile/utils.h"

//十进制转二进制
void DecToBin(const int dec_num, vector<bool>& bin);

//二进制转十进制
void BinToDec(const vector<bool>& bin, int& dec_num);

//二进制转格雷码
void BinToGrayCode(const vector<bool>& bin, vector<bool>& grayCode);

//格雷码转二进制
void GrayCodeToBin(const vector<bool>& grayCode, vector<bool>& bin);

//产生格雷码图像
void GenGrayCodeImg(vector<Mat>& grayCodeImgs);

//产生光栅图像
void GenGatingImg(vector<Mat>& phaseImgs);

//计算相位主值
void ComputePhasePriVal(vector<Mat>& phaseImgs, Mat& phasePriVal);

//解包裹相位
void GrayCodeWarpPhase(vector<Mat>& grayCodeImgs, Mat& phaseImg, Mat& warpPhaseImg);