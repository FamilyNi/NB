#pragma once
#include "utils.h"

//产生格雷码图像
void GenGrayCodeImg(vector<Mat>& grayCodeImgs);

//产生光栅图像
void GenGatingImg(vector<Mat>& phaseImgs);

//计算相位主值
void ComputePhasePriVal(vector<Mat>& phaseImgs, Mat& phasePriVal);

//解包裹相位
void GrayCodeWarpPhase(vector<Mat>& grayCodeImgs, Mat& phaseImg, Mat& warpPhaseImg);