#pragma once
#include "utils.h"

void GenGrayCodeImg(vector<cv::Mat>& grayCodeImgs);

void GenGatingImg(vector<cv::Mat>& phaseImgs);

void ComputePhasePriVal(vector<cv::Mat>& phaseImgs, cv::Mat& phasePriVal);

void GrayCodeWarpPhase(vector<cv::Mat>& grayCodeImgs, cv::Mat& phaseImg, cv::Mat& warpPhaseImg);