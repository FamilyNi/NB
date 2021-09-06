#pragma once
#include "utils.h"
#include "opencv2/features2d.hpp"

//双线性插值
void BilinearInterpolation(const Mat& img, float x, float y, int& value);

//统计跳变数
int ComputeJumpNum(vector<bool>& res);

//提取LBP特征
void ExtractLBPFeature(const Mat& srcImg, Mat& lbpFeature, float raduis, int ptsNum);

//LBP检测直线
void LBPDetectLine(const Mat& srcImg, Mat& lbpFeature, float raduis, int ptsNum);

void LBPfeaturesTest();

void TestMMSER();
