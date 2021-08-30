#pragma once
#include "utils.h"

//˫���Բ�ֵ
void BilinearInterpolation(const Mat& img, float x, float y, int& value);

//ͳ��������
int ComputeJumpNum(vector<bool>& res);

//��ȡLBP����
void ExtractLBPFeature(const Mat& srcImg, Mat& lbpFeature, float raduis, int ptsNum);

//LBP���ֱ��
void LBPDetectLine(const Mat& srcImg, Mat& lbpFeature, float raduis, int ptsNum);

void LBPfeaturesTest();
