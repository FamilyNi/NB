#pragma once
#include "utils.h"

//����������ͼ��
void GenGrayCodeImg(vector<Mat>& grayCodeImgs);

//������դͼ��
void GenGatingImg(vector<Mat>& phaseImgs);

//������λ��ֵ
void ComputePhasePriVal(vector<Mat>& phaseImgs, Mat& phasePriVal);

//�������λ
void GrayCodeWarpPhase(vector<Mat>& grayCodeImgs, Mat& phaseImg, Mat& warpPhaseImg);