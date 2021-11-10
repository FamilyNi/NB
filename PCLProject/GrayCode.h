#pragma once
#include "include/BaseOprFile/utils.h"

//ʮ����ת������
void DecToBin(const int dec_num, vector<bool>& bin);

//������תʮ����
void BinToDec(const vector<bool>& bin, int& dec_num);

//������ת������
void BinToGrayCode(const vector<bool>& bin, vector<bool>& grayCode);

//������ת������
void GrayCodeToBin(const vector<bool>& grayCode, vector<bool>& bin);

//����������ͼ��
void GenGrayCodeImg(vector<Mat>& grayCodeImgs);

//������դͼ��
void GenGatingImg(vector<Mat>& phaseImgs);

//������λ��ֵ
void ComputePhasePriVal(vector<Mat>& phaseImgs, Mat& phasePriVal);

//�������λ
void GrayCodeWarpPhase(vector<Mat>& grayCodeImgs, Mat& phaseImg, Mat& warpPhaseImg);