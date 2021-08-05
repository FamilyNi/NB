#pragma once
#include "utils.h"
#include "opencv2/features2d.hpp"

//ͼ��ת�Ҷ�ͼ
void ImageToGray(Mat& srcImg, Mat& grayImg);

//��ȡSIFT�ǵ�
void ExtractSiftPt(Mat& srcIMg, Mat& targetImg, vector<KeyPoint>& srcPts, vector<KeyPoint>& targetPts);
