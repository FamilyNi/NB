#pragma once
#include "utils.h"
#include "opencv2/features2d.hpp"

//图像转灰度图
void ImageToGray(Mat& srcImg, Mat& grayImg);

//提取SIFT角点
void ExtractSiftPt(Mat& srcIMg, Mat& targetImg, vector<KeyPoint>& srcPts, vector<KeyPoint>& targetPts);
