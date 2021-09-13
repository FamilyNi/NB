#pragma once

#include "utils.h"

/*计算图像的直方图*/
void Img_ComputeImgHist(Mat& srcImg, Mat& hist);

/*绘制灰度直方图*/
void Img_DrawHistImg(Mat& hist);
