#pragma once
#include "utils.h"

/*说明：
	srcImg：[in]被增强图像
	dstImg：[out]增强后的图像
*/

//熵最大的阈值分割(熵越大系统越不稳定)
NB_API void Img_MaxEntropySeg(Mat &srcImg, Mat &dstImg);

/*迭代自适应二值化
	eps：[in]终止条件
*/
NB_API void Img_IterTresholdSeg(Mat &srcImg, Mat &dstImg, double eps);

//各向异性的图像分割
void NB_AnisImgSeg(Mat &srcImg, Mat &dstImg, int WS, double C_Thr, int lowThr, int highThr);


void ImgSegTest();
