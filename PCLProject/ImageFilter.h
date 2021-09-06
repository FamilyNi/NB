#pragma once
#include "utils.h";
#include "FFT.h"

/*说明：
	Img：开头表示空域滤波
	ImgF：开头表示频率域滤波
	srcImg：[in]被滤波图像
	dstImg：[out]滤波后的图像
*/

/*引导滤波:
	guidImg：[in]引导图像	
	size：[in]滤波器大小
*/
void Img_GuidFilter(Mat& srcImg, Mat& guidImg, Mat& dstImg, int size, float eps);

/*自适应Canny滤波
	size：[in]滤波器大小
	sigma：[in]高低阈值比例
*/
void Img_AdaptiveCannyFilter(Mat& srcImg, Mat& dstImg, int size, double sigma);

/*频率域滤波
	radius：[in]滤波半径
	mode：[in]表示低通或者高通
*/
void ImgF_FreqFilter(Mat& srcImg, Mat& dstImg, double radius, int mode);

/*同泰滤波：
	radius：[in]滤波半径
	L：[in]低分量
	H：[in]高分量
	c：[in]指数调节系数
*/
void ImgF_HomoFilter(Mat& srcImg, Mat& dstImg, double radius, double L, double H, double c);

void FilterTest();
