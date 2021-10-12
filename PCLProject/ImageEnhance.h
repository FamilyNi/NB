#pragma once

#include "utils.h";

/*说明：
	srcImg：[in]被增强图像
	dstImg：[out]增强后的图像
*/

/*对数增强
	计算公式：dstImg = c * log(srcImg + 0.5)
*/
void Img_LogEnhance(Mat& srcImg, Mat& dstImg, double c);

/*Gamma变换增强
	计算公式：dstImg = pow(srcImg / 255.0, gamma) * 255.0
	gamma < 1：图像整体变亮，对比对降低
	gamma > 1：图像整体变暗，对比对增加
*/
void Img_GammaEnhance(Mat& srcImg, Mat& dstImg, double gamma);

/*图像增强
	table：查找表
*/
void Img_Enhance(Mat& srcImg, Mat& table, Mat &dstImg);

/*haclon中的emphasize算子
	计算公式：dstImg = (srcImg - mean) * Factor + srcImg
	mean：均值滤波后的图像
	ksize：[in]均值滤波的大小
	factor：[in]比例因子
*/
void Img_EmphasizeEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize, double factor);

/*halcon中的illuminate算子：
	计算公式：
	mean：均值滤波后的图像
	ksize：[in]均值滤波的大小
	factor：[in]比例因子
*/
void Img_IlluminateEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize, double factor);

/*局部标准差图像增强：
	计算公式：dstImg = mean(srcImg) + G * (srcImg - mean(srcImg))
*/
void Img_LSDEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize);

void EnhanceTest();