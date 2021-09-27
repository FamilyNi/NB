#pragma once
#include "utils.h"

/*说明：
	srcImg：[in]被增强图像
	dstImg：[out]增强后的图像
*/

enum IMG_SEG {
	IMG_SEG_LIGHT = 0,
	IMG_SEG_DARK = 1,
	IMG_SEG_EQUL = 2,
	IMG_SEG_NOTEQUL = 3
};

/*整体阈值分割*/
NB_API void Img_Seg(Mat& srcImg, Mat& dstImg, double thres, IMG_SEG mode);

/*选择灰度区间:
	thresVal1：[in]低阈值
	thresVal2：[in]高阈值
	mode：[in]二值化模式---IMG_SEG_LIGHT：gray > thresVal1 && gray < thresVal2
					   IMG_SEG_DARK：gray < thresVal1 && gray > thresVal2
*/
NB_API void Img_SelectGraySeg(Mat& srcImg, Mat& dstImg, uchar thresVal1, uchar thresVal2, IMG_SEG mode);

/*熵最大的阈值分割(熵越大系统越不稳定)：
	mode：[in]二值化模式---IMG_SEG_LIGHT：选择图像亮的部分
					   IMG_SEG_DARK：选择图像暗的部分
*/
NB_API void Img_MaxEntropySeg(Mat& srcImg, Mat& dstImg, IMG_SEG mode);

/*迭代自适应二值化
	eps：[in]终止条件
	mode：[in]二值化模式---IMG_SEG_LIGHT：选择图像亮的部分
					IMG_SEG_DARK：选择图像暗的部分
*/
NB_API void Img_IterTresholdSeg(Mat& srcImg, Mat& dstImg, double eps, IMG_SEG mode);

/*局部自适应阈值分割
	size：[in]滤波器大小
	stdDevScale：[in]标准差的缩放
	absThres：[in]绝对阈值
	mode：[in]滤波模式
*/
NB_API void Img_LocAdapThresholdSeg(Mat& srcImg, Mat& dstImg, cv::Size size, double stdDevScale, double absThres, IMG_SEG mode);

/*迟滞分割*/
NB_API void Img_HysteresisSeg(Mat& srcImg, Mat& dstImg, double thresVal1, double thresVal2);

/*Halcon中的点检测*/
NB_API void Img_DotImgSeg(Mat& srcImg, Mat& dstImg, int size, IMG_SEG mode);

/*区域生长*/
NB_API void Img_RegionGrowSeg(Mat& srcImg, Mat& labels, int dist_c, int dist_r, int thresVal, int minRegionSize);

//各向异性的图像分割
void NB_AnisImgSeg(Mat &srcImg, Mat &dstImg, int WS, double C_Thr, int lowThr, int highThr);


void ImgSegTest();
