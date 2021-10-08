#pragma once
#include "utils.h"

/*图像的灰度共生距*/
// 枚举灰度共生矩阵的方向
enum GLCM_ORIT
{
	GLCM_HORIZATION = 0,		// 水平
	GLCM_VERTICAL = 1,			// 垂直
	GLCM_ANGLE45 = 2,			// 45度角
	GLCM_ANGLE135 = 3			// 135度角
};

//计算灰度共生矩
void Img_ComputeGLCM(Mat& srcImg, Mat& MatGLCM, int grayLevel, GLCM_ORIT angle);

// 计算水平灰度共生矩阵
void Img_GLCM_0(Mat& srcImg, Mat& MatGLCM);

// 计算垂直灰度共生矩阵
void Img_GLCM_90(Mat& srcImg, Mat &MatGLCM);

// 计算 45 度灰度共生矩阵
void Img_GLCM_45(Mat& srcImg, Mat& MatGLCM);

// 计算 135 度灰度共生矩阵
void Img_GLCM_135(Mat& srcImg, Mat& MatGLCM);

//计算总频数
void cal_total_number(Mat &MatGLCM, int &number);

//计算灰度共生矩的能量/一致性
void Img_GLCMEnergy(Mat& MatGLCM, double& energy);

//获得灰度共生矩的区域同质性
void Img_GLCMHomogeneity(Mat& MatGLCM, double& homogeneity);

//获得会读共生矩的对比对
void Img_GLCMContrast(Mat& MatGLCM, double& contrast);

//获得灰度共生矩阵的相关度
void Img_GLCMCorrelation(Mat& MatGLCM, double& correlation);

//获得灰度共生矩的熵
void Img_GLCMEntropy(Mat& MatGLCM, double& entropy);

//获得ux、uy
void get_ux_uy(Mat &MatGLCM, float &ux, float &uy);

//获得sx、sy
void get_sx_sy(Mat &MatGLCM, float &ux, float &uy, float &sx, float &sy);
