#pragma once
#include "OpenCV_Utils.h"

//显示图像频谱图
void ImgF_DisplayFreqImg(Mat& fftImg, Mat& freqImg);

//快速傅里叶变换
void ImgF_FFT(Mat& srcImg, Mat& complexImg);

//快速傅里叶逆变换
void ImgF_InvFFT(Mat& fftImg, Mat& invFFTImg);

//滤波器对称赋值
void IngF_SymmetricAssignment(Mat& filter);

//理想的单通滤波器
void ImgF_GetIdealFilter(Mat &filter, int imgW, int imgH, double radius, int passMode);

//高斯单通滤波器
void ImgF_GetGaussianFilter(Mat &filter, int imgW, int imgH, double radius, int passMode);

//带状滤波器
void ImgF_GetBandFilter(Mat &filter, int imgW, int imgH, double lr, double hr, int passMode);

//巴特沃尔斯录波器
void ImgF_GetBLPFFilter(Mat &filter, int imgW, int imgH, double radius, int n, int passMode);

//同态滤波器
void ImgF_GetHomoFilter(Mat &filter, int imgW, int imgH, double radius, double L, double H, double c);

//获取频率滤波器
void ImgF_GetFilter(Mat& filter, int imgW, int imgH, double lr, double hr, int passMode, IMGF_MODE filterMode);

void FFTTest();