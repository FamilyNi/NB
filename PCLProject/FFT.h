#pragma once
/*已于2019年1月5号全部完成测试*/

#include "utils.h"

//显示图像频谱图
void ImgF_DisplayFreqImg(Mat& fftImg, Mat& freqImg);

//快速傅里叶变换
void ImgF_FFT(Mat& srcImg, Mat& complexImg);

//快速傅里叶逆变换
void ImgF_InvFFT(Mat& fftImg, Mat& invFFTImg);

//滤波器对称赋值
void IngF_SymmetricAssignment(Mat& filter);

//高斯低通滤波器
void ImgF_GetGaussianFilter(Mat &filter, int width, int height, double radius, int mode);

//带状滤波器
void ImgF_GetBandFilter(Mat &filter, int width, int height, double lr, double hr, int mode);

//巴特沃尔斯录波器
void ImgF_GetBLPFFilter(Mat &filter, int width, int height, double radius, int n, int mode);

//同态滤波器
void ImgF_GetHomoFilter(Mat &filter, int width, int height, double radius, double L, double H, double c);

void nb_fft_filter(Mat &srcImg, Mat &filter, Mat &dstImg);

void FFTTest();