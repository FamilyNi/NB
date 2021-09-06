#pragma once
/*����2019��1��5��ȫ����ɲ���*/

#include "utils.h"

//��ʾͼ��Ƶ��ͼ
void ImgF_DisplayFreqImg(Mat& fftImg, Mat& freqImg);

//���ٸ���Ҷ�任
void ImgF_FFT(Mat& srcImg, Mat& complexImg);

//���ٸ���Ҷ��任
void ImgF_InvFFT(Mat& fftImg, Mat& invFFTImg);

//�˲����ԳƸ�ֵ
void IngF_SymmetricAssignment(Mat& filter);

//��˹��ͨ�˲���
void ImgF_GetGaussianFilter(Mat &filter, int width, int height, double radius, int mode);

//��״�˲���
void ImgF_GetBandFilter(Mat &filter, int width, int height, double lr, double hr, int mode);

//�����ֶ�˹¼����
void ImgF_GetBLPFFilter(Mat &filter, int width, int height, double radius, int n, int mode);

//̬ͬ�˲���
void ImgF_GetHomoFilter(Mat &filter, int width, int height, double radius, double L, double H, double c);

void nb_fft_filter(Mat &srcImg, Mat &filter, Mat &dstImg);

void FFTTest();