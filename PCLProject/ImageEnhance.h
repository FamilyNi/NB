#pragma once

#include "utils.h";

/*˵����
	srcImg��[in]����ǿͼ��
	dstImg��[out]��ǿ���ͼ��
*/

/*������ǿ
	���㹫ʽ��dstImg = c * log(srcImg + 0.5)
*/
NB_API void Img_LogEnhance(Mat& srcImg, Mat& dstImg, double c);

/*Gamma�任��ǿ
	���㹫ʽ��dstImg = pow(srcImg / 255.0, gamma) * 255.0
	gamma < 1��ͼ������������ԱȶԽ���
	gamma > 1��ͼ������䰵���Աȶ�����
*/
NB_API void Img_GammaEnhance(Mat& srcImg, Mat& dstImg, double gamma);

/*ͼ����ǿ
	table�����ұ�
*/
NB_API void Img_Enhance(Mat& srcImg, Mat& table, Mat &dstImg);

/*haclon�е�emphasize����
	���㹫ʽ��dstImg = (srcImg - mean) * Factor + srcImg
	mean����ֵ�˲����ͼ��
	ksize��[in]��ֵ�˲��Ĵ�С
	factor��[in]��������
*/
NB_API void Img_EmphasizeEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize, double factor);

//halcon�е�illuminate����================================================
NB_API void Img_IlluminateEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize, float factor);

/*ͼ���������ǿ
	���㹫ʽ��dstImg = a * srcImg + b;
*/
NB_API void Img_LinerEnhance(Mat& srcImg, Mat& dstImg, double a, double b);

void EnhanceTest();