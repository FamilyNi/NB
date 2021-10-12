#pragma once

#include "utils.h";

/*˵����
	srcImg��[in]����ǿͼ��
	dstImg��[out]��ǿ���ͼ��
*/

/*������ǿ
	���㹫ʽ��dstImg = c * log(srcImg + 0.5)
*/
void Img_LogEnhance(Mat& srcImg, Mat& dstImg, double c);

/*Gamma�任��ǿ
	���㹫ʽ��dstImg = pow(srcImg / 255.0, gamma) * 255.0
	gamma < 1��ͼ������������ԱȶԽ���
	gamma > 1��ͼ������䰵���Աȶ�����
*/
void Img_GammaEnhance(Mat& srcImg, Mat& dstImg, double gamma);

/*ͼ����ǿ
	table�����ұ�
*/
void Img_Enhance(Mat& srcImg, Mat& table, Mat &dstImg);

/*haclon�е�emphasize����
	���㹫ʽ��dstImg = (srcImg - mean) * Factor + srcImg
	mean����ֵ�˲����ͼ��
	ksize��[in]��ֵ�˲��Ĵ�С
	factor��[in]��������
*/
void Img_EmphasizeEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize, double factor);

/*halcon�е�illuminate���ӣ�
	���㹫ʽ��
	mean����ֵ�˲����ͼ��
	ksize��[in]��ֵ�˲��Ĵ�С
	factor��[in]��������
*/
void Img_IlluminateEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize, double factor);

/*�ֲ���׼��ͼ����ǿ��
	���㹫ʽ��dstImg = mean(srcImg) + G * (srcImg - mean(srcImg))
*/
void Img_LSDEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize);

void EnhanceTest();