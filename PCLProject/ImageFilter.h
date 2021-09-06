#pragma once
#include "utils.h";
#include "FFT.h"

/*˵����
	Img����ͷ��ʾ�����˲�
	ImgF����ͷ��ʾƵ�����˲�
	srcImg��[in]���˲�ͼ��
	dstImg��[out]�˲����ͼ��
*/

/*�����˲�:
	guidImg��[in]����ͼ��	
	size��[in]�˲�����С
*/
void Img_GuidFilter(Mat& srcImg, Mat& guidImg, Mat& dstImg, int size, float eps);

/*����ӦCanny�˲�
	size��[in]�˲�����С
	sigma��[in]�ߵ���ֵ����
*/
void Img_AdaptiveCannyFilter(Mat& srcImg, Mat& dstImg, int size, double sigma);

/*Ƶ�����˲�
	radius��[in]�˲��뾶
	mode��[in]��ʾ��ͨ���߸�ͨ
*/
void ImgF_FreqFilter(Mat& srcImg, Mat& dstImg, double radius, int mode);

/*̩ͬ�˲���
	radius��[in]�˲��뾶
	L��[in]�ͷ���
	H��[in]�߷���
	c��[in]ָ������ϵ��
*/
void ImgF_HomoFilter(Mat& srcImg, Mat& dstImg, double radius, double L, double H, double c);

void FilterTest();
