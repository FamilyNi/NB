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
NB_API void Img_GuidFilter(Mat& srcImg, Mat& guidImg, Mat& dstImg, int size, float eps);

/*����ӦCanny�˲�
	size��[in]�˲�����С
	sigma��[in]�ߵ���ֵ����
*/
NB_API void Img_AdaptiveCannyFilter(Mat& srcImg, Mat& dstImg, int size, double sigma);

/*Gabar�˲�
*/
NB_API void Img_GabarFilter(Mat& srcImg, Mat& dstImg);

/*Ƶ�����˲�
	srcImg����ͨ��ͼ��
	lr��[in]�˲��Ͱ뾶
	hr��[in]�˲��߰뾶
	�õ�ͨ�˲���ʱȡlr����״�˲������߶�ȡ����Ϊ BLPF �˲���ʱhrΪָ�� n
	passMode��[in]��ʾ��ͨ���߸�ͨ--0 ��ʾ��ͨ��1 ��ʾ��ͨ
	filterMode��[in]��ʾ�˲�������
*/
NB_API void ImgF_FreqFilter(Mat& srcImg, Mat& dstImg, double lr, double hr, int passMode, IMGF_MODE filterMode);

/*̩ͬ�˲���
	radius��[in]�˲��뾶
	L��[in]�ͷ���
	H��[in]�߷���
	c��[in]ָ������ϵ��
*/
NB_API void ImgF_HomoFilter(Mat& srcImg, Mat& dstImg, double radius, double L, double H, double c);

void FilterTest();
