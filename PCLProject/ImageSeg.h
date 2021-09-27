#pragma once
#include "utils.h"

/*˵����
	srcImg��[in]����ǿͼ��
	dstImg��[out]��ǿ���ͼ��
*/

enum IMG_SEG {
	IMG_SEG_LIGHT = 0,
	IMG_SEG_DARK = 1,
	IMG_SEG_EQUL = 2,
	IMG_SEG_NOTEQUL = 3
};

/*������ֵ�ָ�*/
NB_API void Img_Seg(Mat& srcImg, Mat& dstImg, double thres, IMG_SEG mode);

/*ѡ��Ҷ�����:
	thresVal1��[in]����ֵ
	thresVal2��[in]����ֵ
	mode��[in]��ֵ��ģʽ---IMG_SEG_LIGHT��gray > thresVal1 && gray < thresVal2
					   IMG_SEG_DARK��gray < thresVal1 && gray > thresVal2
*/
NB_API void Img_SelectGraySeg(Mat& srcImg, Mat& dstImg, uchar thresVal1, uchar thresVal2, IMG_SEG mode);

/*��������ֵ�ָ�(��Խ��ϵͳԽ���ȶ�)��
	mode��[in]��ֵ��ģʽ---IMG_SEG_LIGHT��ѡ��ͼ�����Ĳ���
					   IMG_SEG_DARK��ѡ��ͼ�񰵵Ĳ���
*/
NB_API void Img_MaxEntropySeg(Mat& srcImg, Mat& dstImg, IMG_SEG mode);

/*��������Ӧ��ֵ��
	eps��[in]��ֹ����
	mode��[in]��ֵ��ģʽ---IMG_SEG_LIGHT��ѡ��ͼ�����Ĳ���
					IMG_SEG_DARK��ѡ��ͼ�񰵵Ĳ���
*/
NB_API void Img_IterTresholdSeg(Mat& srcImg, Mat& dstImg, double eps, IMG_SEG mode);

/*�ֲ�����Ӧ��ֵ�ָ�
	size��[in]�˲�����С
	stdDevScale��[in]��׼�������
	absThres��[in]������ֵ
	mode��[in]�˲�ģʽ
*/
NB_API void Img_LocAdapThresholdSeg(Mat& srcImg, Mat& dstImg, cv::Size size, double stdDevScale, double absThres, IMG_SEG mode);

/*���ͷָ�*/
NB_API void Img_HysteresisSeg(Mat& srcImg, Mat& dstImg, double thresVal1, double thresVal2);

/*Halcon�еĵ���*/
NB_API void Img_DotImgSeg(Mat& srcImg, Mat& dstImg, int size, IMG_SEG mode);

/*��������*/
NB_API void Img_RegionGrowSeg(Mat& srcImg, Mat& labels, int dist_c, int dist_r, int thresVal, int minRegionSize);

//�������Ե�ͼ��ָ�
void NB_AnisImgSeg(Mat &srcImg, Mat &dstImg, int WS, double C_Thr, int lowThr, int highThr);


void ImgSegTest();
