#pragma once
#include "utils.h"

/*˵����
	srcImg��[in]����ǿͼ��
	dstImg��[out]��ǿ���ͼ��
*/

//��������ֵ�ָ�(��Խ��ϵͳԽ���ȶ�)
NB_API void Img_MaxEntropySeg(Mat &srcImg, Mat &dstImg);

/*��������Ӧ��ֵ��
	eps��[in]��ֹ����
*/
NB_API void Img_IterTresholdSeg(Mat &srcImg, Mat &dstImg, double eps);

//�������Ե�ͼ��ָ�
void NB_AnisImgSeg(Mat &srcImg, Mat &dstImg, int WS, double C_Thr, int lowThr, int highThr);


void ImgSegTest();
