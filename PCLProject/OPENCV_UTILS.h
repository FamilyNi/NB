#pragma once
#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#ifdef NB
#define NB_API __declspec(dllexport)   
#else  
#define NB_API __declspec(dllimport)   
#endif

typedef struct IMGINFO
{
	int imgW;
	int imgH;
	int imgC;
	IMGINFO() :imgW(0), imgH(0), imgC(0)
	{}
};

typedef struct SPAPLEMODELINFO
{
	int pyrNumber;       // ����������
	int minContourLen;   //��������С����
	int maxContourLen;  //��������󳤶�
	int lowVal;    //������ȡ����ֵ
	int highVal;   //������ȡ����ֵ
	int extContouMode;  //������ȡģʽ
	int step;   //ѡ�㲽��
	float startAng;
	float endAng;
	float angStep;
	SPAPLEMODELINFO() :pyrNumber(1), minContourLen(0), maxContourLen(1e9),
		lowVal(15), highVal(30), step(3), extContouMode(0)
	{}
};

inline void ByteToMat(const uchar* inData, cv::Mat &srcImg, IMGINFO &imgInfo)
{
	if (imgInfo.imgC == 1)
	{
		srcImg = cv::Mat(imgInfo.imgH, imgInfo.imgW, CV_8UC1, (uchar *)inData);
	}
	if (imgInfo.imgC == 3)
	{
		srcImg = cv::Mat(imgInfo.imgH, imgInfo.imgW, CV_8UC3, (uchar *)inData);
	}
}
