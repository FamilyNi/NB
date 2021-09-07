#include "ImageSeg.h"
#include "OpenCV_Utils.h"

//熵最大的阈值分割(熵越大系统越不稳定)=====================================================
NB_API void Img_MaxEntropySeg(Mat &srcImg, Mat &dstImg)
{
	Mat hist;
	Img_ComputeImgHist(srcImg, hist);
	int thresVal = 0;
	double entropy = 0;
	double* pHist = hist.ptr<double>();
	for (int index = 0; index < 256; ++index)
	{
		//计算背景熵
		float b_p = 0.0;
		for (int i = 0; i < index; ++i)
		{
			b_p += pHist[i];
		}
		float b_entropy = 0.0;
		for (int i = 0; i < index; ++i)
		{
			float p_i = pHist[i] / b_p;
			b_entropy += -p_i * log(p_i + 1e-8);
		}
		//计算前景熵
		float f_p = 1 - b_p;
		float f_entropy = 0.0;
		for (int i = index; i < 256; ++i)
		{
			float p_i = pHist[i] / b_p;
			f_entropy += -p_i * log(p_i + 1e-8);
		}
		if (entropy < (b_entropy + f_entropy))
		{
			entropy = b_entropy + f_entropy;
			thresVal = index;
		}
	}
	cv::threshold(srcImg, dstImg, thresVal, 255, THRESH_BINARY_INV);
}
//=========================================================================================

//迭代自适应二值化=========================================================================
void Img_IterTresholdSeg(Mat &srcImg, Mat &region, double eps)
{
	Mat hist;
	Img_ComputeImgHist(srcImg, hist);
	double* pHist = hist.ptr<double>();
	float iniT = 0.0;
	for (int i = 0; i < 256; ++i)
		iniT += (i * pHist[i]);
	float m1 = 0.0;
	float m2 = 0.0;
	float T = 0.0;
	while (abs(iniT - T) < eps)
	{
		for (int i = 0; i < iniT; ++i)
		{
			m1 += (i * pHist[i]);
		}
		for (int i = (int)iniT; i < 256; ++i)
		{
			m2 += (i * pHist[i]);
		}
		T = iniT;
		iniT = (m1 + m2) / 2;
	}
	threshold(srcImg, region, iniT, 255, THRESH_BINARY);
}
//=========================================================================================

//各向异性的图像分割======================================================
void NB_AnisImgSeg(Mat &srcImg, Mat &dstImg, int WS, double C_Thr, int lowThr, int highThr)
{
	Mat imgCoherency, imgOrientation;
	Mat img;
	srcImg.convertTo(img, CV_32F);
	Mat imgDiffX, imgDiffY, imgDiffXY;
	Sobel(img, imgDiffX, CV_32F, 1, 0, 3);
	Sobel(img, imgDiffY, CV_32F, 0, 1, 3);
	multiply(imgDiffX, imgDiffY, imgDiffXY);

	Mat imgDiffXX, imgDiffYY;
	multiply(imgDiffX, imgDiffX, imgDiffXX);
	multiply(imgDiffY, imgDiffY, imgDiffYY);

	Mat J11, J22, J12;
	boxFilter(imgDiffXX, J11, CV_32F, Size(WS, WS));
	boxFilter(imgDiffYY, J22, CV_32F, Size(WS, WS));
	boxFilter(imgDiffXY, J12, CV_32F, Size(WS, WS));

	Mat tmp1, tmp2, tmp3, tmp4;
	tmp1 = J11 + J22;
	tmp2 = J11 - J22;
	multiply(tmp2, tmp2, tmp2);
	multiply(J12, J12, tmp3);
	sqrt(tmp2 + 4.0 * tmp3, tmp4);

	Mat lambda1, lambda2;
	lambda1 = tmp1 + tmp4;
	lambda2 = tmp1 - tmp4;

	divide(lambda1 - lambda2, lambda1 + lambda2, imgCoherency);
	phase(J22 - J11, 2.0*J12, imgOrientation, true);
	imgOrientation = 0.5*imgOrientation;


	Mat imgCoherencyBin;
	imgCoherencyBin = imgCoherency > C_Thr;
	Mat imgOrientationBin;
	inRange(imgOrientation, Scalar(lowThr), Scalar(highThr), imgOrientationBin);
	Mat imgBin;
	imgBin = imgCoherencyBin & imgOrientationBin;
	normalize(imgCoherency, imgCoherency, 0, 255, NORM_MINMAX);
	normalize(imgOrientation, imgOrientation, 0, 255, NORM_MINMAX);
}
//========================================================================



void ImgSegTest()
{
	string imgPath = "../image/b.bmp";
	Mat srcImg = imread(imgPath, 0);

	Mat dstImg;
	NB_AnisImgSeg(srcImg, dstImg, 5, 20, 10, 100);
	Mat t = dstImg.clone();
}