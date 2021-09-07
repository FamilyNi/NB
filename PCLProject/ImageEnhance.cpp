#include "ImageEnhance.h"

//对数增强==================================================================
NB_API void Img_LogEnhance(Mat& srcImg, Mat& dstImg, double c)
{
	CV_Assert(c > 0);
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(c * log(i + 1) + 0.5);
	Img_Enhance(srcImg, lookUpTable, dstImg);
}
//==========================================================================

//Gamma变换增强=============================================================
NB_API void Img_GammaEnhance(Mat& srcImg, Mat& dstImg, double gamma)
{
	CV_Assert(gamma >= 0);
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	Img_Enhance(srcImg, lookUpTable, dstImg);
}
//==========================================================================

//图像增强==================================================================
NB_API void Img_Enhance(Mat& srcImg, Mat& table, Mat &dstImg)
{
	dstImg = Mat(srcImg.size(), srcImg.type(), Scalar(0));
	cv::LUT(srcImg, table, dstImg);
}
//=========================================================================

//haclon中的emphasize算子==================================================
NB_API void Img_EmphasizeEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize, double factor)
{
	Mat meanImg = Mat(srcImg.size(), srcImg.type(),cv::Scalar(0));
	boxFilter(srcImg, meanImg, srcImg.type(), ksize);
	dstImg = Mat(srcImg.size(), srcImg.type(), cv::Scalar(0));
	int row = srcImg.rows;
	int col = srcImg.cols;
	int channel = srcImg.channels();
	uchar* pSrc = srcImg.ptr<uchar>();
	uchar* pMean = meanImg.ptr<uchar>();
	uchar* pDst = dstImg.ptr<uchar>();
	int step = channel * col;
#pragma omp parallel for num_threads(4)
	for (int y = 0; y < row; ++y)
	{
		int offset = y * step;
		for (int x = 0; x < step; x += channel)
		{
			for (int c_ = 0; c_ < channel; ++c_)
			{
				int offset_ = offset + x + c_;
				double src_x = (double)pSrc[offset_];
				double mean_x = (double)pMean[offset_];
				pDst[offset_] = static_cast<uchar>(round((src_x - mean_x) * factor) + src_x);
			}
		}
	}
}
//=========================================================================

//halcon中的illuminate算子=================================================
NB_API void Img_IlluminateEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize, float factor)
{
	Mat meanImg = Mat(srcImg.size(), srcImg.type(), cv::Scalar(0));
	boxFilter(srcImg, meanImg, srcImg.type(), ksize);
	dstImg = Mat(srcImg.size(), srcImg.type(), cv::Scalar(0));
	int row = srcImg.rows;
	int col = srcImg.cols;
	int channel = srcImg.channels();
	uchar* pSrc = srcImg.ptr<uchar>();
	uchar* pMean = meanImg.ptr<uchar>();
	uchar* pDst = dstImg.ptr<uchar>();
	int step = channel * col;
#pragma omp parallel for num_threads(4)
	for (int y = 0; y < row; ++y)
	{
		int offset = y * step;
		for (int x = 0; x < step; x += channel)
		{
			for (int c_ = 0; c_ < channel; ++c_)
			{
				int offset_ = offset + x + c_;
				double src_x = (double)pSrc[offset_];
				double mean_x = (double)pMean[offset_];
				pDst[offset_] = static_cast<uchar>(round((127 - mean_x) * factor + src_x));
			}
		}
	}
}
//==========================================================================

//图像的线性增强============================================================
NB_API void Img_LinerEnhance(Mat& srcImg, Mat& dstImg, double a, double b)
{
	dstImg = srcImg * a + b;
}
//==========================================================================

void EnhanceTest()
{
	string imgPath = "2.jpg";
	Mat srcImg = imread(imgPath, 0);

	Mat dstImg;
	Img_IlluminateEnhance(srcImg, dstImg, cv::Size(50,50), 0.6);
	Mat t1 = (srcImg - dstImg) * 100;
	Mat t = dstImg.clone();
}