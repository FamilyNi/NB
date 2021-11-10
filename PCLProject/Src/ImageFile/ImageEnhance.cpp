#include "../../include/ImageFile/ImageEnhance.h"

//对数增强==================================================================
void Img_LogEnhance(Mat& srcImg, Mat& dstImg, double c)
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
void Img_GammaEnhance(Mat& srcImg, Mat& dstImg, double gamma)
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
void Img_Enhance(Mat& srcImg, Mat& table, Mat &dstImg)
{
	dstImg = Mat(srcImg.size(), srcImg.type(), Scalar(0));
	cv::LUT(srcImg, table, dstImg);
}
//=========================================================================

//haclon中的emphasize算子==================================================
void Img_EmphasizeEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize, double factor)
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
#pragma omp parallel for
	for (int y = 0; y < row; ++y)
	{
		int offset = y * step;
		for (int x = 0; x < step; x += channel)
		{
			for (int c_ = 0; c_ < channel; ++c_)
			{
				int offset_ = offset + x + c_;
				int src_x = (int)pSrc[offset_];
				int mean_x = (int)pMean[offset_];
				int value = (round((src_x - mean_x) * factor) + src_x);
				if (value > 0 && value < 256)
					pDst[offset_] = value;
				if (value > 255)
					pDst[offset_] = 255;
			}
		}
	}
}
//=========================================================================

//halcon中的illuminate算子=================================================
void Img_IlluminateEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize, double factor)
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
#pragma omp parallel for
	for (int y = 0; y < row; ++y)
	{
		int offset = y * step;
		for (int x = 0; x < step; x += channel)
		{
			for (int c_ = 0; c_ < channel; ++c_)
			{
				int offset_ = offset + x + c_;
				int src_x = (int)pSrc[offset_];
				int mean_x = (int)pMean[offset_];
				int value = (round((127 - mean_x) * factor) + src_x);
				if (value > 0 && value < 256)
					pDst[offset_] = value;
				if (value > 255)
					pDst[offset_] = 255;
			}
		}
	}
}
//==========================================================================

//局部标准差图像增强=======================================================
void Img_LSDEnhance(Mat &srcImg, Mat &dstImg, cv::Size ksize)
{
	dstImg = Mat(srcImg.size(), srcImg.type(), cv::Scalar(0));
	Mat meanImg(srcImg.size(), srcImg.type(), cv::Scalar(0));
	cv::boxFilter(srcImg, meanImg, meanImg.type(), ksize);
	int row = srcImg.rows;
	int col = srcImg.cols;
	int channel = srcImg.channels();
	int step = col * channel;
	uchar* pSrc = srcImg.ptr<uchar>(0);
	uchar* pMean = meanImg.ptr<uchar>(0);
	uchar* pDst = dstImg.ptr<uchar>(0);
	int half_y = ksize.height / 2;
	int half_x = ksize.width / 2;

	cv::Scalar mean, stdDev;
	cv::meanStdDev(srcImg, mean, stdDev);
	for (int y = 0; y < row; ++y)
	{
		int s_y = std::max(0, y - half_y);
		int e_y = std::min(row, y + half_y);
		int offset_y = y * step;
		for (int x = 0; x < col; ++x)
		{
			int s_x = std::max(0, x - half_x);
			int e_x = std::min(col, x + half_x);
			Mat rect = srcImg(Range(s_y, e_y), Range(s_x, e_x));
			cv::Scalar mean_, stdDev_;
			cv::meanStdDev(rect, mean_, stdDev_);
			int offset_x = offset_y + x * channel;
			for (int c_ = 0; c_ < channel; ++c_)
			{
				int offset_ = offset_x + c_;
				double src_x = (double)pSrc[offset_];
				double mean_x = (double)pMean[offset_];
				double value = (src_x - mean_x) * stdDev[c_] / stdDev_[c_] + mean_x;
				if (value > 0 && value < 256)
					pDst[offset_] = value;
				if (value > 255)
					pDst[offset_] = 255;
			}
		}
	}
}
//==========================================================================

//局部最大值最小值之差增强==================================================
void Img_GrayRangeEnhance(Mat& srcImg, Mat& dstImg, int mask_h, int mask_w)
{
	Mat maxImg(srcImg.size(), srcImg.type(), cv::Scalar(0));
	Mat minImg(srcImg.size(), srcImg.type(), cv::Scalar(0));

	Mat kenerl = getStructuringElement(MorphShapes::MORPH_RECT, cv::Size(mask_w, mask_h));
	cv::erode(srcImg, minImg, kenerl);
	cv::dilate(srcImg, maxImg, kenerl);

	if (dstImg.size() != srcImg.size() || dstImg.type() != srcImg.type())
		dstImg = Mat(srcImg.size(), srcImg.type(), cv::Scalar(0));
	uchar* pDst = dstImg.ptr<uchar>(0);
	uchar* pMax = maxImg.ptr<uchar>(0);
	uchar* pMin = minImg.ptr<uchar>(0);
	int row = srcImg.rows;
	int col = srcImg.cols;
	if (dstImg.isContinuous() && maxImg.isContinuous() && minImg.isContinuous())
	{
		col *= row; row = 1;
	}
	for (int y = 0; y < row; ++y, pDst += col, pMax += col, pMin += col)
	{
		for (int x = 0; x < col; ++x)
		{
			int value = (int)pMax[x] - (int)pMin[x];
			if (value > 0 && value < 256)
				pDst[x] = value;
			if (value > 255)
				pDst[x] = 255;
		}
	}
}
//==========================================================================

//Halcon中的点检测==========================================================
void Img_DotImgEnhance(Mat& srcImg, Mat& dstImg, int size, IMG_SEG mode)
{
	int length = size + 2;
	double center = length / 2;
	double radius1_2 = (size / 2.0) * (size / 2.0);
	double radisu2_2 = (length / 2.0) * (length / 2.0);
	double positive = 0;
	double negtive = 0;
	Mat kenerl = Mat(length, length, CV_32FC1, Scalar(0));
	float *pKenerl = kenerl.ptr<float>(0);
	for (int y = 0; y < length; ++y)
	{
		double y_2 = (y - center) * (y - center);
		for (int x = 0; x < length; ++x)
		{
			double x_2 = (x - center) * (x - center);
			if (x_2 + y_2 < radius1_2)
				positive--;
			else if (x_2 + y_2 < radisu2_2)
				negtive++;
		}
	}
	double sum = 0.0f;
	for (int y = 0; y < length; ++y, pKenerl += length)
	{
		double y_2 = (y - center) * (y - center);
		for (int x = 0; x < length; ++x)
		{
			double x_2 = (x - center) * (x - center);
			if (x_2 + y_2 < radius1_2)
				pKenerl[x] = negtive;
			else if (x_2 + y_2 < radisu2_2)
			{
				pKenerl[x] = positive;
				sum -= positive;
			}
		}
	}
	sum = std::max(sum, EPS);
	switch (mode)
	{
	case IMG_SEG::IMG_SEG_LIGHT:
		kenerl /= sum; break;
	case IMG_SEG::IMG_SEG_DARK:
		kenerl /= -sum; break;
	}
	dstImg = Mat(srcImg.size(), CV_8UC1, Scalar(0));
	cv::filter2D(srcImg, dstImg, CV_8UC1, kenerl);
}
//==========================================================================

void EnhanceTest()
{
	string imgPath = "C:/Users/Administrator/Desktop/testimage/halconimg.bmp";
	Mat srcImg = imread(imgPath, 0);

	Mat dstImg;
	Img_GrayRangeEnhance(srcImg, dstImg, 7, 7);
	Mat t1 = (srcImg - dstImg) * 100;
	Mat t = dstImg.clone();
}