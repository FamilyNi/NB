#include "ImageFilter.h"

//引导滤波=============================================================================
void Img_GuidFilter(Mat& srcImg, Mat& guidImg, Mat& dstImg, int size, float eps)
{
	if (!dstImg.empty())
		dstImg.release();
	//CV_CheckEQ(srcImg.empty(), 1, "原始图像不存在");
	CV_CheckEQ(srcImg.size(), guidImg.size(), "引导图像错误");
	CV_CheckTypeEQ(srcImg.type(), guidImg.type(), "引导图像错误");
	cv::Size imgSize = srcImg.size();
	dstImg = Mat(imgSize, srcImg.type(), cv::Scalar(0));

	Mat SrcImg_32f(imgSize, CV_32F, cv::Scalar(0));
	Mat GuidImg_32f(imgSize, CV_32F, cv::Scalar(0));

	srcImg.convertTo(SrcImg_32f, CV_32F);
	guidImg.convertTo(GuidImg_32f, CV_32F);

	cv::Size winSize(size, size);
	//计算 I*I，I*P
	Mat img_IP = SrcImg_32f.mul(GuidImg_32f);
	Mat img_II = GuidImg_32f.mul(GuidImg_32f);

	//计算均值
	Mat mean_p, mean_I, mean_IP, mean_II;
	cv::boxFilter(srcImg, mean_p, CV_32F, winSize);
	cv::boxFilter(guidImg, mean_I, CV_32F, winSize);
	cv::boxFilter(img_IP, mean_IP, CV_32F, winSize);
	cv::boxFilter(img_II, mean_II, CV_32F, winSize);

	//计算 IP 的协方差矩阵以及 I 的方差矩阵
	Mat var_IP = mean_IP - mean_I.mul(mean_p);
	Mat var_II = mean_II - mean_I.mul(mean_I) + eps;

	//计算 a、b;
	Mat a, b;
	cv::divide(var_IP, var_II, a);
	b = mean_p - a.mul(mean_I);
	//计算 a、b 的均值
	Mat mean_a, mean_b;
	cv::boxFilter(a, mean_a, CV_32F, winSize);
	cv::boxFilter(b, mean_b, CV_32F, winSize);

	uchar* pDstImg = dstImg.ptr<uchar>();
	float* pFGuidImg = GuidImg_32f.ptr<float>();
	float* pMean_a = mean_a.ptr<float>();
	float* pMean_b = mean_b.ptr<float>();
	int step = imgSize.width * srcImg.channels();
	for (int y = 0; y < imgSize.height; ++y, pDstImg += step,
		pMean_a += step, pMean_b += step, pFGuidImg += step)
	{
		for (int x = 0; x < step; ++x)
		{
			pDstImg[x] = static_cast<uchar>(pMean_a[x] * pFGuidImg[x] + pMean_b[x]);
		}
	}
}
//=====================================================================================

//自适应Canny滤波======================================================================
void Img_AdaptiveCannyFilter(Mat& srcImg, Mat& dstImg, int size, double sigma)
{
	cv::Scalar midVal = cv::mean(srcImg);
	double minVal = midVal[0] * (1 - sigma);
	double maxVal = midVal[0] * (1 + sigma);
	cv::Canny(srcImg, dstImg, minVal, maxVal, size);
}
//=====================================================================================

//频率域滤波===========================================================================
void ImgF_FreqFilter(Mat& srcImg, Mat& dstImg, double radius, int mode)
{
	int imgH = srcImg.rows, imgW = srcImg.cols;
	Mat filter;
	int M = getOptimalDFTSize(imgH);
	int N = getOptimalDFTSize(imgW);
	ImgF_GetGaussianFilter(filter, N, M, radius, mode);
	Mat srcImg_32F;
	srcImg.convertTo(srcImg_32F, CV_32F);

	int channels = srcImg_32F.channels();
	vector<Mat> idftImg(srcImg_32F.channels());
	Mat* pSplitImg = new Mat[channels];
	Mat* pMergeImg = new Mat[channels];
	cv::split(srcImg_32F, pSplitImg);
#pragma omp parallel for
	for (int i = 0; i < channels; ++i)
	{
		Mat fftImg;
		ImgF_FFT(pSplitImg[i], fftImg);
		Mat fftFilterImg = fftImg.mul(filter);

		Mat invFFTImg;
		ImgF_InvFFT(fftFilterImg, invFFTImg);
		Mat roi = invFFTImg(Rect(0, 0, imgW, imgH));
		double minVal, maxVal;
		minMaxLoc(roi, &minVal, &maxVal, NULL, NULL);
		roi = (roi - minVal) / (maxVal - minVal) * 255.0;
		roi.convertTo(pMergeImg[i], CV_8UC1);
	}
	cv::merge(pMergeImg, channels, dstImg);
	delete[] pSplitImg;
	delete[] pMergeImg;
}
//=====================================================================================

//同泰滤波=============================================================================
void ImgF_HomoFilter(Mat& srcImg, Mat& dstImg, double radius, double L, double H, double c)
{
	int imgH = srcImg.rows, imgW = srcImg.cols;
	
	Mat filter;
	int M = getOptimalDFTSize(imgH);
	int N = getOptimalDFTSize(imgW);
	ImgF_GetHomoFilter(filter, N, M, radius, L, H, c);

	Mat srcImg_32F;
	srcImg.convertTo(srcImg_32F, CV_32F);
	int channels = srcImg_32F.channels();
	vector<Mat> idftImg(srcImg_32F.channels());
	Mat* pSplitImg = new Mat[channels];
	Mat* pMergeImg = new Mat[channels];
	cv::split(srcImg_32F, pSplitImg);
#pragma omp parallel for
	for (int i = 0; i < channels; ++i)
	{
		cv::log((pSplitImg[i] + 1), pSplitImg[i]);
		Mat fftImg;
		ImgF_FFT(pSplitImg[i], fftImg);
		Mat fftFilterImg = fftImg.mul(filter);
		Mat invFFTImg;
		ImgF_InvFFT(fftFilterImg, invFFTImg);
		Mat roi = invFFTImg(Rect(0, 0, imgW, imgH));
		cv::normalize(roi, roi, 0, 1, NORM_MINMAX);
		cv::exp(roi, roi);
		double minVal, maxVal;
		minMaxLoc(roi, &minVal, &maxVal, NULL, NULL);
		roi = (roi - minVal) / (maxVal - minVal) * 255.0;
		roi.convertTo(pMergeImg[i], CV_8UC1);
	}
	cv::merge(pMergeImg, channels, dstImg);
	delete[] pSplitImg;
	delete[] pMergeImg;
}
//=====================================================================================

void FilterTest()
{
	string imgPath = "1.jpg";
	Mat srcImg = imread(imgPath, 1);

	Mat dstImg;
	ImgF_HomoFilter(srcImg, dstImg, 10, 0.5, 1.5, 1);
	Mat t = dstImg.clone();
}