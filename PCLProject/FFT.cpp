#include "FFT.h"

//显示图像频谱图==========================================================
void ImgF_DisplayFreqImg(Mat& fftImg, Mat& freqImg)
{
	if (fftImg.type() != CV_32FC2)
		return;
	Mat planes[2];
	split(fftImg, planes);
	Mat mag;
	magnitude(planes[0], planes[1], mag);
	mag += Scalar::all(1);
	log(mag, mag);

	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(mag, freqImg, 0, 1, NORM_MINMAX);
}
//========================================================================

//快速傅里叶变换==========================================================
void ImgF_FFT(Mat& srcImg, Mat& complexImg)
{
	int r = srcImg.rows;
	int c = srcImg.cols;
	int M = getOptimalDFTSize(r);
	int N = getOptimalDFTSize(c);
	Mat padded;
	copyMakeBorder(srcImg, padded, 0, M - r, 0, N - c, cv::BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
}
//========================================================================

//快速傅里叶逆变换========================================================
void ImgF_InvFFT(Mat& fftImg, Mat& invFFTImg)
{
	Mat iDft[] = { Mat_<float>(), Mat::zeros(fftImg.size(), CV_32F) };
	idft(fftImg, fftImg);
	split(fftImg, iDft);
	magnitude(iDft[0], iDft[1], invFFTImg);
}
//========================================================================

//滤波器对称赋值==========================================================
void IngF_SymmetricAssignment(Mat& filter)
{
	if (filter.type() != CV_32FC2)
	{
		return;
	}
	int width = filter.cols, height = filter.rows;
	int step = 2 * width;
	float* pData = filter.ptr<float>(0);
	for (int y = height / 2; y < height; ++y)
	{
		int offset_1 = step * y;
		int y_ = height - 1 - y;
		int offset_2 = step * y_;
		for (int x = 0; x < width / 2; ++x)
		{
			pData[2 * x + offset_1] = pData[2 * x + offset_2];
			pData[2 * x + 1 + offset_1] = pData[2 * x + 1 + offset_2];
		}
	}

	for (int y = 0; y < height; ++y)
	{
		int offset_1 = step * y;
		int y_ = y > height / 2 ? height - 1 - y : y;
		int offset_2 = step * y_;
		for (int x = width / 2; x < width; ++x)
		{
			int x_ = width - x - 1;
			pData[2 * x + offset_1] = pData[2 * x_ + offset_2];
			pData[2 * x + 1 + offset_1] = pData[2 * x_ + 1 + offset_2];
		}
	}
}
//========================================================================

//gauss低通滤波器=========================================================
void ImgF_GetGaussianFilter(Mat &filter, int width, int height, double radius, int mode)
{
	if (!filter.empty())
	{
		filter.release();
	}
	if (mode == 0)
		filter = Mat(Size(width, height), CV_32FC2, Scalar::all(0));
	else if (mode == 1)
		filter = Mat(Size(width, height), CV_32FC2, Scalar::all(1));
	else
		return;
	double half_w = width / 2;
	double half_h = height / 2;
	radius = std::min(radius, std::min(half_w, half_h));

	float *pData = filter.ptr<float>(0);
	double radius_22 = 2 * radius * radius;
	int step = 2 * width;
	for (int y = 0; y < half_h; ++y)
	{
		int offset = step * y;
		for (int x = 0; x < half_w; ++x)
		{
			double r_ = x * x + y * y;
			double value = exp(-r_ / radius_22);
			value = mode == 0 ? value : 1 - value;
			pData[2 * x + offset] = value;
			pData[2 * x + 1 + offset] = value;
		}
	}
	IngF_SymmetricAssignment(filter);
}
//========================================================================

//常规带阻滤波器==========================================================
void ImgF_GetBandFilter(Mat &filter, int width, int height, double lr, double hr, int mode)
{
	double minLen = std::min(width, height) / 2.0;
	CV_CheckLT(lr, minLen, "最小半径不能大于图像最小边长的的二分之一");
	if (!filter.empty())
	{
		filter.release();
	}
	if (mode == 0)
		filter = Mat(Size(width, height), CV_32FC2, Scalar::all(0));
	else if (mode == 1)
		filter = Mat(Size(width, height), CV_32FC2, Scalar::all(1));
	else
		return;
	hr = std::min(hr, minLen);
	double lr_2 = lr * lr;
	double hr_2 = hr * hr;
	float *pData = filter.ptr<float>(0);
	int step = 2 * width;
	for (int y = 0; y < hr; y++)
	{
		int offset = step * y;
		for (int x = 0; x < hr; ++x)
		{
			float r_ = x * x + y * y;
			if (r_ > lr_2 && r_ < hr_2)
			{
				double value = mode == 0 ? 1 : 0;
				pData[2 * x + offset] = value;
				pData[2 * x + 1 + offset] = value;
			}
		}
	}
	for (int y = height - hr - 1; y < height; ++y)
	{
		int offset = step * y;
		int y_ = height - 1 - y;
		for (int x = 0; x < hr; ++x)
		{
			double r_ = x * x + y_ * y_;
			if (r_ > lr_2 && r_ < hr_2)
			{
				double value = mode == 0 ? 1 : 0;
				pData[2 * x + offset] = value;
				pData[2 * x + 1 + offset] = value;
			}
		}
	}
}
//========================================================================

//构建blpf滤波器==========================================================
void ImgF_GetBLPFFilter(Mat &filter, int width, int height, double radius, int n, int mode)
{
	if (!filter.empty())
	{
		filter.release();
	}
	if (mode == 0)
		filter = Mat(Size(width, height), CV_32FC2, Scalar::all(0));
	else if (mode == 1)
		filter = Mat(Size(width, height), CV_32FC2, Scalar::all(1));
	else
		return;
	double half_w = width / 2;
	double half_h = height / 2;
	radius = std::min(radius, std::min(half_w, half_h));

	int n_2 = 2 * n;
	float *pData = filter.ptr<float>(0);
	int step = 2 * width;
	double radius_2 = radius * radius;
	for (int y = 0; y < half_h; ++y)
	{
		int offset = step * y;
		for (int x = 0; x < half_w; ++x)
		{
			double r_ = x * x + y * y;
			double value = 1.0 / (1.0 + std::pow(r_ / radius_2, n_2));
			value = mode == 0 ? value : 1 - value;
			pData[2 * x + offset] = value;
			pData[2 * x + 1 + offset] = value;
		}
	}
	IngF_SymmetricAssignment(filter);
}
//========================================================================

//构建同态滤波器==========================================================
void ImgF_GetHomoFilter(Mat &filter, int width, int height, double radius, double L, double H, double c)
{
	if (!filter.empty())
	{
		filter.release();
	}
	float diff = H - L;
	filter = Mat(Size(width, height), CV_32FC2, Scalar::all(0));
	radius = std::min(radius, std::min(width / 2.0, height / 2.0));

	float *pData = filter.ptr<float>(0);
	int step = 2 * width;
	double radius_2 = radius * radius;
	for (int y = 0; y < height / 2; ++y)
	{
		int offset = step * y;
		for (int x = 0; x < width / 2; ++x)
		{
			double r_ = x * x + y * y;
			float value = (H - L) * (1 - exp(-c * r_ / radius_2)) + L;
			pData[2 * x + offset] = value;
			pData[2 * x + 1 + offset] = value;
		}
	}
	IngF_SymmetricAssignment(filter);
}
//========================================================================

//频率滤波================================================================
void nb_fft_filter(Mat &srcImg, Mat &filter, Mat &dstImg)
{
	int r = srcImg.rows;
	int c = srcImg.cols;
	int M = getOptimalDFTSize(r);
	int N = getOptimalDFTSize(c);
	copyMakeBorder(filter, filter, 0, M - r, 0, N - c, cv::BORDER_CONSTANT, Scalar::all(0));

	Mat fftImg;
	ImgF_FFT(srcImg, fftImg);

	Mat fftFilterImg = fftImg.mul(filter);

	Mat freqImg;
	ImgF_DisplayFreqImg(fftFilterImg, freqImg);

	Mat invFFTImg;
	ImgF_InvFFT(fftFilterImg, invFFTImg);
	Mat roi = invFFTImg(Rect(0, 0, c, r));
	double minVal, maxVal;
	minMaxLoc(roi, &minVal, &maxVal, NULL, NULL);
	roi = (roi - minVal)/ (maxVal - minVal) * 255.0;
	////cv::norm(roi, roi, NORM_MINMAX);
	//cv::exp(roi, roi);
	//roi = roi - 1;
	roi.convertTo(dstImg, CV_8UC1);
	//F2UChar(roi, dstImg);
}
//========================================================================

//FFT测试=================================================================
void FFTTest()
{
	string filename = "1.jpg";
	Mat img = imread(samples::findFile(filename), 0);
	Mat filter;
	int width = img.cols, height = img.rows;
	float radius = 100;
	ImgF_GetBLPFFilter(filter, width, height, 50, 100, 1);

	//resize(img, img, Size(1000, 700));
	//Mat imgF;
	//img.convertTo(imgF, CV_32FC1);
	//Mat imgExp;
	//log((imgF + 1), imgExp);
	//Mat filter;
	//get_blpf_filter(filter, img.cols, img.rows, 200, 2);
	Mat dstImg;
	nb_fft_filter(img, filter, dstImg);
	int t = 1;
}