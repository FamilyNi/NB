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

//理想的单通滤波器========================================================
void ImgF_GetIdealFilter(Mat &filter, int imgW, int imgH, double radius, int passMode)
{
	if (!filter.empty())
	{
		filter.release();
	}
	if (passMode == 0)
		filter = Mat(Size(imgW, imgH), CV_32FC2, Scalar::all(0));
	else if (passMode == 1)
		filter = Mat(Size(imgW, imgH), CV_32FC2, Scalar::all(1));
	else
		return;
	double half_w = imgW / 2;
	double half_h = imgH / 2;
	radius = std::min(radius, std::min(half_w, half_h));

	float *pData = filter.ptr<float>(0);
	double radius_2 = radius * radius;
	int step = 2 * imgW;
	for (int y = 0; y < radius; ++y)
	{
		int offset = step * y;
		for (int x = 0; x < radius; ++x)
		{
			float r_ = x * x + y * y;
			if (r_ < radius_2)
			{
				double value = passMode == 0 ? 1 : 0;
				pData[2 * x + offset] = value;
				pData[2 * x + 1 + offset] = value;
			}
		}
	}
	IngF_SymmetricAssignment(filter);
}
//========================================================================

//gauss低通滤波器=========================================================
void ImgF_GetGaussianFilter(Mat &filter, int imgW, int imgH, double radius, int passMode)
{
	if (!filter.empty())
	{
		filter.release();
	}
	if (passMode == 0)
		filter = Mat(Size(imgW, imgH), CV_32FC2, Scalar::all(0));
	else if (passMode == 1)
		filter = Mat(Size(imgW, imgH), CV_32FC2, Scalar::all(1));
	else
		return;
	double half_w = imgW / 2;
	double half_h = imgH / 2;
	radius = std::min(radius, std::min(half_w, half_h));

	float *pData = filter.ptr<float>(0);
	double radius_22 = 2 * radius * radius;
	int step = 2 * imgW;
	for (int y = 0; y < half_h; ++y)
	{
		int offset = step * y;
		for (int x = 0; x < half_w; ++x)
		{
			double r_ = x * x + y * y;
			double value = exp(-r_ / radius_22);
			value = passMode == 0 ? value : 1 - value;
			pData[2 * x + offset] = value;
			pData[2 * x + 1 + offset] = value;
		}
	}
	IngF_SymmetricAssignment(filter);
}
//========================================================================

//常规带阻滤波器==========================================================
void ImgF_GetBandFilter(Mat &filter, int imgW, int imgH, double lr, double hr, int passMode)
{
	double minLen = std::min(imgW, imgH) / 2.0;
	CV_CheckLT(lr, minLen, "最小半径不能大于图像最小边长的的二分之一");
	if (!filter.empty())
	{
		filter.release();
	}
	if (passMode == 0)
		filter = Mat(Size(imgW, imgH), CV_32FC2, Scalar::all(0));
	else if (passMode == 1)
		filter = Mat(Size(imgW, imgH), CV_32FC2, Scalar::all(1));
	else
		return;
	hr = std::min(hr, minLen);
	double lr_2 = lr * lr;
	double hr_2 = hr * hr;
	float *pData = filter.ptr<float>(0);
	int step = 2 * imgW;
	for (int y = 0; y < hr; y++)
	{
		int offset = step * y;
		for (int x = 0; x < hr; ++x)
		{
			float r_ = x * x + y * y;
			if (r_ > lr_2 && r_ < hr_2)
			{
				double value = passMode == 0 ? 1 : 0;
				pData[2 * x + offset] = value;
				pData[2 * x + 1 + offset] = value;
			}
		}
	}
	IngF_SymmetricAssignment(filter);
}
//========================================================================

//构建blpf滤波器==========================================================
void ImgF_GetBLPFFilter(Mat &filter, int imgW, int imgH, double radius, int n, int passMode)
{
	if (!filter.empty())
	{
		filter.release();
	}
	if (passMode == 0)
		filter = Mat(Size(imgW, imgH), CV_32FC2, Scalar::all(0));
	else if (passMode == 1)
		filter = Mat(Size(imgW, imgH), CV_32FC2, Scalar::all(1));
	else
		return;
	double half_w = imgW / 2;
	double half_h = imgH / 2;
	radius = std::min(radius, std::min(half_w, half_h));

	int n_2 = 2 * n;
	float *pData = filter.ptr<float>(0);
	int step = 2 * imgW;
	double radius_2 = radius * radius;
	for (int y = 0; y < half_h; ++y)
	{
		int offset = step * y;
		for (int x = 0; x < half_w; ++x)
		{
			double r_ = x * x + y * y;
			double value = 1.0 / (1.0 + std::pow(r_ / radius_2, n_2));
			value = passMode == 0 ? value : 1 - value;
			pData[2 * x + offset] = value;
			pData[2 * x + 1 + offset] = value;
		}
	}
	IngF_SymmetricAssignment(filter);
}
//========================================================================

//构建同态滤波器==========================================================
void ImgF_GetHomoFilter(Mat &filter, int imgW, int imgH, double radius, double L, double H, double c)
{
	if (!filter.empty())
	{
		filter.release();
	}
	float diff = H - L;
	filter = Mat(Size(imgW, imgH), CV_32FC2, Scalar::all(0));
	radius = std::min(radius, std::min(imgW / 2.0, imgH / 2.0));

	float *pData = filter.ptr<float>(0);
	int step = 2 * imgW;
	double radius_2 = radius * radius;
	for (int y = 0; y < imgH / 2; ++y)
	{
		int offset = step * y;
		for (int x = 0; x < imgW / 2; ++x)
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

//获取频率滤波器==========================================================
void ImgF_GetFilter(Mat& filter, int imgW, int imgH, double lr, double hr, int passMode, IMGF_MODE filterMode)
{
	switch (filterMode)
	{
	case IMGF_IDEAL:
		ImgF_GetIdealFilter(filter, imgW, imgH, lr, passMode);
		break;
	case IMGF_GAUSSIAN:
		ImgF_GetGaussianFilter(filter, imgW, imgH, lr, passMode);
		break;
	case IMGF_BAND:
		ImgF_GetBandFilter(filter, imgW, imgH, lr, hr, passMode);
		break;
	case IMGF_BLPF:
		ImgF_GetBLPFFilter(filter, imgW, imgH, lr, (int)hr, passMode);
		break;
	default:
		break;
	}
}
//========================================================================
