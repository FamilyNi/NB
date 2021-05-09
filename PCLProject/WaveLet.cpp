#include "WaveLet.h"

void wavelet(const string& name, cv::Mat &lowFilter, cv::Mat &highFilter)
{
	if (name == "haar" || name == "db1")
	{
		int N = 2;
		lowFilter = cv::Mat::zeros(1, N, CV_32F);
		highFilter = cv::Mat::zeros(1, N, CV_32F);
		lowFilter.at<float>(0, 0) = 1 / sqrtf(N);
		lowFilter.at<float>(0, 1) = 1 / sqrtf(N);

		highFilter.at<float>(0, 0) = -1 / sqrtf(N);
		highFilter.at<float>(0, 1) = 1 / sqrtf(N);
	}
	if (name == "sym2")
	{
		int N = 4;
		float h[] = { -0.483, 0.836, -0.224, -0.129 };
		float l[] = { -0.129, 0.224,    0.837, 0.483 };

		lowFilter = cv::Mat::zeros(1, N, CV_32F);
		highFilter = cv::Mat::zeros(1, N, CV_32F);

		for (int i = 0; i < N; i++)
		{
			lowFilter.at<float>(0, i) = l[i];
			highFilter.at<float>(0, i) = h[i];
		}
	}
}

void waveletDecompose(const cv::Mat &srcImg, const cv::Mat &lowFilter, const cv::Mat &highFilte, cv::Mat& dstImg)
{
	dstImg = cv::Mat_<float>(srcImg);
	int c = srcImg.cols;
	const cv::Mat &_lowFilter = cv::Mat_<float>(lowFilter);
	const cv::Mat &_highFilter = cv::Mat_<float>(highFilte);
	
	//频域滤波或时域卷积；ifft( fft(x) * fft(filter)) = cov(x,filter) 
	cv::Mat dst1 = cv::Mat::zeros(1, c, dstImg.type());
	cv::Mat dst2 = cv::Mat::zeros(1, c, dstImg.type());
	cv::filter2D(dstImg, dst1, -1, _lowFilter);
	cv::filter2D(dstImg, dst2, -1, _highFilter);

	//下采样
	cv::Mat downDst1 = cv::Mat::zeros(1, c / 2, dstImg.type());
	cv::Mat downDst2 = cv::Mat::zeros(1, c / 2, dstImg.type());
	cv::resize(dst1, downDst1, downDst1.size());
	cv::resize(dst2, downDst2, downDst2.size());

	//数据拼接
	for (int i = 0; i < c / 2; ++i)
	{
		dstImg.at<float>(0,i) = downDst1.at<float>(0, i);
		dstImg.at<float>(0, i + c / 2) = downDst2.at<float>(0, i);
	}
}

void WDT(cv::Mat &srcImg, cv::Mat& dstImg, const string& name, const int level)
{
	int r = srcImg.rows;
	int c = srcImg.cols;
	cv::Mat _src = cv::Mat_<float>(srcImg);
	dstImg = cv::Mat::zeros(r, c, _src.type());
	cv::Mat lowFilter, highFilter;
	wavelet(name, lowFilter, highFilter);
	int t = 1;
	while (t <= level)
	{
		//行小波变换
		for (int y = 0; y < r; ++y)
		{
			cv::Mat oneRow = cv::Mat::zeros(1, c, _src.type());
			for (int x = 0; x < c; ++x)
			{
				oneRow.at<float>(0, x) = _src.at<float>(y, x);
			}
			waveletDecompose(oneRow, lowFilter, highFilter, oneRow);
			for (int x = 0; x < c; ++x)
			{
				dstImg.at<float>(y, x) = oneRow.at<float>(0, x);
			}
		}

		cv::Mat ucharMat;
		FloatMatToUcharMat(dstImg, ucharMat);
		//列小波变换
		for (int x = 0; x < c; ++x)
		{
			cv::Mat oneCol = cv::Mat::zeros(r, 1, _src.type());
			for (int y = 0; y < r; ++y)
			{
				oneCol.at<float>(y, 0) = dstImg.at<float>(y, x);
			}
			waveletDecompose(oneCol.t(), lowFilter, highFilter, oneCol);
			cv::Mat oneCol_t = oneCol.t();
			for (int y = 0; y < r; ++y)
			{
				dstImg.at<float>(y, x) = oneCol_t.at<float>(y, 0);
			}
		}
		r /= 2;
		c /= 2;
		++t;
		_src = dstImg;
	}
}


void WaveLetTest()
{
	cv::Mat image = cv::imread("F:/nbcode/1.jpg", 0);
	cv::Mat dstImg;
	WDT(image, dstImg, "sym2", 3);

	cv::Mat ucharMat;
	FloatMatToUcharMat(dstImg, ucharMat);
}