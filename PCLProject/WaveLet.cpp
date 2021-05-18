#include "WaveLet.h"

WaveLetTransformer::WaveLetTransformer(const string &name, uint level) :
	m_Level(level)
{
	SetFilter(name);
}

void WaveLetTransformer::SetFilter(const string& name)
{
	m_Name = name;
	if (!m_LowFilter.empty())
		m_LowFilter.release();
	if (!m_HighFilter.empty())
		m_HighFilter.release();

	if (name == "haar" || name == "db1")
	{
		int N = 2;
		m_LowFilter = cv::Mat::zeros(1, N, CV_32F);
		m_HighFilter = cv::Mat::zeros(1, N, CV_32F);
		m_LowFilter.at<float>(0, 0) = 1 / sqrtf(N);
		m_LowFilter.at<float>(0, 1) = 1 / sqrtf(N);

		m_HighFilter.at<float>(0, 0) = -1 / sqrtf(N);
		m_HighFilter.at<float>(0, 1) = 1 / sqrtf(N);
	}
	if (name == "sym2")
	{
		int N = 4;
		float h[] = { -0.483, 0.836, -0.224, -0.129 };
		float l[] = { -0.129, 0.224, 0.837, 0.483 };

		m_LowFilter = cv::Mat::zeros(1, N, CV_32F);
		m_HighFilter = cv::Mat::zeros(1, N, CV_32F);

		for (int i = 0; i < N; i++)
		{
			m_LowFilter.at<float>(0, i) = l[i];
			m_HighFilter.at<float>(0, i) = h[i];
		}
	}
}

void WaveLetTransformer::Set_I_Filter(const string& name)
{
	if (!m_Low_I_Filter.empty())
		m_Low_I_Filter.release();
	if (!m_HighFilter.empty())
		m_HighFilter.release();

	if (name == "haar" || name == "db1")
	{
		int N = 2;
		m_Low_I_Filter = cv::Mat::zeros(1, N, CV_32F);
		m_High_I_Filter = cv::Mat::zeros(1, N, CV_32F);

		m_Low_I_Filter.at<float>(0, 0) = 1 / sqrtf(N);
		m_Low_I_Filter.at<float>(0, 1) = 1 / sqrtf(N);

		m_High_I_Filter.at<float>(0, 0) = 1 / sqrtf(N);
		m_High_I_Filter.at<float>(0, 1) = -1 / sqrtf(N);
	}
	if (name == "sym2")
	{
		int N = 4;
		float h[] = { -0.1294,-0.2241,0.8365,-0.4830 };
		float l[] = { 0.4830, 0.8365, 0.2241, -0.1294 };

		m_Low_I_Filter = cv::Mat::zeros(1, N, CV_32F);
		m_High_I_Filter = cv::Mat::zeros(1, N, CV_32F);

		for (int i = 0; i < N; i++)
		{
			m_Low_I_Filter.at<float>(0, i) = l[i];
			m_High_I_Filter.at<float>(0, i) = h[i];
		}
	}
}

void WaveLetTransformer::GetOddR(const cv::Mat& srcImg, cv::Mat& oddRImg)
{
	int c = srcImg.cols;
	int r = srcImg.rows / 2;
	oddRImg = cv::Mat(cv::Size(c, r), srcImg.type(), cv::Scalar(0.0f));
	for (int i = 0; i < r; ++i)
	{
		srcImg.row(2 * i).copyTo(oddRImg.row(i));
	}
}

void WaveLetTransformer::GetOddC(const cv::Mat& srcImg, cv::Mat& oddCImg)
{
	int c = srcImg.cols / 2;
	int r = srcImg.rows;
	oddCImg = cv::Mat(cv::Size(c, r), srcImg.type(), cv::Scalar(0.0f));
	for (int i = 0; i < c; ++i)
	{
		srcImg.col(2 * i).copyTo(oddCImg.col(i));
	}
}

//小波分解
void WaveLetTransformer::WaveletDT(const cv::Mat& srcImg)
{
	cv::Mat src = cv::Mat_<float>(srcImg);
	int c = srcImg.cols;
	int r = srcImg.rows;
	m_Decompose = cv::Mat(cv::Size(c, r), src.type(), cv::Scalar(0));
	cv::Mat m_LowFilter_T = m_LowFilter.t();
	cv::Mat m_HighFilter_T = m_HighFilter.t();
	for (int i = 0; i < m_Level; ++i)
	{
		//行滤波
		cv::Mat dstLowR_, dstHighR_;
	
		cv::filter2D(src, dstLowR_, -1, m_LowFilter); //低通滤波---可加速，但需要自己写，麻烦
		cv::filter2D(src, dstHighR_, -1, m_HighFilter); //高通滤波---可加速，但需要自己写，麻烦
		
		cv::Mat dstLowR, dstHighR;
		GetOddC(dstLowR_, dstLowR);
		GetOddC(dstHighR_, dstHighR);

		//列滤波
		cv::Mat CMat1_, CMat2_, CMat3_, CMat4_;
		//行低频部分
		cv::filter2D(dstLowR, CMat1_, -1, m_LowFilter_T); //低通滤波
		cv::filter2D(dstLowR, CMat2_, -1, m_HighFilter_T); //高通滤波
		//行高频部分
		cv::filter2D(dstHighR, CMat3_, -1, m_LowFilter_T); //低通滤波
		cv::filter2D(dstHighR, CMat4_, -1, m_HighFilter_T); //高通滤波

		cv::Mat CMat1, CMat2, CMat3, CMat4;
		GetOddR(CMat1_, CMat1);
		GetOddR(CMat2_, CMat2);
		GetOddR(CMat3_, CMat3);
		GetOddR(CMat4_, CMat4);

		r /= 2; c /= 2;
		CMat1.copyTo(m_Decompose(cv::Rect(0, 0, c, r)));
		CMat2.copyTo(m_Decompose(cv::Rect(c, 0, c, r)));
		CMat3.copyTo(m_Decompose(cv::Rect(0, r, c, r)));
		CMat4.copyTo(m_Decompose(cv::Rect(c, r, c, r)));
		src = CMat1;		
	}
}

//列方向插值
void WaveLetTransformer::InterC(const cv::Mat& srcImg, cv::Mat& oddCImg)
{
	int c = srcImg.cols;
	if (oddCImg.cols != 2 * c)
		return;
	for (int i = 0; i < c; ++i)
	{
		srcImg.col(i).copyTo(oddCImg.col(2 * i));
	}
}

//行方向插值
void WaveLetTransformer::InterR(const cv::Mat& srcImg, cv::Mat& oddCImg)
{
	int r = srcImg.rows;
	if (oddCImg.rows != 2 * r)
		return;
	for (int i = 0; i < r; ++i)
	{
		srcImg.row(i).copyTo(oddCImg.row(2 * i));
	}
}

//小波重建
void WaveLetTransformer::IWaveletDT(cv::Mat& outMatImg)
{
	Set_I_Filter(m_Name);
	cv::Mat m_LowFilter_T = m_Low_I_Filter.t();
	cv::Mat m_HighFilter_T = m_High_I_Filter.t();
	int r = m_Decompose.rows / pow(2, m_Level);
	int c = m_Decompose.cols / pow(2, m_Level);
	for (int i = 0; i < m_Level; ++i)
	{
		cv::Mat CMat1, CMat2, CMat3, CMat4;
		CMat1 = m_Decompose(cv::Rect(0, 0, c, r));
		CMat2 = m_Decompose(cv::Rect(c, 0, c, r));
		CMat3 = m_Decompose(cv::Rect(0, r, c, r));
		CMat4 = m_Decompose(cv::Rect(c, r, c, r));

		r *= 2;
		cv::Mat CMat1_(r, c, CMat1.type(), cv::Scalar(0.0f));
		cv::Mat CMat2_(r, c, CMat2.type(), cv::Scalar(0.0f));
		cv::Mat CMat3_(r, c, CMat3.type(), cv::Scalar(0.0f));
		cv::Mat CMat4_(r, c, CMat4.type(), cv::Scalar(0.0f));

		InterR(CMat1, CMat1_);
		InterR(CMat2, CMat2_);
		InterR(CMat3, CMat3_);
		InterR(CMat4, CMat4_);

		//行低频部分
		cv::Mat CMat1_L, CMat1_H, oddImgR1;
		cv::filter2D(CMat1_, CMat1_L, -1, m_LowFilter_T); //低通滤波
		cv::filter2D(CMat2_, CMat1_H, -1, m_HighFilter_T); //高通滤波
		oddImgR1 = CMat1_L + CMat1_H;
		//行高频部分
		cv::Mat CMat2_L, CMat2_H, oddImgR2;
		cv::filter2D(CMat3_, CMat2_L, -1, m_LowFilter_T); //低通滤波
		cv::filter2D(CMat4_, CMat2_H, -1, m_HighFilter_T); //高通滤波
		oddImgR2 = CMat2_L + CMat2_H;

		c *= 2;
		cv::Mat RMat1_(r, c, CMat1.type(), cv::Scalar(0.0f));
		cv::Mat RMat2_(r, c, CMat2.type(), cv::Scalar(0.0f));


		InterC(oddImgR1, RMat1_);
		InterC(oddImgR2, RMat2_);
		cv::Mat RMat1_L, RMat1_H, oddImgR;
		cv::filter2D(RMat1_, RMat1_L, -1, m_Low_I_Filter); //低通滤波
		cv::filter2D(RMat2_, RMat1_H, -1, m_High_I_Filter); //高通滤波
		oddImgR = (RMat1_L + RMat1_H);

		cv::Mat ucharMat;
		FloatMatToUcharMat(oddImgR, ucharMat);

		oddImgR.copyTo(m_Decompose(cv::Rect(0, 0, c, r)));
	}
	m_Decompose.convertTo(outMatImg, CV_8UC1);
}

void WaveLetTest()
{
	WaveLetTransformer wlf("sym2", 3);
	cv::Mat image = cv::imread("F:/nbcode/1.jpg", 0);
	wlf.WaveletDT(image);
	cv::Mat outImg;
	wlf.IWaveletDT(outImg);
}