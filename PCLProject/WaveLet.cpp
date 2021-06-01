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

	if (!m_LowFilter_T.empty())
		m_LowFilter_T.release();
	if (!m_HighFilter_T.empty())
		m_HighFilter_T.release();
	m_LowFilter_T = m_LowFilter.t();
	m_HighFilter_T = m_HighFilter.t();
}

void WaveLetTransformer::Set_I_Filter(const string& name)
{
	m_Name = name;
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

	if (!m_Low_I_Filter_T.empty())
		m_Low_I_Filter_T.release();
	if (!m_High_I_Filter_T.empty())
		m_High_I_Filter_T.release();
	m_Low_I_Filter_T = m_Low_I_Filter.t();
	m_High_I_Filter_T = m_High_I_Filter.t();
}

//行分解
void WaveLetTransformer::R_Decompose(const cv::Mat& src, cv::Mat& dst_R_L, cv::Mat& dst_R_H)
{
	cv::Mat dstLowR_, dstHighR_;
	cv::filter2D(src, dstLowR_, -1, m_LowFilter); //低通滤波---可加速，但需要自己写，麻烦
	cv::filter2D(src, dstHighR_, -1, m_HighFilter); //高通滤波---可加速，但需要自己写，麻烦
	GetOddC(dstLowR_, dst_R_L);
	GetOddC(dstHighR_, dst_R_H);
}

//列分解
void WaveLetTransformer::C_Decompose(cv::Mat& dst_R_L, cv::Mat& dst_R_H,
	cv::Mat& CMat1, cv::Mat& CMat2, cv::Mat& CMat3, cv::Mat& CMat4)
{
	cv::Mat CMat1_, CMat2_, CMat3_, CMat4_;
	//行低频部分
	cv::filter2D(dst_R_L, CMat1_, -1, m_LowFilter_T); //低通滤波
	cv::filter2D(dst_R_L, CMat2_, -1, m_HighFilter_T); //高通滤波
	//行高频部分
	cv::filter2D(dst_R_H, CMat3_, -1, m_LowFilter_T); //低通滤波
	cv::filter2D(dst_R_H, CMat4_, -1, m_HighFilter_T); //高通滤波

	GetOddR(CMat1_, CMat1);
	GetOddR(CMat2_, CMat2);
	GetOddR(CMat3_, CMat3);
	GetOddR(CMat4_, CMat4);
}

//小波行重建
void WaveLetTransformer::R_Recontrcution(cv::Mat& src_C_L,
	cv::Mat& src_C_H, cv::Mat& dst_R, int r, int c)
{
	cv::Mat RMat1_(r, c, src_C_L.type(), cv::Scalar(0.0f));
	cv::Mat RMat2_(r, c, src_C_L.type(), cv::Scalar(0.0f));

	InterC(src_C_L, RMat1_);
	InterC(src_C_H, RMat2_);
	cv::Mat RMat1_L, RMat1_H;
	cv::filter2D(RMat1_, RMat1_L, -1, m_Low_I_Filter); //低通滤波
	cv::filter2D(RMat2_, RMat1_H, -1, m_High_I_Filter); //高通滤波
	dst_R = (RMat1_L + RMat1_H);
}

//小波列重建
void WaveLetTransformer::C_Reconstruction(cv::Mat& src, cv::Mat& dst_C_L,
	cv::Mat& dst_C_H, int r, int c)
{
	int r_ = r / 2;
	cv::Mat CMat1, CMat2, CMat3, CMat4;
	CMat1 = m_Decompose(cv::Rect(0, 0, c, r_));
	CMat2 = m_Decompose(cv::Rect(c, 0, c, r_));
	CMat3 = m_Decompose(cv::Rect(0, r_, c, r_));
	CMat4 = m_Decompose(cv::Rect(c, r_, c, r_));

	cv::Mat CMat1_(r, c, CMat1.type(), cv::Scalar(0.0f));
	cv::Mat CMat2_(r, c, CMat2.type(), cv::Scalar(0.0f));
	cv::Mat CMat3_(r, c, CMat3.type(), cv::Scalar(0.0f));
	cv::Mat CMat4_(r, c, CMat4.type(), cv::Scalar(0.0f));

	InterR(CMat1, CMat1_); InterR(CMat2, CMat2_);
	InterR(CMat3, CMat3_); InterR(CMat4, CMat4_);

	//行低频部分
	cv::Mat CMat1_L, CMat1_H;
	cv::filter2D(CMat1_, CMat1_L, -1, m_Low_I_Filter_T); //低通滤波
	cv::filter2D(CMat2_, CMat1_H, -1, m_High_I_Filter_T); //高通滤波
	dst_C_L = CMat1_L + CMat1_H;
	//行高频部分
	cv::Mat CMat2_L, CMat2_H;
	cv::filter2D(CMat3_, CMat2_L, -1, m_Low_I_Filter_T); //低通滤波
	cv::filter2D(CMat4_, CMat2_H, -1, m_High_I_Filter_T); //高通滤波
	dst_C_H = CMat2_L + CMat2_H;
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
	for (int i = 0; i < m_Level; ++i)
	{
		//行滤波		
		cv::Mat dst_R_L, dst_R_H;
		R_Decompose(src, dst_R_L, dst_R_H);

		//列滤波
		cv::Mat CMat1, CMat2, CMat3, CMat4;
		C_Decompose(dst_R_L, dst_R_H, CMat1, CMat2, CMat3, CMat4);

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
	int r = m_Decompose.rows / pow(2, m_Level);
	int c = m_Decompose.cols / pow(2, m_Level);
	for (int i = 0; i < m_Level; ++i)
	{
		r *= 2;
		cv::Mat dst_C_L, dst_C_H;
		C_Reconstruction(m_Decompose, dst_C_L, dst_C_H, r, c);

		//行重建
		c *= 2;
		cv::Mat dst_R;
		R_Recontrcution(dst_C_L, dst_C_H, dst_R, r, c);
		dst_R.copyTo(m_Decompose(cv::Rect(0, 0, c, r)));
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