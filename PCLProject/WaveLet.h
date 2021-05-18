#pragma once
#include "utils.h"

class WaveLetTransformer 
{
public:
	WaveLetTransformer(const string &name, uint level);

	void SetLevel(uint level) { m_Level = level; }

	uint GetLevel() { return m_Level; }

	cv::Mat GetWaveLetL() { return m_LowFilter; }

	cv::Mat GetWaveLetH() { return m_HighFilter; }

	//设置小波分解滤波器
	void SetFilter(const string& name);

	//设置小波重构滤波器
	void Set_I_Filter(const string& name);

	//行分解
	void R_Decompose(const cv::Mat& src, cv::Mat& dst_R_L, cv::Mat& dst_R_H);

	//列分解
	void C_Decompose(cv::Mat& dst_R_L, cv::Mat& dst_R_H, cv::Mat& CMat1,
		cv::Mat& CMat2, cv::Mat& CMat3, cv::Mat& CMat4);

	//小波行重建
	void R_Recontrcution(cv::Mat& src_C_L, cv::Mat& src_C_H, cv::Mat& dst_R, int r, int c);

	//小波列重建
	void C_Reconstruction(cv::Mat& src, cv::Mat& dst_C_L, cv::Mat& dst_C_H, int r, int c);

	//获取偶数行
	void GetOddR(const cv::Mat& srcImg, cv::Mat& oddRImg);

	//获取偶数列
	void GetOddC(const cv::Mat& srcImg, cv::Mat& oddCImg);

	//列方向插值
	void InterC(const cv::Mat& srcImg, cv::Mat& oddCImg);

	//行方向插值
	void InterR(const cv::Mat& srcImg, cv::Mat& oddCImg);

	//小波分解
	void WaveletDT(const cv::Mat& srcImg);

	//小波重建
	void IWaveletDT(cv::Mat& outMatImg);

private:
	cv::Mat m_LowFilter;
	cv::Mat m_HighFilter;
	cv::Mat m_LowFilter_T;
	cv::Mat m_HighFilter_T;

	cv::Mat m_Low_I_Filter;
	cv::Mat m_High_I_Filter;
	cv::Mat m_Low_I_Filter_T;
	cv::Mat m_High_I_Filter_T;

	cv::Mat m_Decompose;
	uint m_Level;
	string m_Name;
};

void WaveLetTest();