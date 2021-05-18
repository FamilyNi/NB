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

	//����С���ֽ��˲���
	void SetFilter(const string& name);

	//����С���ع��˲���
	void Set_I_Filter(const string& name);

	//��ȡż����
	void GetOddR(const cv::Mat& srcImg, cv::Mat& oddRImg);

	//��ȡż����
	void GetOddC(const cv::Mat& srcImg, cv::Mat& oddCImg);

	//�з����ֵ
	void InterC(const cv::Mat& srcImg, cv::Mat& oddCImg);

	//�з����ֵ
	void InterR(const cv::Mat& srcImg, cv::Mat& oddCImg);

	//С���ֽ�
	void WaveletDT(const cv::Mat& srcImg);

	//С���ؽ�
	void IWaveletDT(cv::Mat& outMatImg);

private:
	cv::Mat m_LowFilter;
	cv::Mat m_HighFilter;
	cv::Mat m_Low_I_Filter;
	cv::Mat m_High_I_Filter;
	cv::Mat m_Decompose;
	uint m_Level;
	string m_Name;
};

void WaveLetTest();