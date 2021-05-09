#pragma once
#include "utils.h"

void WDT(cv::Mat &srcImg, cv::Mat& dstImg, const string& name, const int level);

void wavelet(const string& name, cv::Mat &lowFilter,  cv::Mat &highFilter);

void waveletDecompose(const cv::Mat &srcImg, const cv::Mat &lowFilter, const cv::Mat &highFilte, cv::Mat& dstImg);

void WaveLetTest();