#pragma once
#include "utils.h"

/*椭圆方程标准化*/
void Img_EllipseNormalization(cv::Vec6d& ellipse_, cv::Vec6d& normEllipse);

/*点到椭圆的距离--超简单版，不建议采用*/
template <typename T>
void Img_PtsToEllipseDist(T& pt, cv::Vec6d& ellipse, double& dist);

/*最小二乘法拟合椭圆*/
template <typename T>
void Img_OLSFitEllipse(vector<T>& pts, vector<double>& weights, cv::Vec6d& ellipse);

/*huber计算权重：
	sphere：[in]
	weights：[out]权重
*/
template <typename T>
void Img_HuberEllipseWeights(vector<T>& pts, cv::Vec6d& ellipse, vector<double>& weights);

/*Turkey计算权重：
	sphere：[in]
	weights：[out]权重
*/
template <typename T>
void Img_TurkeyEllipseWeights(vector<T>& pts, cv::Vec6d& ellipse, vector<double>& weights);

/*拟合椭圆*/
template <typename T>
void Img_FitEllipse(vector<T>& pts, cv::Vec6d& ellipse, int k, NB_MODEL_FIT_METHOD method);


void Img_EllipseTest();
