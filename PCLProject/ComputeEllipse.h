#pragma once
#include "utils.h"

/*��Բ���̱�׼��*/
void Img_EllipseNormalization(cv::Vec6d& ellipse_, cv::Vec6d& normEllipse);

/*�㵽��Բ�ľ���--���򵥰棬���������*/
template <typename T>
void Img_PtsToEllipseDist(T& pt, cv::Vec6d& ellipse, double& dist);

/*��С���˷������Բ*/
template <typename T>
void Img_OLSFitEllipse(vector<T>& pts, vector<double>& weights, cv::Vec6d& ellipse);

/*huber����Ȩ�أ�
	sphere��[in]
	weights��[out]Ȩ��
*/
template <typename T>
void Img_HuberEllipseWeights(vector<T>& pts, cv::Vec6d& ellipse, vector<double>& weights);

/*Turkey����Ȩ�أ�
	sphere��[in]
	weights��[out]Ȩ��
*/
template <typename T>
void Img_TurkeyEllipseWeights(vector<T>& pts, cv::Vec6d& ellipse, vector<double>& weights);

/*�����Բ*/
template <typename T>
void Img_FitEllipse(vector<T>& pts, cv::Vec6d& ellipse, int k, NB_MODEL_FIT_METHOD method);


void Img_EllipseTest();
