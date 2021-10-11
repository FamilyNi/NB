#pragma once
#include "utils.h"
#include "OpenCV_Utils.h"

/*˵����
	ֱ�߷��̣� a * x + b * y + c = 0
	pts��[in]ƽ���ϵĵ��
	line��---[0]��a��[1]��b��[2]��c
*/

/*�������ֱ�ߣ�
	pt1��pt2��[in]ƽ���ϵ�������
	line��[out]�����ֱ��---[0]��x�����Ϸ�������
				[1]��y����ķ�������[2]��cֵ
*/
template <typename T>
void Img_TwoPtsComputeLine(T& pt1, T& pt2, cv::Vec3d& line);

/*���һ�²����㷨����ֱ��
	pts��[in]ƽ���ϵĵ��
	line��[out]�����Բ---[0]��x�����Ϸ�������
				[1]��y����ķ�������[2]��cֵ
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
template <typename T1, typename T2>
void Img_RANSACComputeLine(vector<T1>& pts, T2& line, vector<T1>& inlinerPts, double thres);

/*��С���˷����ֱ�ߣ�
	weights��[in]Ȩ��
	line��[out]
*/
template <typename T1, typename T2>
void Img_OLSFitLine(vector<T1>& pts, vector<double>& weights,T2& line);

/*Huber����Ȩ�أ�
	line��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void Img_HuberLineWeights(vector<T1>& pts,T2& line, vector<double>& weights);

/*Tukey����Ȩ�أ�
	line��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void Img_TukeyLineWeights(vector<T1>& pts, T2& line, vector<double>& weights);

/*ֱ����ϣ�
	line��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
template <typename T1, typename T2>
void Img_FitLine(vector<T1>& pts,T2& line, int k, NB_MODEL_FIT_METHOD method);

void LineTest();
