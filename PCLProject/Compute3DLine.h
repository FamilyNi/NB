#pragma once
#include "utils.h"

/*˵����
	ֱ�߷��̣� x = a * t + x0; y = b * t + y0; z = c * t + z0
	pts��[in]�ռ��еĵ��
	line��---[0]��a��[1]��b��[2]��c��[3]��x0��[4]��y0��[5]��z0
*/

/*�ռ�������ֱ�ߣ�
	pt1��pt2��[in]�ռ��е���������
	line��[out]
*/
template <typename T>
void PC_OLSFit3DLine(T& pt1, T& pt2, cv::Vec6d& line);

/*��С���˷���Ͽռ�ֱ�ߣ�
	weights��[in]Ȩ��
	line��[out]
*/
template <typename T>
void PC_OLSFit3DLine(vector<T>& pts, vector<double>& weights, cv::Vec6d& line);

/*Huber����Ȩ�أ�
	line��[in]
	weights��[out]Ȩ��
*/
template <typename T>
void PC_Huber3DLineWeights(vector<T>& pts, cv::Vec6d& line, vector<double>& weights);

/*Turkey����Ȩ�أ�
	line��[in]
	weights��[out]Ȩ��
*/
template <typename T>
void PC_Turkey3DLineWeights(vector<T>& pts, cv::Vec6d& line, vector<double>& weights);

/*�ռ�ֱ����ϣ�
	line��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
template <typename T>
void PC_Fit3DLine(vector<T>& pts, cv::Vec6d& line, int k, NB_MODEL_FIT_METHOD method);


void PC_3DLineTest();