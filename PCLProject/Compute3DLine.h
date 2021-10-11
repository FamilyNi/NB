#pragma once
#include "utils.h"

/*˵����
	ֱ�߷��̣� x = a * t + x0; y = b * t + y0; z = c * t + z0
	pts��[in]�ռ��еĵ��
	line��---[0]��a��[1]��b��[2]��c��[3]��x0��[4]��y0��[5]��z0
*/

/*��������*/
template <typename T1, typename T2>
void PC_TwoPtsComputeLine(T1& pt1, T1& pt2, T2& line);

/*���һ�²����㷨����ռ�ֱ�ߣ�
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
template <typename T1, typename T2>
void PC_RANSACComputeLine(vector<T1>& pts, T2& line, vector<T1>& inlinerPts, double thres);

/*��С���˷���Ͽռ�ֱ�ߣ�
	weights��[in]Ȩ��
	line��[out]
*/
template <typename T1, typename T2>
void PC_OLSFit3DLine(vector<T1>& pts, vector<double>& weights, T2& line);

/*Huber����Ȩ�أ�
	line��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void PC_Huber3DLineWeights(vector<T1>& pts, T2& line, vector<double>& weights);

/*Tukey����Ȩ�أ�
	line��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void PC_Tukey3DLineWeights(vector<T1>& pts, T2& line, vector<double>& weights);

/*�ռ�ֱ����ϣ�
	line��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
template <typename T1, typename T2>
void PC_Fit3DLine(vector<T1>& pts, T2& line, int k, NB_MODEL_FIT_METHOD method);


void PC_3DLineTest();