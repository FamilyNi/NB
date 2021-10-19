#pragma once
#include "utils.h"

/*˵����
	ƽ�淽�̣�a * x + b * y + c * z + d = 0
	pts��[in]�ռ��еĵ��
	plane��ƽ��---[0]��a��[1]��b��[2]��c��[3]��d
*/

/*�������ƽ�棺
	pt1��pt2��pt3��[in]�ռ��е�������
	plane��[out]
*/
template <typename T1, typename T2>
void PC_ThreePtsComputePlane(T1& pt1, T1& pt2, T1& pt3, T2& plane);

/*���һ�²����㷨����ƽ�棺
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
template <typename T1, typename T2>
void PC_RANSACComputePlane(vector<T1>& pts, T2& plane, vector<T1>& inlinerPts, double thres);

/*��С���˷����ƽ�棺
	weights��[in]Ȩ��
	plane��[out]
*/
template <typename T1, typename T2>
void PC_OLSFitPlane(vector<T1>& pts, vector<double>& weights, T2& plane);

/*huber����Ȩ�أ�
	plane��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void PC_HuberPlaneWeights(vector<T1>& pts, T2& plane, vector<double>& weights);

/*Tukey����Ȩ�أ�
	plane��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void PC_TukeyPlaneWeights(vector<T1>& pts, T2& plane, vector<double>& weights);

/*ƽ����ϣ�
	plane��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
template <typename T1, typename T2>
void PC_FitPlane(vector<T1>& pts,T2& plane, int k, NB_MODEL_FIT_METHOD method);

void PC_PlaneTest();