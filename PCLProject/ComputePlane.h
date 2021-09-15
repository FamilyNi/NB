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
template <typename T>
void PC_ThreePtsComputePlane(T& pt1, T& pt2, T& pt3, cv::Vec4d& plane);

/*��С���˷����ƽ�棺
	weights��[in]Ȩ��
	plane��[out]
*/
template <typename T>
void PC_OLSFitPlane(vector<T>& pts, vector<double>& weights, cv::Vec4d& plane);

/*huber����Ȩ�أ�
	plane��[in]
	weights��[out]Ȩ��
*/
template <typename T>
void PC_HuberPlaneWeights(vector<T>& pts, cv::Vec4d& plane, vector<double>& weights);

/*Turkey����Ȩ�أ�
	plane��[in]
	weights��[out]Ȩ��
*/
template <typename T>
void PC_TurkeyPlaneWeights(vector<T>& pts, cv::Vec4d& plane, vector<double>& weights);

/*ƽ����ϣ�
	plane��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
template <typename T>
void PC_FitPlane(vector<T>& pts, cv::Vec4d& plane, int k, NB_MODEL_FIT_METHOD method);


void PC_PlaneTest();