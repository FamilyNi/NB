#pragma once
#include "utils.h"

/*˵����
	�򷽳̣�(x - a)^2 + (y - b)^2 + (z - c)^2 = r^2
	��Ϸ��̣�x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
	pts��[in]�ռ��еĵ��
	sphere��Բ---[0]��a��[1]��b��[2]��c��[3]��r
*/

/*�㵽԰�ľ���*/
template <typename T1, typename T2>
void PC_PtToShpereDist(T1& pt, T2& sphere, double& dist);

/*�ĵ������
	sphere��[out]
*/
template <typename T1, typename T2>
void PC_FourPtsComputeSphere(vector<T1>& pts, T2& sphere);

/*���һ�²����㷨������
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
template <typename T1, typename T2>
void PC_RANSACComputeSphere(vector<T1>& pts, T2& sphere, vector<T1>& inlinerPts, double thres);

/*��С���˷������
	weights��[in]Ȩ��
	sphere��[out]�����
*/
template <typename T1, typename T2>
void PC_OLSFitSphere(vector<T1>& pts, vector<double>& weights, T2& sphere);

/*Huber����Ȩ�أ�
	sphere��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void PC_HuberSphereWeights(vector<T1>& pts, T2& sphere, vector<double>& weights);

/*Tukey����Ȩ�أ�
	sphere��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void PC_TukeySphereWeights(vector<T1>& pts, T2& sphere, vector<double>& weights);

/*�����
	sphere��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
template <typename T1, typename T2>
void PC_FitSphere(vector<T1>& pts, T2& sphere, int k, NB_MODEL_FIT_METHOD method);

void PC_SphereTest();