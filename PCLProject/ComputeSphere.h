#pragma once
#include "utils.h"

/*˵����
	�򷽳̣�(x - a)^2 + (y - b)^2 + (z - c)^2 = r^2
	��Ϸ��̣�x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
	pts��[in]�ռ��еĵ��
	sphere��Բ---[0]��a��[1]��b��[2]��c��[3]��r
*/

/*�ĵ������
	sphere��[out]
*/
template <typename T>
void PC_FourPtsComputeSphere(vector<T>& pts, cv::Vec4d& sphere);

/*��С���˷������
	weights��[in]Ȩ��
	sphere��[out]
*/
template <typename T>
void PC_OLSFitSphere(vector<T>& pts, vector<double>& weights, cv::Vec4d& sphere);

/*huber����Ȩ�أ�
	sphere��[in]
	weights��[out]Ȩ��
*/
template <typename T>
void PC_HuberSphereWeights(vector<T>& pts, cv::Vec4d& sphere, vector<double>& weights);

/*Turkey����Ȩ�أ�
	sphere��[in]
	weights��[out]Ȩ��
*/
template <typename T>
void PC_TurkeySphereWeights(vector<T>& pts, cv::Vec4d& sphere, vector<double>& weights);

/*�����
	sphere��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
template <typename T>
void PC_FitSphere(vector<T>& pts, cv::Vec4d& sphere, int k, NB_MODEL_FIT_METHOD method);


void PC_SphereTest();