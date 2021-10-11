#pragma once
#include "utils.h"

/*˵����
	��Ϸ��̣�a * x^2 + b * x * y + c * y * y + d * x + e * y + f = 0
	pts��[in]ƽ���ϵĵ��
	ellipse��Բ---[0]��a��[1]��b��[2]��c��[3]��d��[4]��e��[5]��f
*/

/*��Բ���̱�׼��*/
template <typename T>
void Img_EllipseNormalization(T& ellipse_, T& normEllipse);

/*�㵽��Բ�ľ���--���򵥰棬���������*/
template <typename T1, typename T2>
void Img_PtsToEllipseDist(T1& pt, T2& ellipse, double& dist);

/*��������Բ*/
template <typename T1, typename T2>
void Img_SixPtsComputeEllipse(vector<T1>& pts, T2& ellipse);

/*���һ�²����㷨������ԲԲ��
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
template <typename T1, typename T2>
void Img_RANSACComputeCircle(vector<T1>& pts, T2& ellipse, vector<T1>& inlinerPts, double thres);

/*��С���˷������Բ��
	weights��[in]Ȩ��
	ellipse��[out]
*/
template <typename T1, typename T2>
void Img_OLSFitEllipse(vector<T1>& pts, vector<double>& weights, T2& ellipse);

/*huber����Ȩ�أ�
	ellipse��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void Img_HuberEllipseWeights(vector<T1>& pts, T2& ellipse, vector<double>& weights);

/*Turkey����Ȩ�أ�
	ellipse��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void Img_TukeyEllipseWeights(vector<T1>& pts, T2& ellipse, vector<double>& weights);

/*�����Բ��
	ellipse��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
template <typename T1, typename T2>
void Img_FitEllipse(vector<T1>& pts, T2& ellipse, int k, NB_MODEL_FIT_METHOD method);


void Img_EllipseTest();
