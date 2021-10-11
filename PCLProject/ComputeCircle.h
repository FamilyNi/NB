#pragma once
#include "utils.h"
#include "OpenCV_Utils.h"

/*˵����
	԰���̣�(x - a)^2 + (y - b)^2 = r^2
	��Ϸ��̣�x^2 + y^2 + A * x + B * y + C = 0
	pts��[in]ƽ���ϵĵ��
	circle��Բ---[0]��a��[1]��b��[2]��r
*/

/*������Բ��
	pt1��pt2��pt3��[in]ƽ�治���ߵ�������
	circle��[out]�����Բ---[0]��xԲ�ġ�[1]��yԲ�ġ�[2]���뾶
*/
template <typename T1, typename T2>
void Img_ThreePtsComputeCicle(T1& pt1, T1& pt2, T1& pt3, T2& circle);

/*���һ�²����㷨����Բ��
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
template <typename T1, typename T2>
void Img_RANSACComputeCircle(vector<T1>& pts, T2& circle, vector<T1>& inlinerPts, double thres);

/*��С���˷����԰��
	weights��[in]Ȩ��
	circle��[out]
*/
template <typename T1, typename T2>
void Img_OLSFitCircle(vector<T1>& pts, vector<double>& weights,T2& circle);

/*huber����Ȩ�أ�
	circle��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void Img_HuberCircleWeights(vector<T1>& pts,T2& circle, vector<double>& weights);

/*Turkey����Ȩ�أ�
	circle��[in]
	weights��[out]Ȩ��
*/
template <typename T1, typename T2>
void Img_TukeyCircleWeights(vector<T1>& pts, T2& circle, vector<double>& weights);

/*���԰
	circle��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
template <typename T1, typename T2>
void Img_FitCircle(vector<T1>& pts, T2& circle, int k, NB_MODEL_FIT_METHOD method);

void CircleTest();