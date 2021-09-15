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
template <typename T>
void Img_ThreePtsComputeCicle(T& pt1, T& pt2, T& pt3, cv::Vec3d& circle);

/*���һ�²����㷨����Բ��
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
template <typename T>
void Img_RANSACComputeCircle(vector<T>& pts, cv::Vec3d& circle, vector<T>& inlinerPts, double thres);

/*��С���˷����԰��
	weights��[in]Ȩ��
	circle��[out]
*/
template <typename T>
void Img_OLSFitCircle(vector<T>& pts, vector<double>& weights, cv::Vec3d& circle);

/*huber����Ȩ�أ�
	circle��[in]
	weights��[out]Ȩ��
*/
template <typename T>
void Img_HuberCircleWeights(vector<T>& pts, cv::Vec3d& circle, vector<double>& weights);

/*Turkey����Ȩ�أ�
	circle��[in]
	weights��[out]Ȩ��
*/
template <typename T>
void Img_TurkeyCircleWeights(vector<T>& pts, cv::Vec3d& circle, vector<double>& weights);

/*���԰
	circle��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
template <typename T>
void Img_FitCircle(vector<T>& pts, cv::Vec3d& circle, int k, NB_MODEL_FIT_METHOD method);

void CircleTest();