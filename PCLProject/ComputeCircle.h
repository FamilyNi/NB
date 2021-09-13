#pragma once
#include "utils.h"
#include "OpenCV_Utils.h"

/*������Բ��
	pt1��pt2��pt3��[in]ƽ�治���ߵ�������
	circle��[out]�����Բ---[0]��xԲ�ġ�[1]��yԲ�ġ�[2]���뾶
*/
template <typename T>
void Img_ThreePointComputeCicle(T& pt1, T& pt2, T& pt3, cv::Vec3d& circle);

/*���һ�²����㷨����Բ��
	pts��[in]ƽ���ϵĵ��
	circle��[out]�����Բ---[0]��xԲ�ġ�[1]��yԲ�ġ�[2]���뾶
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
template <typename T>
void Img_RANSACComputeCircle(vector<T>& pts, cv::Vec3d& circle, vector<T>& inlinerPts, double thres);

void CircleTest();