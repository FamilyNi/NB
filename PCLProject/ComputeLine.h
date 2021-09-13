#pragma once
#include "utils.h"
#include "OpenCV_Utils.h"

/*�������ֱ�ߣ�
	pt1��pt2��[in]ƽ���ϵ�������
	line��[out]�����ֱ��---[0]��x�����Ϸ�������
				[1]��y����ķ�������[2]��cֵ
*/
template <typename T>
void Img_TwoPtsComputeLine(T& pt1, T& pt2, cv::Vec3d& line);

/*���һ�²����㷨����ֱ��
	pts��[in]ƽ���ϵĵ��
	line��[out]�����Բ---[0]��x�����Ϸ�������
				[1]��y����ķ�������[2]��cֵ
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
template <typename T>
void Img_RANSACComputeLine(vector<T>& pts, cv::Vec3d& line, vector<T>& inlinerPts, double thres);

/*Turkeyֱ�����*/
void Img_TurkeyFitLine(vector<cv::Point>& pts, cv::Vec3d& line, int k, double thres);

void LineTest();
