#pragma once
#include "utils.h"

/*�㵽ƽ��ľ��룺
	pt��[in]�����
	plane��[in]ƽ�淽��
	dist��[out]�������
*/
template <typename T1, typename T2>
void PC_PtToPlaneDist(T1& pt, T2& plane, double& dist);

/*������һ��*/
template <typename T>
void PC_VecNormal(T& p);

/*�㵽ƽ���ͶӰ��
	pt��[in]�����
	plane��[in]ƽ�淽��
	projPt��[out]ͶӰ��
*/
template <typename T1, typename T2, typename T3>
void PC_PtProjPlanePt(T1& pt, T2& plane, T3& projPt);

/*�ռ�㵽�ռ�ֱ�ߵľ���
	pt��[in]�����
	line��[in]ֱ�߷���
	dist��[out]�������
*/
template <typename T1, typename T2>
void PC_PtToLineDist(T1& pt, T2& line, double& dist);

/*�ռ�㵽�ռ�ֱ�ߵ�ͶӰ
	pt��[in]�����
	line��[in]ֱ�߷���
	projPt��[out]ͶӰ��
*/
template <typename T1, typename T2, typename T3>
void PC_PtProjLinePt(T1& pt, T2& line, T3& projPt);

/*��ά�������
	vec1��vec2��[in]��ʾ����1��2
	vec��[out]��˺�Ľ��
*/
template <typename T1, typename T2, typename T3>
void PC_VecCross(T1& vec1, T2& vec2, T3& vec, bool isNormal);

/*��������֮��ľ���--��ά*/
template <typename T1, typename T2>
void Img_ComputePPDist(T1& pt1, T2& pt2, double& dist);

/*ƽ���ϵ㵽ֱ�ߵ�ͶӰ
	pt��[in]�����
	line��[in]ֱ�߷���
	projPt��[out]ͶӰ��
*/
template <typename T1, typename T2, typename T3>
void Img_PtProjLinePt(T1& pt, T2& line, T3& projPt);

/*�޸����˹��ʽ��
	rotAxis��[in]��ת��
	rotAng��[in]������֮��ļн�
	rotMat��[out]��ת����
*/
template <typename T>
void RodriguesFormula(T& rotAxis, double rotAng, cv::Mat& rotMat);

