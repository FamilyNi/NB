#pragma once
#include "utils.h"

/*˵����
	�ռ�԰������һ�㷽�̣�x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
				���߷���(x - a)*vx + (y - b)*vy + (z - c)*vz = 0
				vx��vy��vzΪԲ����ƽ��ķ�����
	pts��[in]�ռ��еĵ��
	circle��---[0]��Բ��x��[1]��Բ��y��[2]��Բ��z��[3]���뾶��[4]��vx��[5]��vy��[6]��vz
*/

/*�㵽Բ�ľ���*/
template <typename T1, typename T2>
void PC_PtToCircleDist(T1& pt, T2& circle, double& dist);

/*�������԰��
	pt1��pt2��pt3��[in]�ռ��е�������
	circle��[out]
*/
template <typename T1, typename T2>
void PC_ThreePtsComputeCircle(T1& pt1, T1& pt2, T1& pt3, T2& circle);

/*���һ�²����㷨����Բ��
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
template <typename T1, typename T2>
void PC_RANSACComputeCircle(vector<T1>& pts, T2& circle, vector<T1>& inlinerPts, double thres);

/*��С���˷���Ͽռ�ռ�԰��
	weights��[in]Ȩ��
	circle��[out]
*/
template <typename T1, typename T2>
void PC_OLSFit3DCircle(vector<T1>& pts, vector<double>& weights, T2& circle);

/*��С���˷����Բ��
	circle��[out]���Բ
	weights��[in]Ȩ��
*/
template <typename T1, typename T2>
void PC_HuberCircleWeights(vector<T1>& pts, T2& circle, vector<double>& weights);

/*Tukey����Ȩ�أ�
	circle��[out]���Բ
	weights��[in]Ȩ��
*/
template <typename T1, typename T2>
void PC_TukeyCircleWeights(vector<T1>& pts, T2& circle, vector<double>& weights);

void PC_CircleTest();
