#pragma once
#include "utils.h"

/*˵����
	�ռ�԰������һ�㷽�̣�x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
				���߷���(x - a)*vx + (y - b)*vy + (z - c)*vz = 0
	pts��[in]�ռ��еĵ��
	circle��---[0]��a��[1]��b��[2]��c��[3]��x0��[4]��y0��[5]��z0
*/

/*�㵽Բ�ľ���*/
template <typename T1, typename T2>
void PC_PtToCircleDist(T1& pt, T2& circle, double& dist);

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
