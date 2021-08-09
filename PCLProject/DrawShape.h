#pragma once
#include "utils.h"

/*������:
	sphere��[out]�������
	center��[in]����
	raduis��[in]�뾶
	step��[in]�ǶȲ���
*/
void DrawSphere(PC_XYZ::Ptr& sphere, P_XYZ& center, double raduis, double step);

/*���������棺
	ellipsoid��[out]�����������
	center��[in]���������λ��
	a��b��c��[in]�ֱ�Ϊx��y��z����᳤
	step��[in]�ǶȲ���
*/
void DrawEllipsoid(PC_XYZ::Ptr& ellipsoid, P_XYZ& center, double a, double b, double c, double step);