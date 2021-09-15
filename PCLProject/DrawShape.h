#pragma once
#include "utils.h"
#include "MathOpr.h"

/*����ƽ�棺
	linePC��[in]���ֱ��
	min_x��max_x��[in] x��������ֵ����Сֵ
	min_y��max_y��[in] y��������ֵ����Сֵ
	min_z��max_z��[in] z��������ֵ����Сֵ
	line��[in]ֱ�߲���
	step��[in]����
*/
void PC_DrawLine(PC_XYZ::Ptr& linePC, double min_x, double max_x, double min_y,
	double max_y, double min_z, double max_z, cv::Vec6d& line, double step);

/*����ƽ�棺
	planePC��[in]���ƽ��
	min_x��max_x��[in] x��������ֵ����Сֵ
	min_y��max_y��[in] y��������ֵ����Сֵ
	min_z��max_z��[in] z��������ֵ����Сֵ
	plane��[in]ƽ�����
	step��[in]����
*/
void PC_DrawPlane(PC_XYZ::Ptr& planePC, double min_x, double max_x, double min_y, 
	double max_y, double min_z, double max_z, cv::Vec4d& plane, double step);


/*������:
	spherePC��[out]�������
	center��[in]����
	raduis��[in]�뾶
	step��[in]�ǶȲ���
*/
void PC_DrawSphere(PC_XYZ::Ptr& spherePC, P_XYZ& center, double raduis, double step);

/*���������棺
	ellipsoidPC��[out]�����������
	center��[in]���������λ��
	a��b��c��[in]�ֱ�Ϊx��y��z����᳤
	step��[in]�ǶȲ���
*/
void PC_DrawEllipsoid(PC_XYZ::Ptr& ellipsoidPC, P_XYZ& center, double a, double b, double c, double step);

/*���������
	srcPC��[in]ԭʼ����
	noisePC��[out]��������
	range��[in]������С
	step��[in]���Ʋ���
*/
void PC_AddNoise(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& noisePC, int range, int step);