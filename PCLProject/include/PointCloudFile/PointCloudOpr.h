#pragma once
#include "../BaseOprFile/utils.h"

//������Ƶ���С��Χ��
void PC_ComputeOBB(const PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& obb);

/*��ȡ���ƣ�
	indexes��[in]��������
	dstPC��[out]����ĵ���
*/
void PC_ExtractPC(const PC_XYZ::Ptr& srcPC, vector<int>& indexes, PC_XYZ::Ptr& dstPC);

/*����ֱ��ͶӰƽ����
	dstPC��[out]����ĵ���
	size��[in]��������ڵ�ĸ���
	thresVal��[in]��������ڵ�ĸ���
*/
void PC_LineProjSmooth(const PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, int size, double thresVal);

//������Ƶ�����
void PC_GetPCGravity(PC_XYZ::Ptr &srcPC, P_XYZ &gravity);

//������ͶӰ��XY��άƽ��
void PC_ProjectToXY(PC_XYZ::Ptr &srcPC, cv::Mat &xyPlane);

//������Ƶķ�����
void PC_ComputePCNormal(PC_XYZ::Ptr &srcPC, PC_N::Ptr &normals, float radius);


