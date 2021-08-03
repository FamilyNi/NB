#pragma once
#include "utils.h"

//������Ƶ���С��Χ��
void PC_ComputeOBB(const PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& obb);

//��ȡ����
void PC_ExtractPC(const PC_XYZ::Ptr& srcPC, vector<int>& indexes, PC_XYZ::Ptr& dstPC);

//����ֱ��ͶӰƽ��
void PC_LineProjSmooth(const PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, int size, float thresVal);

//������Ƶ�����
void PC_GetPCGravity(PC_XYZ::Ptr &srcPC, P_XYZ &gravity);

//�ط��߷������ŵ���
void PC_ScalePCBaseNormal(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &normal_p, PC_XYZ::Ptr &dstPC, float scale);

//������ͶӰ��XY��άƽ��
void PC_ProjectToXY(PC_XYZ::Ptr &srcPC, cv::Mat &xyPlane);

//������Ƶķ�����
void PC_ComputePCNormal(PC_XYZ::Ptr &srcPC, PC_N::Ptr &normals, float radius);

//�������
void PC_MulCrossCalNormal(P_XYZ &p_1, P_XYZ &p_2, P_XYZ &normal_p);

//������Ƶ�Э�������
void PC_ComputeCovMat(PC_XYZ::Ptr &pc, Mat& covMat, P_XYZ& gravity);
