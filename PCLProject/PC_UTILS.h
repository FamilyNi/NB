#pragma once
#include "utils.h"

//������Ƶ�����
void ComputePCGravity(PC_XYZ::Ptr &srcPC, P_XYZ &gravity);

//������ͶӰ��XYƽ��
void PC_ProjectToXY(PC_XYZ::Ptr &srcPC, cv::Mat &xyPlane);

//������Ƶķ�����
void ComputePCNormal(PC_XYZ::Ptr &srcPC, PC_N::Ptr &normals, float radius);

//����Э�������
void ComputeCovMat(PC_XYZ::Ptr &pc, cv::Mat &covMat, P_XYZ &gravity);

//������Ƶ����ͼ
void ComputePCRangeImg();