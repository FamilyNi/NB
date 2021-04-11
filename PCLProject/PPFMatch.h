#pragma once
#include "utils.h"
#include <hash_map>

typedef struct PPFFEATRUE
{
	float dist;
	float ang_N1D;
	float ang_N2D;
	float ang_N1N2;
	PPFFEATRUE() :dist(0.0f), ang_N1D(0.0f),
		ang_N2D(0.0f), ang_N1N2(0.0f)
	{}
}PPFFEATRUE;

typedef struct PPFMODEL
{
	PC_XYZ::Ptr modelPC;
	float alphStep;
	float distStep;
	hash_map<string, vector<vector<uint>>> hashMap;
	PPFMODEL() :alphStep(5.0f), distStep(0.1f), modelPC(new PC_XYZ)
	{}
}PPFMODEL;

//�޸����˹��ʽ
void RodriguesFormula(P_N& normal, P_N& rotAxis, float rotAng);

//����PPF����
void ComputePPFFEATRUE(P_XYZ& ref_p, P_XYZ& p_, P_N& ref_pn, P_N& p_n, PPFFEATRUE& ppfFEATRUE);

//�������PPF�������͵�hash����
void PushPPFToHashMap(hash_map<string, vector<vector<uint>>>& hashMap, PPFFEATRUE& ppfFEATRUE, int ref_i, int i_);

//��ȡPPF�ķ�����
void ExtractPPFNormals(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& downSamplepC, PC_N::Ptr& normals, float radius);

//��ת������
void ComputeTransMat(P_XYZ& ref_p, P_N& ref_pn, cv::Mat& transMat);

//����ֲ�����ϵ�µ�alpha
float ComputeLocalAlpha(P_XYZ& ref_p, P_N& ref_pn, P_XYZ& p_);

//����PPFģ��
void CreatePPFModel(PC_XYZ::Ptr& modelPC, PPFMODEL& ppfModel, float distRatio);