#pragma once
#include "../BaseOprFile/utils.h"
#include <hash_map>

typedef struct PPFCELL
{
	uint ref_i;
	float ref_alpha;
	PPFCELL() :ref_i(0), ref_alpha(0.0f)
	{}
}PPFCELL;

typedef struct PPFFEATRUE
{
	float dist;
	float ang_N1D;
	double ang_N2D;
	float ang_N1N2;
	PPFFEATRUE() :dist(0.0f), ang_N1D(0.0f),
		ang_N2D(0.0f), ang_N1N2(0.0f)
	{}
}PPFFEATRUE;

typedef struct PPFMODEL
{
	vector<cv::Mat> refTransMat;
	uint numAng;
	float alphStep;
	float distStep;
	float angThres;
	float distThres;
	hash_map<string, vector<PPFCELL>> hashMap;
	PPFMODEL() :numAng(5.0f), distStep(0.1f),
		angThres(0.0f), distThres(0.0f)
	{
		alphStep = (float)CV_2PI / numAng;
		angThres = (float)CV_2PI / alphStep;
		refTransMat.resize(0);
	}
}PPFMODEL;

typedef struct PPFPose
{
	cv::Mat transMat;
	uint votes;
	uint ref_i;
	uint i_;
	PPFPose() :votes(0), ref_i(0), i_(0)
	{}
}PPFPose;

//����PPF����
void ComputePPFFEATRUE(P_XYZ& ref_p, P_XYZ& p_, P_N& ref_pn, P_N& p_n, PPFFEATRUE& ppfFEATRUE);

//�������PPF�������͵�hash����
void PushPPFToHashMap(hash_map<string, vector<PPFCELL>>& hashMap, PPFFEATRUE& ppfFEATRUE,
	float distStep, float stepAng, int ref_i, float alpha);

//��ȡPPF�ķ�����
void ExtractPPFNormals(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& downSamplepC, PC_N::Ptr& normals, float radius);

//����ֲ�ת������
void ComputeLocTransMat(P_XYZ& ref_p, P_N& ref_pn, cv::Mat& transMat);

//����ֲ�����ϵ�µ�alpha
float ComputeLocalAlpha(P_XYZ& ref_p, P_N& ref_pn, P_XYZ& p_, cv::Mat& refTransMat);

//����PPFģ��
void CreatePPFModel(PC_XYZ::Ptr& modelPC, PPFMODEL& ppfModel, float distRatio);

//����ͶƱ��
void ResetAccumulator(vector<vector<uint>>& accumulator);

//����任����
void ComputeTransMat(cv::Mat& SToGMat, float alpha, const cv::Mat& RToGMat, cv::Mat& transMat);

//����
bool ComparePose(PPFPose& a, PPFPose& b);

//����ת�������ת��
float ComputeRotMatAng(cv::Mat& transMat);

//�ж�����
bool DecisionCondition(PPFPose& a, PPFPose& b, float angThres, float distThres);

//�Ǽ���ֵ����
void NonMaxSuppression(vector<PPFPose>& ppfPoses, vector<PPFPose>& resPoses, float angThres, float distThres);

//����ģ��
void MatchPose(PC_XYZ::Ptr& srcPC, PPFMODEL& ppfModel, vector<PPFPose>& resPoses);

//����ת��
void TransPointCloud(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& dstPC, const cv::Mat& transMat);

//���Գ���
void TestProgram();