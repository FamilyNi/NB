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

//计算PPF特征
void ComputePPFFEATRUE(P_XYZ& ref_p, P_XYZ& p_, P_N& ref_pn, P_N& p_n, PPFFEATRUE& ppfFEATRUE);

//将点对以PPF特征推送到hash表中
void PushPPFToHashMap(hash_map<string, vector<PPFCELL>>& hashMap, PPFFEATRUE& ppfFEATRUE,
	float distStep, float stepAng, int ref_i, float alpha);

//提取PPF的法向量
void ExtractPPFNormals(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& downSamplepC, PC_N::Ptr& normals, float radius);

//计算局部转换矩阵
void ComputeLocTransMat(P_XYZ& ref_p, P_N& ref_pn, cv::Mat& transMat);

//计算局部坐标系下的alpha
float ComputeLocalAlpha(P_XYZ& ref_p, P_N& ref_pn, P_XYZ& p_, cv::Mat& refTransMat);

//创建PPF模板
void CreatePPFModel(PC_XYZ::Ptr& modelPC, PPFMODEL& ppfModel, float distRatio);

//重置投票器
void ResetAccumulator(vector<vector<uint>>& accumulator);

//计算变换矩阵
void ComputeTransMat(cv::Mat& SToGMat, float alpha, const cv::Mat& RToGMat, cv::Mat& transMat);

//排序
bool ComparePose(PPFPose& a, PPFPose& b);

//求旋转矩阵的旋转角
float ComputeRotMatAng(cv::Mat& transMat);

//判定条件
bool DecisionCondition(PPFPose& a, PPFPose& b, float angThres, float distThres);

//非极大值抑制
void NonMaxSuppression(vector<PPFPose>& ppfPoses, vector<PPFPose>& resPoses, float angThres, float distThres);

//查找模板
void MatchPose(PC_XYZ::Ptr& srcPC, PPFMODEL& ppfModel, vector<PPFPose>& resPoses);

//点云转换
void TransPointCloud(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& dstPC, const cv::Mat& transMat);

//测试程序
void TestProgram();