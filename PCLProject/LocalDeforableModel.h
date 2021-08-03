#pragma once
#include "utils.h"
#include "ShapeModelBase.h"

struct LocalDeforModelInfo
{
	vector<Point2f> coord;
	vector<Point2f> grad;
	Point2f gravity;
	vector<vector<uint>> segContIdx;
	vector<float> label;
	LocalDeforModelInfo():gravity(0,0)
	{
		coord.clear();
		grad.clear();
		//gravity.clear();
		segContIdx.clear();
		label.clear();
	}
};


struct LocalDeforModel
{
	int pyrNumber;
	float startAng;
	float endAng;
	float angStep;
	float minScore;
	float greediness;
	vector<LocalDeforModelInfo> v_LocalDeforModel;
	LocalDeforModel() :pyrNumber(0), startAng(0.0f), endAng(0.0f),
		angStep(0.0f), minScore(0.5f), greediness(0.9f)
	{
		v_LocalDeforModel.resize(0);
	}
};

//����ģ��
void CreateLocalDeforableModel(Mat &modImg, LocalDeforModel* &model, SPAPLEMODELINFO &shapeModelInfo);

//��ȡ����
void ExtractModelContour(Mat &srcImg, SPAPLEMODELINFO &shapeModelInfo, vector<vector<Point>> &contours);

//��ȡģ���ݶ�
void ExtractModelInfo(Mat &srcImg, vector<Point> &contour, vector<Point2f> &v_Coord, vector<Point2f> &v_Grad, vector<float> &v_Amplitude);

//ģ������
void GetKNearestPoint(vector<Point2f> &contours, vector<Point2f> &grads, LocalDeforModelInfo &localDeforModelInfo);

//��ȡÿ��������������
void ComputeSegContGravity(LocalDeforModelInfo &localDeforModelInfo);

//�Ծ�����ģ����ǩ
void LabelContour(LocalDeforModelInfo& localDeforModelInfo);

//����ÿ���������ķ�����
void ComputeContourNormal(const vector<Point2f>& contour, const vector<vector<uint>>& segContIdx, vector<Point2f>& normals);

//�ƶ�����
void TranslationContour(const vector<Point2f>& contour, const vector<uint>& contIdx,
	const Point2f& normals, vector<Point2f>& tranContour, int transLen);

//����ƥ��
void TopMatch(const Mat &s_x, const Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<uint>>& segIdx,
	const vector<Point2f>& contNormals, float minScore, float angle, MatchRes& reses, vector<int>& v_TransLen);


//ƥ��
void LocalDeforModelMatch(Mat &modImg, LocalDeforModel* &model);

void LocalDeforModelTest();