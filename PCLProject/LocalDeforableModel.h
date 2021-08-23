#pragma once
#include "utils.h"
#include "ShapeModelBase.h"

struct LocalDeforModelInfo
{
	vector<Point2f> coord;
	vector<Point2f> grad;
	Point2f gravity;
	vector<Point3f> normals_;
	vector<vector<uint>> segContIdx;
	vector<int> segContMapIdx;
	vector<float> dists;
	vector<float> label;
	LocalDeforModelInfo():gravity(0,0)
	{
		coord.clear();
		grad.clear();
		normals_.clear();
		segContIdx.clear();
		label.clear();
		dists.clear();
		segContMapIdx.clear();
	}
};

struct LocalDeforModel
{
	int pyrNum;
	float startAng;
	float endAng;
	float angStep;
	float minScore;
	float greediness;
	vector<LocalDeforModelInfo> models;
	LocalDeforModel() :pyrNum(0), startAng(0.0f), endAng(0.0f),
		angStep(0.0f), minScore(0.5f), greediness(0.9f)
	{
		models.resize(0);
	}
};

struct LocalMatchRes:public MatchRes
{
	vector<int> translates;
	LocalMatchRes()
	{
		translates.clear();
	}
	bool operator<(const LocalMatchRes &other) const
	{
		return score > other.score;
	}
};

//����ģ��
void CreateLocalDeforableModel(Mat &modImg, LocalDeforModel* &model, SPAPLEMODELINFO &shapeModelInfo);

//��ȡ����
void ExtractModelContour(Mat &srcImg, SPAPLEMODELINFO &shapeModelInfo, vector<vector<Point>> &contours);

//ģ������
void GetKNearestPoint(vector<Point2f> &contours, vector<Point2f> &grads, LocalDeforModelInfo &localDeforModelInfo);

//��ȡÿ��������������
void ComputeSegContGravity(LocalDeforModelInfo &localDeforModelInfo);

//�����������ķ�������
void ComputeSegContourVec(LocalDeforModel &model);

//�������Ļ�ȡÿ��С������ӳ������
void GetMapIndex(LocalDeforModel& localDeforModel);

//�Ծ�����ģ����ǩ
void LabelContour(LocalDeforModelInfo& localDeforModelInfo);

//����ÿ���������ķ�����
void ComputeContourNormal(const vector<Point2f>& contour, const vector<vector<uint>>& segContIdx, vector<Point2f>& normals);

//�ƶ�����
void TranslationContour(const vector<Point2f>& contour, const vector<uint>& contIdx,
	const Point3f& normals, vector<Point2f>& tranContour, int transLen);

//����ƥ��
void TopMatch(const Mat &s_x, const Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad,
	const vector<vector<uint>>& segIdx, const vector<Point3f>& contNormals, float minScore, float angle, LocalMatchRes& reses);

//ƥ��
void Match(const Mat &image, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<uint>>& segIdx, const vector<Point3f>& contNormals,
	cv::Point center, float minScore, float angle, vector<int>& transLen_down, LocalMatchRes& reses);

//��ȡƽ����
void GetTranslation(vector<int>& segContMapIdx, vector<int>& transLen_up, vector<int>& transLen_down);

//��ת��������
void RotContourVec(const vector<Point2f>& srcVec, vector<Point2f>& dstVec, float angle);

//�ϲ�ӳ�䵽�²�
void UpMapToDown(LocalDeforModelInfo& up_, LocalDeforModelInfo& down_, vector<int>& transLen_up, vector<vector<int>>& transLen_down);

//ƥ��
void LocalDeforModelMatch(Mat &modImg, LocalDeforModel* &model);

void LocalDeforModelTest();