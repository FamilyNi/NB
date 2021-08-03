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

//创建模板
void CreateLocalDeforableModel(Mat &modImg, LocalDeforModel* &model, SPAPLEMODELINFO &shapeModelInfo);

//提取轮廓
void ExtractModelContour(Mat &srcImg, SPAPLEMODELINFO &shapeModelInfo, vector<vector<Point>> &contours);

//提取模板梯度
void ExtractModelInfo(Mat &srcImg, vector<Point> &contour, vector<Point2f> &v_Coord, vector<Point2f> &v_Grad, vector<float> &v_Amplitude);

//模板点聚类
void GetKNearestPoint(vector<Point2f> &contours, vector<Point2f> &grads, LocalDeforModelInfo &localDeforModelInfo);

//求取每个子轮廓的重心
void ComputeSegContGravity(LocalDeforModelInfo &localDeforModelInfo);

//对聚类后的模板打标签
void LabelContour(LocalDeforModelInfo& localDeforModelInfo);

//计算每个子轮廓的法向量
void ComputeContourNormal(const vector<Point2f>& contour, const vector<vector<uint>>& segContIdx, vector<Point2f>& normals);

//移动轮廓
void TranslationContour(const vector<Point2f>& contour, const vector<uint>& contIdx,
	const Point2f& normals, vector<Point2f>& tranContour, int transLen);

//顶层匹配
void TopMatch(const Mat &s_x, const Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<uint>>& segIdx,
	const vector<Point2f>& contNormals, float minScore, float angle, MatchRes& reses, vector<int>& v_TransLen);


//匹配
void LocalDeforModelMatch(Mat &modImg, LocalDeforModel* &model);

void LocalDeforModelTest();