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

//创建模板
void CreateLocalDeforableModel(Mat &modImg, LocalDeforModel* &model, SPAPLEMODELINFO &shapeModelInfo);

//提取轮廓
void ExtractModelContour(Mat &srcImg, SPAPLEMODELINFO &shapeModelInfo, vector<vector<Point>> &contours);

//模板点聚类
void GetKNearestPoint(vector<Point2f> &contours, vector<Point2f> &grads, LocalDeforModelInfo &localDeforModelInfo);

//求取每个子轮廓的重心
void ComputeSegContGravity(LocalDeforModelInfo &localDeforModelInfo);

//计算子轮廓的方向向量
void ComputeSegContourVec(LocalDeforModel &model);

//根据中心获取每个小轮廓的映射索引
void GetMapIndex(LocalDeforModel& localDeforModel);

//对聚类后的模板打标签
void LabelContour(LocalDeforModelInfo& localDeforModelInfo);

//计算每个子轮廓的法向量
void ComputeContourNormal(const vector<Point2f>& contour, const vector<vector<uint>>& segContIdx, vector<Point2f>& normals);

//移动轮廓
void TranslationContour(const vector<Point2f>& contour, const vector<uint>& contIdx,
	const Point3f& normals, vector<Point2f>& tranContour, int transLen);

//顶层匹配
void TopMatch(const Mat &s_x, const Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad,
	const vector<vector<uint>>& segIdx, const vector<Point3f>& contNormals, float minScore, float angle, LocalMatchRes& reses);

//匹配
void Match(const Mat &image, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<uint>>& segIdx, const vector<Point3f>& contNormals,
	cv::Point center, float minScore, float angle, vector<int>& transLen_down, LocalMatchRes& reses);

//获取平移量
void GetTranslation(vector<int>& segContMapIdx, vector<int>& transLen_up, vector<int>& transLen_down);

//旋转方向向量
void RotContourVec(const vector<Point2f>& srcVec, vector<Point2f>& dstVec, float angle);

//上层映射到下层
void UpMapToDown(LocalDeforModelInfo& up_, LocalDeforModelInfo& down_, vector<int>& transLen_up, vector<vector<int>>& transLen_down);

//匹配
void LocalDeforModelMatch(Mat &modImg, LocalDeforModel* &model);

void LocalDeforModelTest();