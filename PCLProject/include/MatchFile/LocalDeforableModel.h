#pragma once
#include "../BaseOprFile/utils.h"
#include "ShapeModelBase.h"

struct LocalDeforModelInfo
{
	vector<Point2f> coord;
	vector<Point2f> grad;
	Point2f gravity;
	vector<Point2f> normals_;
	vector<vector<int>> segContIdx;
	vector<int> segContMapIdx;
	LocalDeforModelInfo():gravity(0,0)
	{
		coord.clear();
		grad.clear();
		normals_.clear();
		segContIdx.clear();
		segContMapIdx.clear();
	}
};

struct LocalDeforModel
{
	int pyrNum;
	double startAng;
	double endAng;
	double angStep;
	double minScale;
	double maxScale;
	double minScore;
	double greediness;
	int transLen;
	vector<LocalDeforModelInfo> models;

	LocalDeforModel() :pyrNum(0), startAng(0.0), endAng(0.0),
		angStep(0.0), minScale(1.0), maxScale(1.0),
		minScore(0.5), greediness(0.9), transLen(2)
	{
		models.resize(0);
	}
};

struct LocalMatchRes : public MatchRes
{
	vector<int> translates;
	vector<bool> flags;
	vector<Point> gravitys;
	LocalMatchRes()
	{
		translates.clear();
		flags.clear();
		gravitys.clear();
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
void GetKNearestPoint(vector<Point2f> &contours, vector<Point2f> &grads, LocalDeforModelInfo &localDeforModelInfo, int ptNum);

//计算子轮廓的方向向量
void ComputeSegContourVec(LocalDeforModel &model);

//根据中心获取每个小轮廓的映射索引
void GetMapIndex(LocalDeforModel& localDeforModel);

//计算最高层的平移距离
void ComputeTopTransLen(LocalDeforModel& localDeforModel);

//平移轮廓
void TranslationContour(const vector<Point2f>& contour, const vector<int>& contIdx,
	const Point3f& normals, vector<Point2f>& tranContour, int transLen);

//顶层匹配
void TopMatch(const Mat &s_x, const Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<int>>& segIdx,
	const vector<Point3f>& normals_, double minScore, double angle, int transLenP, LocalMatchRes& reses);

//匹配
void Match(const Mat &image, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<int>>& segIdx, const vector<Point2f>& normals_,
	int* center, double minScore, double angle, vector<int>& transLen_down, vector<bool>& contourFlags, /*vector<Point>& gravitys, */LocalMatchRes& reses);

//获取平移量
void GetTranslation(vector<int>& segContMapIdx, LocalMatchRes& res, vector<int>& transLen_down,
	vector<bool>& contourFlags, vector<Point>& gravitys);

//旋转方向向量
void RotContourVec(const vector<Point2f>& srcVec, vector<Point2f>& dstVec, double rotAng);

//绘制匹配到的结果
void DrawLocDeforRes(Mat& image, LocalDeforModelInfo& models, LocalMatchRes& res, vector<bool>& contourFlags);

//匹配
void LocalDeforModelMatch(Mat &modImg, LocalDeforModel* &model);

void LocalDeforModelTest();