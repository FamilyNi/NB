#pragma once
#include "../BaseOprFile/utils.h"
#include "ContourOpr.h"
#include "ShapeModelBase.h"
#include <map>

//模板信息
struct ShapeInfo
{
	vector<Point2f> coord;
	vector<Point2f> grad;
	Point2f gravity;
	ShapeInfo() :gravity(Point2f(0.0f, 0.0f))
	{
		coord.clear();
		grad.clear();
	}
};

//所有层数的模板信息
struct ShapeModel
{
	int pyrNum;
	float startAng;
	float endAng;
	float angStep;
	float minScore;
	float greediness;
	int res_n;       //匹配个数
	int min_x;     //非极大值抑制的x范围 
	int min_y;     //非极大值 抑制y范围
	vector<ShapeInfo> models;
	ShapeModel() :pyrNum(0), startAng(0.0f), endAng(0.0f), angStep(0.0f),
		minScore(0.5f), greediness(0.9f), res_n(0), min_x(0), min_y(0)
	{
		models.resize(0);
	}
};

//创建模板
bool CreateShapeModel(Mat &modImg, ShapeModel* &model, SPAPLEMODELINFO &shapeModelInfo);

//寻找模板
void FindShapeModel(Mat &srcImg, ShapeModel *model, vector<MatchRes> &MatchReses);

//顶层匹配
void TopMatch(Mat &s_x, Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad,
	float minScore, float greediness, float angle, vector<MatchRes>& reses);

//匹配
void MatchShapeModel(const Mat &image, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad,
	float minScore, float greediness, float angle, int *center, MatchRes &matchRes);

//绘制轮廓
void DrawShapeRes(Mat& image, ShapeInfo& models, vector<MatchRes>& res);

//释放模板
void ClearModel(ShapeModel* &pModel);

//测试程序
void shape_match_test();