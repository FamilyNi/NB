#pragma once
#include "OPENCV_UTILS.h"
#include "ContourOpr.h"
#include "ShapeModelBase.h"
#include <map>

//所有层数的模板信息
struct ShapeModel
{
	int pyr_n;      //金字塔层数
	float s_ang;    //起始角度
	float e_ang;    //中止角度
	float angStep; //角度步长
	float minScore; //最小匹配度
	float greediness; //贪婪系数
	int res_n;       //匹配个数
	int min_x;     //非极大值抑制的x范围 
	int min_y;     //非极大值 抑制y范围
	vector<ShapeInfo> ShapeInfos;
	ShapeModel() :pyr_n(0), s_ang(0.0f), e_ang(0.0f), angStep(0.0f),
		minScore(0.5f), greediness(0.9f), res_n(0), min_x(0), min_y(0)
	{
		ShapeInfos.resize(0);
	}
};

//创建模板
bool CreateShapeModel(Mat &modImg, ShapeModel* &model, SPAPLEMODELINFO &shapeModelInfo);

//获得模板信息
void CreateModelInfo(vector<vector<Point2f>> &vv_Coord,vector<vector<float>> &vv_GradX, 
	vector<vector<float>> &vv_GradY, vector<Point2f> &v_Gravity, ShapeModel* &pShapeModel);

//寻找模板
void FindShapeModel(Mat &srcImg, ShapeModel *model, vector<MatchRes> &MatchReses);

//顶层匹配
void TopMatch(Mat &s_x, Mat &s_y, Mat &r_x, Mat &r_y, int p_n, float minScore, 
	float greediness, float angle, int min_x, int min_y, vector<MatchRes>& reses);

//匹配
void MatchShapeModel(const Mat &img, Mat &r_x, Mat &r_y, int p_n,
		float minScore, float greediness, float angle, int *center, MatchRes &matchRes);

//释放模板
void clear_model(ShapeModel* &pShapeModel);

//测试程序
void shape_match_test();