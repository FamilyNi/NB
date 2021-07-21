#pragma once
#include "OPENCV_UTILS.h"
#include "ContourOpr.h"
#include "ShapeModelBase.h"
#include <map>

//���в�����ģ����Ϣ
struct ShapeModel
{
	int pyr_n;      //����������
	float s_ang;    //��ʼ�Ƕ�
	float e_ang;    //��ֹ�Ƕ�
	float angStep; //�ǶȲ���
	float minScore; //��Сƥ���
	float greediness; //̰��ϵ��
	int res_n;       //ƥ�����
	int min_x;     //�Ǽ���ֵ���Ƶ�x��Χ 
	int min_y;     //�Ǽ���ֵ ����y��Χ
	vector<ShapeInfo> ShapeInfos;
	ShapeModel() :pyr_n(0), s_ang(0.0f), e_ang(0.0f), angStep(0.0f),
		minScore(0.5f), greediness(0.9f), res_n(0), min_x(0), min_y(0)
	{
		ShapeInfos.resize(0);
	}
};

//����ģ��
bool CreateShapeModel(Mat &modImg, ShapeModel* &model, SPAPLEMODELINFO &shapeModelInfo);

//���ģ����Ϣ
void CreateModelInfo(vector<vector<Point2f>> &vv_Coord,vector<vector<float>> &vv_GradX, 
	vector<vector<float>> &vv_GradY, vector<Point2f> &v_Gravity, ShapeModel* &pShapeModel);

//Ѱ��ģ��
void FindShapeModel(Mat &srcImg, ShapeModel *model, vector<MatchRes> &MatchReses);

//����ƥ��
void TopMatch(Mat &s_x, Mat &s_y, Mat &r_x, Mat &r_y, int p_n, float minScore, 
	float greediness, float angle, int min_x, int min_y, vector<MatchRes>& reses);

//ƥ��
void MatchShapeModel(const Mat &img, Mat &r_x, Mat &r_y, int p_n,
		float minScore, float greediness, float angle, int *center, MatchRes &matchRes);

//�ͷ�ģ��
void clear_model(ShapeModel* &pShapeModel);

//���Գ���
void shape_match_test();