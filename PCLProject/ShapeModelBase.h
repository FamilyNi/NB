#pragma once
#include "utils.h"

//模板信息
struct ShapeInfo
{
	int p_n; //模板点个数 
	Mat x_;  //用来储存x方向上的坐标、梯度
	Mat y_; //用来储存y方向上的坐标、梯度
	Point2f g_; //模板质心
	ShapeInfo() :p_n(0), g_(Point2f(0.0f, 0.0f))
	{
		x_.release();
		y_.release();
	}
};

struct SPAPLEMODELINFO
{
	int pyrNumber;       // 金子塔层数
	int minContourLen;   //轮廓的最小长度
	int maxContourLen;  //轮廓的最大长度
	int lowVal;    //轮廓提取低阈值
	int highVal;   //轮廓提取高阈值
	int extContouMode;  //轮廓提取模式
	int step;   //选点步长
	float startAng;
	float endAng;
	float angStep;
	SPAPLEMODELINFO() :pyrNumber(1), minContourLen(0), maxContourLen(1e9),
		lowVal(15), highVal(30), step(3), extContouMode(0)
	{}
};

//输出结果
struct MatchRes
{
	int c_x;  //x中心
	int c_y;  //y_中心
	float angle;
	float score;
	MatchRes() :c_x(0), c_y(0), angle(0.0f), score(0.0f)
	{ }
	bool operator<(const MatchRes& res)
	{
		return this->score > res.score;
	}
	void reset()
	{
		this->c_x = 0;
		this->c_y = 0;
		this->angle = 0.0f;
		this->score = 0.0f;
	}
};

//计算非极大值抑制的长宽
void  ComputeNMSRange(vector<Point2f>& contour, int& min_x, int& min_y);

//计算图像金字塔
void get_pyr_image(Mat &srcImg, vector<Mat> &pyrImg, int pyrNumber);

//提取模板的轮廓
void ExtractModelContour(Mat &srcImg, SPAPLEMODELINFO &shapeModelInfo, vector<Point> &contour);

//提取模板 信息
void ExtractModelInfo(Mat &srcImg, vector<Point> &contour, vector<Point2f> &v_Coord, vector<Point2f> &v_Grad, vector<float> &v_Amplitude);

//归一化梯度
void NormalGrad(int grad_x, int grad_y, float &grad_xn, float &grad_yn);

//比较分数大小
bool CompareSore(MatchRes& a, MatchRes& b);

//非极大值抑制
void NMS(vector<MatchRes> &MatchReses, vector<MatchRes> &nmsRes, int x_min, int y_min);

//计算梯度
void ComputeGrad(const Mat &srcImg, int idx_x, int idx_y, int& grad_x, int& grad_y);

//坐标旋转以及梯度
void RotateCoordGrad(const Mat &x_, const Mat &y_, Mat &r_x, Mat &r_y, float rotAng);
void RotateCoordGrad(const vector<Point2f> &coord, const vector<Point2f> &grad,
	vector<Point2f> &r_coord, vector<Point2f> &r_grad, float rotAng);

//绘制轮廓
void draw_contours(Mat &srcImg, float *pCoord_x, float *pCoord_y, Point offset, int length);

//继续绘制轮廓
void draw_contours(Mat &srcImg, vector<Point2f> &v_Coord, Point2f offset);
void draw_contours(Mat &srcImg, vector<Point2f> &contours, vector<uint> &index, Point2f offset);

//减少匹配点个数
void ReduceMatchPoint(vector<Point2f> &v_Coord, vector<Point2f> &v_Grad, vector<float> &v_Amplitude,
	vector<Point2f> &v_RedCoord, vector<Point2f> &v_RedGrad, int step = 3);