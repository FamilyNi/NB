#include "ShapeMatch.h"
#include <omp.h>

//创建模板=============================================================================
bool CreateShapeModel(Mat &modImg, ShapeModel* &model, SPAPLEMODELINFO &shapeModelInfo)
{
	ClearModel(model);
	model = new ShapeModel;

	model->angStep = shapeModelInfo.angStep;
	model->startAng = shapeModelInfo.startAng;
	model->endAng = shapeModelInfo.endAng;

	vector<Mat> imgPry;
	GetPyrImg(modImg, imgPry, shapeModelInfo.pyrNumber);

	for (int i = 0; i < imgPry.size(); i++)
	{
		//提取轮廓
		vector<Point> v_Coord_;
		ExtractModelContour(imgPry[i], shapeModelInfo, v_Coord_);
		if (v_Coord_.size() < 10)
			break;
		//提取模板梯度信息
		vector<Point2f> v_Coord, v_Grad;
		vector<float> v_Amplitude;
		ExtractModelInfo(imgPry[i], v_Coord_, v_Coord, v_Grad, v_Amplitude);
		if (v_Coord.size() < 10)
			break;
		//减少轮廓点个数
		ShapeInfo models;
		ReduceMatchPoint(v_Coord, v_Grad, v_Amplitude, models.coord, models.grad, shapeModelInfo.step);
		//计算重心
		GetContourGravity(models.coord, models.gravity);
		//中心化轮廓
		Point2f gravity = Point2f(-models.gravity.x, -models.gravity.y);
		TranContour(models.coord, gravity);
		Mat colorImg;
		cvtColor(imgPry[i], colorImg, COLOR_GRAY2BGR);
		DrawContours(colorImg, models.coord, models.gravity);
		model->models.push_back(models);
		model->pyrNum++;
	}
	ComputeNMSRange(model->models[model->pyrNum - 1].coord, model->min_x, model->min_y);
	return true;
}
//========================================================================================

//匹配================================================================================================
void TopMatch(Mat &s_x, Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad,
	float minScore, float greediness, float angle, vector<MatchRes>& reses)
{
	vector<MatchRes> reses_;
	int maxW = s_x.cols - 2;
	int maxH = s_x.rows - 2;
	float NormGreediness = ((1 - greediness * minScore) / (1 - greediness)) / r_coord.size();
	float anMinScore = 1 - minScore;
	float NormMinScore = minScore / r_coord.size();

	for (int y = 2; y < maxH; ++y)
	{
		for (int x = 2; x < maxW; ++x)
		{
			float partial_score = 0.0f, score = 0.0f;
			int sum = 0.0;
			for (int index = 0; index < r_coord.size(); index++)
			{
				int cur_x = x + r_coord[index].x;
				int cur_y = y + r_coord[index].y;
				++sum;
				if (cur_x < 2 || cur_y < 2 || cur_x > maxW || cur_y > maxH)
					continue;
				short gx = s_x.at<short>(cur_y, cur_x);
				short gy = s_y.at<short>(cur_y, cur_x);
				if (abs(gx) > 0 || abs(gy) > 0)
				{
					float grad_x = 0.0f, grad_y = 0.0f;
					NormalGrad((int)gx, (int)gy, grad_x, grad_y);
					partial_score += (grad_x * r_grad[index].x + grad_y * r_grad[index].y);
					score = partial_score / sum;
					if (score < (min(anMinScore + NormGreediness * sum, NormMinScore * sum)))
						break;
				}
			}
			if (score > minScore)
			{
				MatchRes matchRes;
				matchRes.score = score;
				matchRes.c_x = x;
				matchRes.c_y = y;
				matchRes.angle = angle;
				reses.push_back(matchRes);
			}
		}
	}
}
void MatchShapeModel(const Mat &image, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad,
	float minScore, float greediness, float angle, int *center, MatchRes &matchRes)
{
	int maxW = image.cols - 2;
	int maxH = image.rows - 2;

	float NormGreediness = ((1 - greediness * minScore) / (1 - greediness)) / r_coord.size();
	float anMinScore = 1 - minScore;
	float NormMinScore = minScore / r_coord.size();

	for (int y = center[1]; y < center[3]; y++)
	{
		for (int x = center[0]; x < center[2]; x++)
		{
			float partial_score = 0.0f, score = 0.0;
			int sum = 0.0;
			for (int index = 0; index < r_coord.size(); index++)
			{
				int cur_x = x + r_coord[index].x;
				int cur_y = y + r_coord[index].y;
				++sum;
				if (cur_x < 2 || cur_y < 2 || cur_x > maxW || cur_y > maxH)
					continue;

				int gx = 0, gy = 0;
				ComputeGrad(image, cur_x, cur_y, gx, gy);
				if (abs(gx) > 0 || abs(gy) > 0)
				{
					float grad_x = 0.0f, grad_y = 0.0f;
					NormalGrad(gx, gy, grad_x, grad_y);
					partial_score += (grad_x * r_grad[index].x + grad_y * r_grad[index].y);
					score = partial_score / sum;
					if (score < (min(anMinScore + NormGreediness * sum, NormMinScore * sum)))
						break;
				}
			}
			if (score > matchRes.score)
			{
				matchRes.score = score;
				matchRes.c_x = x;
				matchRes.c_y = y;
				matchRes.angle = angle;
			}
		}
	}
}
//====================================================================================================

//绘制轮廓============================================================================================
void DrawShapeRes(Mat& image, ShapeInfo& models, vector<MatchRes>& res)
{
	for (int i = 0; i < res.size(); ++i)
	{
		vector<Point2f> r_coord, r_grad;
		RotateCoordGrad(models.coord, models.grad, r_coord, r_grad, res[i].angle);
		DrawContours(image, r_coord, Point2f(res[i].c_x, res[i].c_y));
	}
}
//====================================================================================================

//重设模板=============================================================================
void ClearModel(ShapeModel* &pModel)
{
	if (pModel != nullptr)
	{
		if (pModel->models.size() != 0)
		{
			pModel->models.clear();
		}
		delete pModel;
		pModel = nullptr;
	}
}
//=====================================================================================

//寻找模板================================================================================
void FindShapeModel(Mat &srcImg, ShapeModel *model, vector<MatchRes> &MatchReses)
{
	if (MatchReses.size() > 0)
		MatchReses.clear();
	const int pyr_n = model->pyrNum - 1;
	vector<Mat> imgPry;
	GetPyrImg(srcImg, imgPry, pyr_n + 1);
	float angStep = model->angStep > 1 ? model->angStep : 1;
	float angleStep_ = angStep * pow(2, pyr_n + 1);

	int angNum = (model->endAng - model->startAng) / angleStep_ + 1;
	//顶层匹配
	Mat sobel_x, sobel_y;
	Sobel(imgPry[pyr_n], sobel_x, CV_16SC1, 1, 0, 3);
	Sobel(imgPry[pyr_n], sobel_y, CV_16SC1, 0, 1, 3);
	vector<vector<MatchRes>> mulMatchRes(angNum);
#pragma omp parallel for
	for (int i = 0; i < angNum; ++i)
	{
		vector<MatchRes> reses;
		float angle = model->startAng + i * angleStep_;
		vector<Point2f> r_coord, r_grad;
		RotateCoordGrad(model->models[pyr_n].coord, model->models[pyr_n].grad, r_coord, r_grad, angle);
		TopMatch(sobel_x, sobel_y, r_coord, r_grad, model->minScore, model->greediness, angle, reses);
		mulMatchRes[i] = reses;
	}

	//进行非极大值抑制
	vector<MatchRes> resNMS;
	ShapeNMS(mulMatchRes, resNMS, model->min_x, model->min_y, model->res_n);

	Mat image0;
	cvtColor(imgPry[pyr_n], image0, COLOR_GRAY2BGR);
	DrawShapeRes(image0, model->models[pyr_n], resNMS);

	//其他层匹配
	vector<MatchRes> reses_(5);
	for (int k = 0; k < resNMS.size(); ++k)
	{
		for (int i = pyr_n - 1; i > -1; --i)
		{
			for (size_t j = 0; j < 5; ++j)
				reses_[j].score = 0.0f;
			angleStep_ = angStep * pow(2, i);
			float minScore = model->minScore / (i + 1);
			int center[4] = { 2 * resNMS[k].c_x - 10, 2 * resNMS[k].c_y - 10, 2 * resNMS[k].c_x + 10, 2 * resNMS[k].c_y + 10 };
#pragma omp parallel for
			for (int j = -2; j <= 2; ++j)
			{
				float angle = resNMS[k].angle + j * angleStep_;
				vector<Point2f> r_coord, r_grad;
				RotateCoordGrad(model->models[i].coord, model->models[i].grad, r_coord, r_grad, angle);
				MatchShapeModel(imgPry[i], r_coord, r_grad, minScore, model->greediness, angle, center, reses_[j+2]);
			}
			std::stable_sort(reses_.begin(), reses_.end());
			resNMS[k] = reses_[0];
		}
	}
	for (size_t i = 0; i < resNMS.size(); ++i)
	{
		if (resNMS[i].score > model->minScore)
			MatchReses.push_back(resNMS[i]);
	}

	Mat img;
	cvtColor(imgPry[0], img, COLOR_GRAY2BGR);
	DrawShapeRes(img, model->models[0], MatchReses);
	return;
}
//====================================================================================================

void shape_match_test()
{
	string imgPath = "../image/model1.bmp";
	Mat modImg = imread(imgPath, 0);
	ShapeModel *model = new ShapeModel;

	SPAPLEMODELINFO shapeModelInfo;
	shapeModelInfo.pyrNumber = 4;
	shapeModelInfo.lowVal = 100;
	shapeModelInfo.highVal = 200;
	shapeModelInfo.step = 3;

	CreateShapeModel(modImg, model, shapeModelInfo);
	vector<MatchRes> v_MatchRes;

	model->startAng = -180;
	model->endAng = 180;
	model->res_n = 3;
	string testImgPath = "../image/f.bmp";
	Mat testImg = imread(testImgPath, 0);
	FindShapeModel(testImg, model, v_MatchRes);
	//Mat testImg = imread("5.png", 0);

	//MatchRes matchRes;
	//double t1 = getTickCount();
	//FindShapeModel(testImg, model, 0.5, -90, 90, 1, 0.9, matchRes);
	//double t = (getTickCount() - t1)/ getTickFrequency();
	//cout << t;
	return;
}