#include "ShapeMatch.h"
#include <omp.h>

//创建模板=============================================================================
bool CreateShapeModel(Mat &modImg, ShapeModel* &model, SPAPLEMODELINFO &shapeModelInfo)
{
	clear_model(model);
	model = new ShapeModel;
	vector<Mat> imgPry;
	get_pyr_image(modImg, imgPry, shapeModelInfo.pyrNumber);

	vector<Point2f> v_Gravity;
	vector<vector<Point2f>> vv_Coord;
	vector<vector<float>> vv_GradX, vv_GradY;
	for (int i = 0; i < imgPry.size(); i++)
	{
		vector<Point2f> v_Coord, v_RedCoord;
		vector<float> v_GradX, v_GradY, v_RedGradX, v_RedGradY, v_Amplitude;
		vector<Point> contour(0);
		ExtractModelContour(imgPry[i], shapeModelInfo, contour);
		if (contour.size() < 20)
			break;
		//提取模板信息
		ExtractModelInfo(imgPry[i], contour, v_Coord, v_GradX, v_GradY, v_Amplitude);
		if (v_Coord.size() != v_GradX.size() || v_GradX.size() != v_GradY.size() || v_Amplitude.size() != v_GradY.size())
			break;
		//减少点的个数
		ReduceMatchPoint(v_Coord, v_GradX, v_GradY, v_Amplitude, v_RedCoord, v_RedGradX, v_RedGradY, shapeModelInfo.step);
		//计算重心
		Point2f gravity;
		GetContourGraviry(v_RedCoord, gravity);
		TranContour(v_RedCoord, gravity);
		v_Gravity.push_back(gravity);
		vv_Coord.push_back(v_RedCoord);
		vv_GradX.push_back(v_RedGradX);
		vv_GradY.push_back(v_RedGradY);
		Mat colorImg;
		cvtColor(imgPry[i], colorImg, COLOR_GRAY2BGR);
		draw_contours(colorImg, v_RedCoord, gravity);
		model->pyr_n++;
	}
	if (vv_Coord.size() != vv_GradX.size() || vv_GradX.size() != vv_GradY.size()
		|| vv_GradY.size() != v_Gravity.size() || vv_Coord.size() != model->pyr_n)
		return false;
	CreateModelInfo(vv_Coord, vv_GradX, vv_GradY, v_Gravity, model);
	ComputeNMSRange(vv_Coord[model->pyr_n - 1], model->min_x, model->min_y);
	return true;
}
//========================================================================================

//获得模板信息============================================================================
void CreateModelInfo(vector<vector<Point2f>> &vv_Coord, vector<vector<float>> &vv_GradX,
	vector<vector<float>> &vv_GradY, vector<Point2f> &v_Gravity, ShapeModel* &pShapeModel)
{
	if (pShapeModel == nullptr)
		return;
	pShapeModel->ShapeInfos.resize(pShapeModel->pyr_n);
	for (int i = 0; i < pShapeModel->pyr_n; i++)
	{
		pShapeModel->ShapeInfos[i].g_ = v_Gravity[i];
		pShapeModel->ShapeInfos[i].x_ = Mat(Size(vv_Coord[i].size(), 2), CV_32FC1, Scalar(0));
		pShapeModel->ShapeInfos[i].y_ = Mat(Size(vv_Coord[i].size(), 2), CV_32FC1, Scalar(0));
		pShapeModel->ShapeInfos[i].p_n = vv_Coord[i].size();

		float* pCoordGradX0 = pShapeModel->ShapeInfos[i].x_.ptr<float>(0);
		float* pCoordGradX1 = pShapeModel->ShapeInfos[i].x_.ptr<float>(1);
		float* pCoordGradY0 = pShapeModel->ShapeInfos[i].y_.ptr<float>(0);
		float* pCoordGradY1 = pShapeModel->ShapeInfos[i].y_.ptr<float>(1);

		for (int j = 0; j < vv_Coord[i].size(); j++)
		{
			pCoordGradX0[j] = vv_Coord[i][j].x;
			pCoordGradX1[j] = vv_GradX[i][j];
			pCoordGradY0[j] = vv_Coord[i][j].y;
			pCoordGradY1[j] = vv_GradY[i][j];
		}
	}
}
//========================================================================================

//寻找模板================================================================================
void FindShapeModel(Mat &srcImg, ShapeModel *model, vector<MatchRes> &MatchReses)
{
	if (MatchReses.size() > 0)
		MatchReses.clear();
	const int pyr_n = model->pyr_n - 1;
	vector<Mat> imgPry;
	get_pyr_image(srcImg, imgPry, pyr_n + 1);
	double t3 = getTickCount();
	float angStep = model->angStep > 1 ? model->angStep : 1;
	float angleStep_ = angStep * pow(2, pyr_n + 1);

	int angNum = (model->e_ang - model->s_ang) / angleStep_ + 1;
	float minScore = model->minScore / (pyr_n + 1);
	int p_n = model->ShapeInfos[pyr_n].p_n;
	//顶层匹配
	Mat sobel_x, sobel_y;
	Sobel(imgPry[pyr_n], sobel_x, CV_16SC1, 1, 0, 3);
	Sobel(imgPry[pyr_n], sobel_y, CV_16SC1, 0, 1, 3);
	const Mat& top_x_ = model->ShapeInfos[pyr_n].x_;
	const Mat& top_y_ = model->ShapeInfos[pyr_n].y_;
	vector<vector<MatchRes>> mulMatchRes(angNum);
#pragma omp parallel for
	for (int i = 0; i < angNum; ++i)
	{
		vector<MatchRes> reses;
		float angle = model->s_ang + i * angleStep_;
		Mat r_x_ = Mat(Size(p_n, 2), CV_32FC1, Scalar(0));
		Mat r_y_ = Mat(Size(p_n, 2), CV_32FC1, Scalar(0));
		RotateCoordGrad(top_x_, top_y_, r_x_, r_y_, angle);
		TopMatch(sobel_x, sobel_y, r_x_, r_y_, p_n, minScore, model->greediness, angle, model->min_x, model->min_y, reses);
		mulMatchRes[i] = reses;
	}

	//进行非极大值抑制
	vector<MatchRes> totalNum;
	for (int i = 0; i < angNum; ++i)
	{
		for (int j = 0; j < mulMatchRes[i].size(); ++j)
		{
			totalNum.push_back(mulMatchRes[i][j]);
		}
	}
	vector<MatchRes> resNMS;	
	NMS(totalNum, resNMS, model->min_x, model->min_y);
	int res_n = std::min((int)resNMS.size(), model->res_n);

	Mat img1;
	Mat r_x, r_y;
	cvtColor(imgPry[pyr_n], img1, COLOR_GRAY2BGR);
	for (int i = 0; i < res_n; ++i)
	{
		RotateCoordGrad(model->ShapeInfos[pyr_n].x_, model->ShapeInfos[pyr_n].y_,
			r_x, r_y, resNMS[i].angle);
		draw_contours(img1, r_x.ptr<float>(0), r_y.ptr<float>(0),
			Point(resNMS[i].c_x, resNMS[i].c_y), model->ShapeInfos[pyr_n].p_n);
	}
	//其他层匹配
	for (int k = 0; k < res_n; ++k)
	{
		MatchRes& res_ = resNMS[k];
		for (int i = pyr_n - 1; i > -1; --i)
		{
			p_n = model->ShapeInfos[i].p_n;
			angleStep_ = angStep * pow(2, i);
			minScore = model->minScore / (i + 1);
			res_.score = 0.0f;
			int center[4] = { 2 * res_.c_x - 10, 2 * res_.c_y - 10,	2 * res_.c_x + 10, 2 * res_.c_y + 10 };
			const Mat& x_ = model->ShapeInfos[i].x_;
			const Mat& y_ = model->ShapeInfos[i].y_;
			const Mat& img = imgPry[i];
#pragma omp parallel for
			for (int j = -2; j <= 2; ++j)
			{
				float angle = res_.angle + j * angleStep_;
				Mat r_x_ = Mat(Size(p_n, 2), CV_32FC1, Scalar(0));
				Mat r_y_ = Mat(Size(p_n, 2), CV_32FC1, Scalar(0));
				RotateCoordGrad(x_, y_, r_x_, r_y_, angle);
				MatchShapeModel(img, r_x_, r_y_, p_n, minScore, model->greediness, angle, center, res_);
			}
		}
	}
	for (size_t i = 0; i < resNMS.size(); ++i)
	{
		if (resNMS[i].score > model->minScore)
			MatchReses.push_back(resNMS[i]);
	}
	double t4 = (getTickCount() - t3) / getTickFrequency();
	cout << "t4 = " << t4 << endl;

	Mat img;
	cvtColor(imgPry[0], img, COLOR_GRAY2BGR);
	for (size_t i = 0; i < MatchReses.size(); ++i)
	{
		RotateCoordGrad(model->ShapeInfos[0].x_, model->ShapeInfos[0].y_,	r_x, r_y, MatchReses[i].angle);
		draw_contours(img, r_x.ptr<float>(0), r_y.ptr<float>(0),
			Point(MatchReses[i].c_x, MatchReses[i].c_y), model->ShapeInfos[0].p_n);
	}
	return;
}
//====================================================================================================

//匹配================================================================================================
void TopMatch(Mat &s_x, Mat &s_y, Mat &r_x, Mat &r_y, int p_n, float minScore,
	float greediness, float angle, int min_x, int min_y, vector<MatchRes>& reses)
{
	vector<MatchRes> reses_;
	int maxW = s_x.cols - 2;
	int maxH = s_x.rows - 2;
	float NormGreediness = ((1 - greediness * minScore) / (1 - greediness)) / p_n;
	float anMinScore = 1 - minScore;
	float NormMinScore = minScore / p_n;

	float* pCoord_x = r_x.ptr<float>(0);
	float* pGrad_x = r_x.ptr<float>(1);
	float* pCoord_y = r_y.ptr<float>(0);
	float* pGrad_y = r_y.ptr<float>(1);

	for (int y = 2; y < maxH; ++y)
	{
		for (int x = 2; x < maxW; ++x)
		{
			float partial_score = 0.0f, score = 0.0f;
			int sum = 0.0;
			for (int index = 0; index < p_n; index++)
			{
				int cur_x = x + pCoord_x[index];
				int cur_y = y + pCoord_y[index];
				++sum;
				if (cur_x < 2 || cur_y < 2 || cur_x > maxW || cur_y > maxH)
					continue;
				short gx = s_x.at<short>(cur_y, cur_x);
				short gy = s_y.at<short>(cur_y, cur_x);
				if (abs(gx) > 0 || abs(gy) > 0)
				{
					float grad_x = 0.0f, grad_y = 0.0f;
					NormalGrad((int)gx, (int)gy, grad_x, grad_y);
					partial_score += (grad_x * pGrad_x[index] + grad_y * pGrad_y[index]);
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
				reses_.push_back(matchRes);
			}
		}
	}
	NMS(reses_, reses, min_x, min_y);
}
void MatchShapeModel(const Mat &image, Mat &RotCG_X, Mat &RotCG_Y, int length,
	float minScore, float greediness, float angle, int *center, MatchRes &matchRes)
{
	int maxW = image.cols - 2;
	int maxH = image.rows - 2;

	float NormGreediness = ((1 - greediness * minScore) / (1 - greediness)) / length;
	float anMinScore = 1 - minScore;
	float NormMinScore = minScore / length;

	float* pCoord_x = RotCG_X.ptr<float>(0);
	float* pGrad_x = RotCG_X.ptr<float>(1);
	float* pCoord_y = RotCG_Y.ptr<float>(0);
	float* pGrad_y = RotCG_Y.ptr<float>(1);

	for (int y = center[1]; y < center[3]; y++)
	{
		for (int x = center[0]; x < center[2]; x++)
		{
			float partial_score = 0.0f, score = 0.0;
			int sum = 0.0;
			for (int index = 0; index < length; index++)
			{
				int cur_x = x + pCoord_x[index];
				int cur_y = y + pCoord_y[index];
				++sum;
				if (cur_x < 2 || cur_y < 2 || cur_x > maxW || cur_y > maxH)
					continue;

				int gx = 0, gy = 0;
				ComputeGrad(image, cur_x, cur_y, gx, gy);
				if (abs(gx) > 0 || abs(gy) > 0)
				{
					float grad_x = 0.0f, grad_y = 0.0f;
					NormalGrad(gx, gy, grad_x, grad_y);
					partial_score += (grad_x * pGrad_x[index] + grad_y * pGrad_y[index]);
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

//重设模板=============================================================================
void clear_model(ShapeModel* &pShapeModel)
{
	if (pShapeModel != nullptr)
	{
		if (pShapeModel->ShapeInfos.size() != 0)
		{
			pShapeModel->ShapeInfos.resize(0);
		}
		delete pShapeModel;
		pShapeModel = nullptr;
	}
}
//=====================================================================================

void shape_match_test()
{
	string imgPath = "D:/data/定位测试图片/model1.bmp";
	Mat modImg = imread(imgPath, 0);
	ShapeModel *model = new ShapeModel;

	SPAPLEMODELINFO shapeModelInfo;
	shapeModelInfo.pyrNumber = 4;
	shapeModelInfo.lowVal = 100;
	shapeModelInfo.highVal = 200;
	shapeModelInfo.step = 3;

	CreateShapeModel(modImg, model, shapeModelInfo);
	vector<MatchRes> v_MatchRes;

	model->s_ang = -180;
	model->e_ang = 180;
	model->res_n = 3;
	string testImgPath = "D:/data/定位测试图片/b.bmp";
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