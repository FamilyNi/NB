#include "opencv2/flann.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/flann.hpp>
#include "../../include/MatchFile/ContourOpr.h"
#include "../../include/MatchFile/LocalDeforableModel.h"

//创建模板================================================================================
void CreateLocalDeforableModel(Mat &modImg, LocalDeforModel* &model, SPAPLEMODELINFO &shapeModelInfo)
{
	if (model == nullptr)
		return;

	model->angStep = shapeModelInfo.angStep;
	model->startAng = shapeModelInfo.startAng;
	model->endAng = shapeModelInfo.endAng;
	model->minScale = shapeModelInfo.minScale;
	model->maxScale = shapeModelInfo.maxScale;

	vector<Mat> imgPry;
	GetPyrImg(modImg, imgPry, shapeModelInfo.pyrNumber);

	int ptNum = 8;
	for (int i = 0; i < imgPry.size(); i++)
	{
		//提取轮廓
		vector<Point> v_Coord_;
		ExtractModelContour(imgPry[i], shapeModelInfo, v_Coord_);
		if (v_Coord_.size() < 1)
			break;
		//提取模板梯度信息
		vector<Point2f> v_Coord, v_Grad;
		vector<float> v_Amplitude;
		ExtractModelInfo(imgPry[i], v_Coord_, v_Coord, v_Grad, v_Amplitude);
		if (v_Coord.size() < 10)
			break;
		//减少轮廓点个数
		vector<Point2f> v_RedCoord, v_RedGrad;
		ReduceMatchPoint(v_Coord, v_Grad, v_Amplitude, v_RedCoord, v_RedGrad, shapeModelInfo.step);
		//聚类
		LocalDeforModelInfo models;
		GetKNearestPoint(v_RedCoord, v_RedGrad, models, ptNum);
		ptNum = std::max(ptNum - 1, 5);
		//计算重心
		GetContourGravity(models.coord, models.gravity);
		//中心化轮廓
		Point2f gravity = Point2f(-models.gravity.x, -models.gravity.y);
		TranContour(models.coord, gravity);

		model->models.push_back(models);
		Mat colorImg;
		cvtColor(imgPry[i], colorImg, COLOR_GRAY2BGR);
		DrawContours(colorImg, models.coord, models.gravity);
		model->pyrNum++;
	}
	//轮廓从上层到下层的映射索引
	GetMapIndex(*model);
	//计算自轮廓的方向向量
	ComputeSegContourVec(*model);
	//计算最高层的平移参数
	ComputeTopTransLen(*model);
}
//========================================================================================

//模板点聚类==============================================================================
void GetKNearestPoint(vector<Point2f> &contours, vector<Point2f> &grads, LocalDeforModelInfo &localDeforModelInfo, int ptNum)
{
	localDeforModelInfo.coord = contours;
	localDeforModelInfo.grad = grads;

	Mat centers, labels;
	int clusterCount = localDeforModelInfo.coord.size() / ptNum;
	Mat points = Mat(localDeforModelInfo.coord);
	kmeans(localDeforModelInfo.coord, clusterCount, labels,	
		TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, 0.1),	3, KMEANS_PP_CENTERS, centers);

	localDeforModelInfo.segContIdx.resize(centers.rows);
	int* pLabel = labels.ptr<int>();
	for (uint i = 0; i < labels.rows; ++i)
	{
		localDeforModelInfo.segContIdx[pLabel[i]].push_back(i);
	}
}
//========================================================================================

//计算子轮廓的法向量======================================================================
void ComputeSegContourVec(LocalDeforModel &model)
{
	if (model.models.size() == 0)
		return;
	for (size_t i = 0; i < model.models.size(); ++i)
	{
		LocalDeforModelInfo& models_ = model.models[i];
		size_t segContNum = models_.segContIdx.size();
		if (models_.normals_.size() != segContNum)
			models_.normals_.resize(segContNum);
		for (size_t j = 0; j < segContNum; ++j)
		{
			const vector<int>& segCont = models_.segContIdx[j];
			vector<Point2f> fitLinePoint(segCont.size());
			for (size_t k = 0; k < segCont.size(); ++k)
			{
				fitLinePoint[k] = models_.coord[segCont[k]];
			}
			Vec4f line_;
			fitLine(fitLinePoint, line_, DIST_L2, 0, 0.01, 0.01);
			models_.normals_[j].x = -line_[1];
			models_.normals_[j].y = line_[0];
			cv::Point2f ref_p = fitLinePoint[0];
			float cosVal = ref_p.x * models_.normals_[j].x + ref_p.y * models_.normals_[j].y;
			if (cosVal > 0)
			{
				models_.normals_[j].x = -models_.normals_[j].x;
				models_.normals_[j].y = -models_.normals_[j].y;
			}
		}
	}
}
//========================================================================================

//根据中心获取每个小轮廓的映射索引========================================================
void GetMapIndex(LocalDeforModel& localDeforModel)
{
	if (localDeforModel.models.size() < 2)
		return;
	for (size_t i = localDeforModel.models.size() - 1; i > 0; --i)
	{
		//获取上层轮廓各段轮廓的重心
		LocalDeforModelInfo& up_ = localDeforModel.models[i];
		vector<Point2f> gravitys_up(up_.segContIdx.size());
		for (size_t j = 0; j < up_.segContIdx.size(); ++j)
		{
			GetIdxContourGravity(up_.coord, up_.segContIdx[j], gravitys_up[j]);
		}

		//knn最近邻搜索
		LocalDeforModelInfo& down_ = localDeforModel.models[i - 1];
		Mat source = cv::Mat(gravitys_up).reshape(1);
		down_.segContMapIdx.resize(down_.segContIdx.size());
		cv::flann::KDTreeIndexParams indexParams(2);
		cv::flann::Index kdtree(source, indexParams);
		for (size_t j = 0; j < down_.segContIdx.size(); ++j)
		{
			Point2f gravity(0.0f, 0.0f);
			GetIdxContourGravity(down_.coord, down_.segContIdx[j], gravity);

			/**KD树knn查询**/
			vector<float> vecQuery(2);//存放查询点
			vecQuery[0] = gravity.x * 0.5f - 2.0f; //查询点x坐标
			vecQuery[1] = gravity.y * 0.5f - 2.0f; //查询点y坐标
			vector<int> vecIndex(1);//存放返回的点索引
			vector<float> vecDist(1);//存放距离
			cv::flann::SearchParams params(32);//设置knnSearch搜索参数
			kdtree.knnSearch(vecQuery, vecIndex, vecDist, 1, params);
			down_.segContMapIdx[j] = vecIndex[0];
		}
	}
}
//========================================================================================

//计算最高层的平移距离====================================================================
void ComputeTopTransLen(LocalDeforModel& localDeforModel)
{
	cv::RotatedRect rect = cv::minAreaRect(localDeforModel.models[0].coord);
	double scale = std::max(abs(localDeforModel.minScale - 1), abs(localDeforModel.maxScale - 1));
	double height = rect.size.height * scale;
	double width = rect.size.width * scale;
	double maxLen = std::max(height, width);
	localDeforModel.transLen = maxLen / pow(2, localDeforModel.pyrNum);
	localDeforModel.transLen = std::max(2, localDeforModel.transLen);
}
//========================================================================================

//平移轮廓================================================================================
void TranslationContour(const vector<Point2f>& contour, const vector<int>& contIdx, 
	const Point2f& normal_, vector<Point2f>& tranContour, int transLen)
{
	if (tranContour.size() != contIdx.size())
		tranContour.resize(contIdx.size());
	for (int i = 0; i < contIdx.size(); ++i)
	{
		tranContour[i].x = contour[contIdx[i]].x + transLen * normal_.x;
		tranContour[i].y = contour[contIdx[i]].y + transLen * normal_.y;
	}
}
//========================================================================================

//顶层匹配================================================================================
void TopMatch(const Mat &s_x, const Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<int>>& segIdx, 
	const vector<Point2f>& normals_, double minScore, double angle, int transLenP, LocalMatchRes& reses)
{
	int segNum = segIdx.size();
	int maxW = s_x.cols - 1, maxH = s_x.rows - 1;
	double NormGreediness = ((1 - 0.9 * minScore) / (1 - 0.9)) / segNum;
	double anMinScore = 1 - minScore, NormMinScore = minScore / segNum;

	vector<int> v_TransLen_(segNum);
	for (int y = 0; y < maxH; ++y)
	{
		for (int x = 0; x < maxW; ++x)
		{
			double partial_score = 0.0f, score = 0.0f;
			for (int index = 0; index < segNum; index++)
			{
				int sum_i = index + 1;
				double segContScore = 0.0f;
				//平移部分==================
				for (int transLen = -1; transLen <= 1; transLen += 1)
				{
					//计算重心点到轮廓的距离
					float segContScore_t = 0.0f;
					vector<Point2f> tranContour;
					TranslationContour(r_coord, segIdx[index], normals_[index], tranContour, transLen);
					for (int i = 0; i < tranContour.size(); ++i)
					{
						int idx = segIdx[index][i];
						int cur_x = x + tranContour[i].x;
						int cur_y = y + tranContour[i].y;
						if (cur_x < 0 || cur_y < 0 || cur_x > maxW || cur_y > maxH)
							continue;
						short gx = s_x.at<short>(cur_y, cur_x);
						short gy = s_y.at<short>(cur_y, cur_x);
						if (abs(gx) > 0 || abs(gy) > 0)
						{
							float grad_x = 0.0f, grad_y = 0.0f;
							NormalGrad((int)gx, (int)gy, grad_x, grad_y);
							segContScore_t += abs(grad_x * r_grad[idx].x + grad_y * r_grad[idx].y);
						}
					}
					if (segContScore < segContScore_t)
					{
						segContScore = segContScore_t;
						if (segContScore > minScore)
						{
							v_TransLen_[index] = transLen;
							reses.flags[index] = true;
							reses.gravitys[index] = Point(x, y);
						}
						else
						{
							reses.flags[index] = false;
						}
					}
				}
				if (segContScore > minScore)
				{
					reses.flags[index] = true;
					partial_score += segContScore / segIdx[index].size();
				}
				else
				{
					reses.flags[index] = false;
				}
				score = partial_score / sum_i;
				if (score < (min(anMinScore + NormGreediness * sum_i, NormMinScore * sum_i)))
					break;
			}
			if (score > reses.score)
			{
				reses.score = score;
				reses.c_x = x;
				reses.c_y = y;
				reses.angle = angle;
				for (int j = 0; j < segNum; ++j)
				{
					reses.translates[j] = v_TransLen_[j];
				}
			}
		}
	}
}
//========================================================================================

//匹配====================================================================================
void Match(const Mat &image, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<int>>& segIdx, const vector<Point2f>& normals_, 
	int* center, double minScore, double angle, vector<int>& transLen_down, vector<bool>& contourFlags, /*vector<Point>& gravitys,*/ LocalMatchRes& reses)
{
	int segNum = segIdx.size();
	float NormGreediness = ((1 - 0.9 * minScore) / (1 - 0.9)) / segNum;
	float anMinScore = 1 - minScore, NormMinScore = minScore / segNum;

	vector<int> v_TransLen_(segNum);
	for (int y = center[1]; y < center[3]; ++y)
	{
		for (int x = center[0]; x < center[2]; ++x)
		{
			float partial_score = 0.0f, score = 0.0f;
			for (int index = 0; index < segNum; index++)
			{
				if (!contourFlags[index])
					continue;
				int sum_i = index + 1;
				double segContScore = 0.0f;
				//平移部分==================
				for (int transLen = -5; transLen <= 5; transLen += 1)
				{
					double segContScore_t = 0.0f;
					vector<Point2f> tranContour;
					TranslationContour(r_coord, segIdx[index], normals_[index], tranContour, transLen + transLen_down[index]);
					for (int i = 0; i < tranContour.size(); ++i)
					{
						uint idx = segIdx[index][i];
						int cur_x = x + tranContour[i].x;
						int cur_y = y + tranContour[i].y;
						if (cur_x < 1 || cur_y < 1 || cur_x > image.cols - 2 || cur_y > image.rows - 2)
							continue;
						int gx = 0, gy = 0;
						ComputeGrad(image, cur_x, cur_y, gx, gy);
						if (abs(gx) > 0 || abs(gy) > 0)
						{
							float grad_x = 0.0f, grad_y = 0.0f;
							NormalGrad(gx, gy, grad_x, grad_y);
							segContScore_t += abs(grad_x * r_grad[idx].x + grad_y * r_grad[idx].y);
						}
					}
					if (segContScore < segContScore_t)
					{
						segContScore = segContScore_t;
						if (segContScore > minScore)
						{
							v_TransLen_[index] = transLen;
						}
					}
				}
				if (segContScore > minScore)
				{
					reses.flags[index] = true;
					partial_score += segContScore / segIdx[index].size();
				}
				else
				{
					reses.flags[index] = false;
				}
				score = partial_score / sum_i;
				if (score < (min(anMinScore + NormGreediness * sum_i, NormMinScore * sum_i)))
					break;
			}
			if (score > reses.score)
			{
				MatchRes matchRes;
				reses.score = score;
				reses.c_x = x;
				reses.c_y = y;
				reses.angle = angle;
				for (size_t j = 0; j < segNum; ++j)
				{
					reses.translates[j] = v_TransLen_[j] + transLen_down[j];
				}
			}
		}
	}
}
//========================================================================================

//旋转方向向量============================================================================
void RotContourVec(const vector<Point2f>& srcVec, vector<Point2f>& dstVec, double rotAng)
{
	double rotRad = rotAng / 180 * CV_PI;
	double sinVal = sin(rotRad);
	double cosVal = cos(rotRad);
	if (dstVec.size() != srcVec.size())
		dstVec.resize(srcVec.size());
	for (int i = 0; i < dstVec.size(); ++i)
	{
		dstVec[i].x = srcVec[i].x * cosVal - srcVec[i].y * sinVal;
		dstVec[i].y = srcVec[i].y * cosVal + srcVec[i].x * sinVal;
	}
}
//========================================================================================

//获取平移量==============================================================================
void GetTranslation(vector<int>& segContMapIdx, LocalMatchRes& res, vector<int>& transLen_down, 
	vector<bool>& contourFlags, vector<Point>& gravitys)
{
	if (transLen_down.size() != segContMapIdx.size())
		transLen_down.resize(segContMapIdx.size());
	if (contourFlags.size() != segContMapIdx.size())
		contourFlags.resize(segContMapIdx.size());
	for (int i = 0; i < segContMapIdx.size(); ++i)
	{
		transLen_down[i] = 2.0 * res.translates[segContMapIdx[i]];
		contourFlags[i] = res.flags[segContMapIdx[i]];
	}
}
//========================================================================================

//绘制匹配到的结果========================================================================
void DrawLocDeforRes(Mat& image, LocalDeforModelInfo& models, LocalMatchRes& res, vector<bool>& contourFlags)
{
	vector<Point2f> r_coord, r_grad;
	RotateCoordGrad(models.coord, models.grad, r_coord, r_grad, res.angle);
	vector<Point2f> r_t_coord;
	vector<Point2f> normals_;
	RotContourVec(models.normals_, normals_, res.angle);
	for (int i = 0; i < models.segContIdx.size(); ++i)
	{
		if (contourFlags[i])
		{
			vector<Point2f> tranContour;
			TranslationContour(r_coord, models.segContIdx[i], normals_[i], tranContour, res.translates[i]);
			DrawContours(image, tranContour, cv::Point2f(res.c_x, res.c_y));
		}
	}
}
//========================================================================================

//匹配====================================================================================
void LocalDeforModelMatch(Mat &srcImg, LocalDeforModel* &model)
{
	const int pyr_n = model->pyrNum - 1;
	vector<Mat> imgPry;
	GetPyrImg(srcImg, imgPry, pyr_n + 1);
	double angStep = model->angStep > 1 ? model->angStep : 1;
	double angleStep_ = angStep * pow(2, pyr_n);

	//顶层匹配
	int angNum = (model->endAng - model->startAng) / angleStep_ + 1;
	Mat sobel_x, sobel_y;
	Sobel(imgPry[pyr_n], sobel_x, CV_16SC1, 1, 0, 3);
	Sobel(imgPry[pyr_n], sobel_y, CV_16SC1, 0, 1, 3);
	vector<LocalMatchRes> reses_(angNum);
#pragma omp parallel for
	for (int i = 0; i < angNum; ++i)
	{
		reses_[i].translates.resize(model->models[pyr_n].segContIdx.size());
		reses_[i].flags.resize(model->models[pyr_n].segContIdx.size());
		reses_[i].gravitys.resize(model->models[pyr_n].segContIdx.size());
		double angle = model->startAng + i * angleStep_;
		vector<Point2f> r_coord, r_grad;
		RotateCoordGrad(model->models[pyr_n].coord, model->models[pyr_n].grad, r_coord, r_grad, angle);
		vector<Point2f> normals_;
		RotContourVec(model->models[pyr_n].normals_, normals_, angle);
		TopMatch(sobel_x, sobel_y, r_coord, r_grad, model->models[pyr_n].segContIdx, normals_, model->minScore, angle, model->transLen, reses_[i]);
	}
	std::stable_sort(reses_.begin(), reses_.end());
	LocalMatchRes res = reses_[0];

	Mat img0;
	cvtColor(imgPry[pyr_n], img0, COLOR_GRAY2BGR);
	DrawLocDeforRes(img0, model->models[pyr_n], res, res.flags);

	reses_.resize(5);
	for (int pyr_num_ = pyr_n - 1;  pyr_num_ > -1; --pyr_num_)
	{	
		for (size_t i = 0; i < 5; ++i)
			reses_[i].score = 0.0f;
		vector<int> transLen_down;
		vector<bool> contourFlags;
		vector<Point> gravitys;
		GetTranslation(model->models[pyr_num_].segContMapIdx, res, transLen_down, contourFlags, gravitys);
		angleStep_ /= 2;
		int center[4] = { 2 * res.c_x - 5, 2 * res.c_y - 5, 2 * res.c_x + 5, 2 * res.c_y + 5 };
#pragma omp parallel for
		for (int i = -2; i <= 2; ++i)
		{
			reses_[i + 2].translates.resize(model->models[pyr_num_].segContIdx.size());
			reses_[i + 2].flags.resize(model->models[pyr_num_].segContIdx.size());
			//reses_[i + 2].gravitys.resize(model->models[pyr_num_].segContIdx.size());
			double angle = res.angle + i * angleStep_;
			vector<Point2f> r_coord, r_grad;
			RotateCoordGrad(model->models[pyr_num_].coord, model->models[pyr_num_].grad, r_coord, r_grad, angle);
			vector<Point2f> contNormals;
			RotContourVec(model->models[pyr_num_].normals_, contNormals, angle);
			Match(imgPry[pyr_num_], r_coord, r_grad, model->models[pyr_num_].segContIdx, contNormals, center, model->minScore, angle, transLen_down, contourFlags, /*gravitys,*/ reses_[i + 2]);
		}
		std::stable_sort(reses_.begin(), reses_.end());
		res = reses_[0];
	}
	Mat img;
	cvtColor(imgPry[0], img, COLOR_GRAY2BGR);
	DrawLocDeforRes(img, model->models[0], res, res.flags);
	return;
}
//========================================================================================

void LocalDeforModelTest()
{
	string imgPath = "C:/Users/Administrator/Desktop/testimage/halconimg_l1.tif";
	Mat modImg = imread(imgPath, 0);
	LocalDeforModel *model = new LocalDeforModel;

	SPAPLEMODELINFO shapeModelInfo;
	shapeModelInfo.extContouMode = 2;
	shapeModelInfo.pyrNumber = 4;
	shapeModelInfo.lowVal = 200;
	shapeModelInfo.highVal = 300;
	shapeModelInfo.step = 3;
	shapeModelInfo.angStep = 1;
	shapeModelInfo.startAng = -180;
	shapeModelInfo.endAng = 180;
	shapeModelInfo.minScale = 0.6;
	shapeModelInfo.maxScale = 0.6;
	CreateLocalDeforableModel(modImg, model, shapeModelInfo);

	//Mat resizeImg;
	//cv::resize(modImg, resizeImg, cv::Size(modImg.cols * 1.2, modImg.rows * 0.8));

	//Mat rotMat = getRotationMatrix2D(Point2f(resizeImg.cols * 0.35, resizeImg.rows * 0.55), 54, 1);
	//Mat rotImg;
	//cv::warpAffine(resizeImg, rotImg, rotMat, resizeImg.size(), INTER_LINEAR, cv::BORDER_REPLICATE);

	Mat testImg = imread("C:/Users/Administrator/Desktop/testimage/halconimg_l2.tif", 0);
	LocalDeforModelMatch(testImg, model);
}