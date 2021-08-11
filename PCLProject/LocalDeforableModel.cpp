#include "LocalDeforableModel.h"
#include "ContourOpr.h"
#include "opencv2/flann.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/flann.hpp>

//创建模板================================================================================
void CreateLocalDeforableModel(Mat &modImg, LocalDeforModel* &model, SPAPLEMODELINFO &shapeModelInfo)
{
	if (model == nullptr)
		return;

	model->angStep = shapeModelInfo.angStep;
	model->startAng = shapeModelInfo.startAng;
	model->endAng = shapeModelInfo.endAng;
	model->greediness = 0.9;
	model->minScore = 0.5;

	vector<Mat> imgPry;
	get_pyr_image(modImg, imgPry, shapeModelInfo.pyrNumber);

	vector<Point2f> v_Gravity;
	vector<vector<float>> vv_GradX, vv_GradY;
	
	for (int i = 0; i < imgPry.size(); i++)
	{
		vector<Point> v_Coord;
		ExtractModelContour(imgPry[i], shapeModelInfo, v_Coord);
		if (v_Coord.size() < 1)
			break;
		vector<Point2f> v_Coord_, v_Grad_;
		vector<float> v_Amplitude;
		ExtractModelInfo(imgPry[i], v_Coord, v_Coord_, v_Grad_, v_Amplitude);		
		//聚类
		LocalDeforModelInfo localDeforModelInfo;
		GetKNearestPoint(v_Coord_, v_Grad_, localDeforModelInfo);
		//对模板打标签
		LabelContour(localDeforModelInfo);
		//计算重心
		ComputeSegContGravity(localDeforModelInfo);
		model->v_LocalDeforModel.push_back(localDeforModelInfo);
		Mat colorImg;
		cvtColor(imgPry[i], colorImg, COLOR_GRAY2BGR);
		//for (int i = 0; i < localDeforModelInfo.segContIdx.size(); ++i)
		//{
		//	draw_contours(colorImg, localDeforModelInfo.coord, localDeforModelInfo.segContIdx[i], localDeforModelInfo.gravity);
		//}
		draw_contours(colorImg, localDeforModelInfo.coord, localDeforModelInfo.gravity);
		model->pyrNumber++;
	}
}
//========================================================================================

//获取模板梯度============================================================================
void ExtractModelInfo(Mat &srcImg, vector<Point> &contour, vector<Point2f> &v_Coord, vector<Point2f> &v_Grad, vector<float> &v_Amplitude)
{
	Mat sobel_x, sobel_y;
	Sobel(srcImg, sobel_x, CV_32FC1, 1, 0, 3);
	Sobel(srcImg, sobel_y, CV_32FC1, 0, 1, 3);
	v_Coord.reserve(contour.size());
	v_Grad.reserve(contour.size());
	v_Amplitude.reserve(contour.size());
	for (size_t i = 0; i < contour.size(); ++i)
	{
		float grad_x = sobel_x.at<float>(contour[i]);
		float grad_y = sobel_y.at<float>(contour[i]);
		if (abs(grad_x) > 1e-8 || abs(grad_y) > 1e-8)
		{
			
			v_Coord.push_back((Point2f)contour[i]);
			float norm = sqrt(grad_x * grad_x + grad_y * grad_y);
			v_Amplitude.push_back(norm);
			v_Grad.push_back(Point2f(grad_x / norm, grad_y / norm));
		}
	}
}
//========================================================================================

//模板点聚类==============================================================================
void GetKNearestPoint(vector<Point2f> &contours, vector<Point2f> &grads, LocalDeforModelInfo &localDeforModelInfo)
{
	localDeforModelInfo.coord = contours;
	localDeforModelInfo.grad = grads;

	Mat centers, labels;
	int clusterCount = localDeforModelInfo.coord.size() / 8;
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

//求取每个子轮廓的重心====================================================================
void ComputeSegContGravity(LocalDeforModelInfo &localDeforModelInfo)
{
	size_t len = localDeforModelInfo.coord.size();
	GetContourGravity(localDeforModelInfo.coord, localDeforModelInfo.gravity);
	for (size_t i = 0; i < len; ++i)
	{
		localDeforModelInfo.coord[i].x -= localDeforModelInfo.gravity.x;
		localDeforModelInfo.coord[i].y -= localDeforModelInfo.gravity.y;
	}
}
//========================================================================================

//对聚类后的模板打标签====================================================================
void LabelContour(LocalDeforModelInfo& localDeforModelInfo)
{
	localDeforModelInfo.label.resize(localDeforModelInfo.segContIdx.size());
	for (size_t i = 0; i < localDeforModelInfo.segContIdx.size(); ++i)
	{
		//计算轮廓标签
		vector<uint>& index = localDeforModelInfo.segContIdx[i];
		size_t len = index.size();
		float sum_gradx = 0.0f;
		float sum_grady = 0.0f;
		for (size_t j = 0; j < len; ++j)
		{
			sum_gradx += localDeforModelInfo.grad[index[j]].x;
			sum_grady += localDeforModelInfo.grad[index[j]].y;
		}
		sum_gradx /= (float)len;
		sum_grady /= (float)len;
		localDeforModelInfo.label[i] = std::sqrt(sum_gradx * sum_gradx + sum_grady * sum_grady);
	}
}
//========================================================================================

//计算每个子轮廓的法向量==================================================================
void ComputeContourNormal(const vector<Point2f>& contour, const vector<vector<uint>>& segContIdx, vector<Point2f>& normals)
{
	size_t segContNum = segContIdx.size();
	if (normals.size() != segContNum)
		normals.resize(segContNum);
	for (size_t i = 0; i < segContNum; ++i)
	{
		const vector<uint>& segCont = segContIdx[i];
		size_t p_number = segCont.size();
		vector<Point2f> fitLinePoint(p_number);
		for (size_t i = 0; i < p_number; ++i)
		{
			fitLinePoint[i] = contour[segCont[i]];
		}
		Vec4f line_;
		fitLine(fitLinePoint, line_, DIST_L1, 0, 0.1, 0.01);
		normals[i].x = -line_[1];
		normals[i].y = line_[0];
		cv::Point2f ref_p = fitLinePoint[0];
		float cosVal = ref_p.x * normals[i].x + ref_p.y * normals[i].y;
		if (cosVal < 0)
		{
			normals[i].x = -normals[i].x;
			normals[i].y = -normals[i].y;
		}
	}
}
//========================================================================================

//移动轮廓================================================================================
void TranslationContour(const vector<Point2f>& contour, const vector<uint>& contIdx, 
	const Point2f& normals, vector<Point2f>& tranContour, int transLen)
{
	if (tranContour.size() != contIdx.size())
		tranContour.resize(contIdx.size());
	for (size_t i = 0; i < contIdx.size(); ++i)
	{
		tranContour[i].x = contour[contIdx[i]].x + transLen * normals.x;
		tranContour[i].y = contour[contIdx[i]].y + transLen * normals.y;
	}
}
//========================================================================================

//顶层匹配================================================================================
void TopMatch(const Mat &s_x, const Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<uint>>& segIdx,
	const vector<Point2f>& contNormals, float minScore, float angle, MatchRes& reses, vector<int>& v_TransLen)
{
	int segNum = segIdx.size();
	int maxW = s_x.cols - 2, maxH = s_x.rows - 2;
	float NormGreediness = ((1 - 0.9 * minScore) / (1 - 0.9)) / segNum;
	float anMinScore = 1 - minScore, NormMinScore = minScore / segNum;

	vector<int> v_TransLen_(segNum);
	for (int y = 2; y < maxH; ++y)
	{
		for (int x = 2; x < maxW; ++x)
		{
			float partial_score = 0.0f, score = 0.0f;
			for (int index = 0; index < segNum; index++)
			{
				int sum_i = index + 1;
				float segContScore = 0.0f;
				//平移部分==================
				for (int transLen = -5; transLen <= 5; transLen += 1)
				{
					float segContScore_t = 0.0f;
					vector<Point2f> tranContour;
					TranslationContour(r_coord, segIdx[index], contNormals[index], tranContour, transLen);
					for (int i = 0; i < tranContour.size(); ++i)
					{
						uint idx = segIdx[index][i];
						int cur_x = x + tranContour[i].x;
						int cur_y = y + tranContour[i].y;
						if (cur_x < 2 || cur_y < 2 || cur_x > maxW || cur_y > maxH)
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
						v_TransLen_[index] = transLen;
					}
				}
				//===============================

				//for (int i = 0; i < segIdx[index].size(); ++i)
				//{
				//	uint idx = segIdx[index][i];
				//	int cur_x = x + r_coord[idx].x;
				//	int cur_y = y + r_coord[idx].y;
				//	if (cur_x < 2 || cur_y < 2 || cur_x > maxW || cur_y > maxH)
				//		continue;
				//	short gx = s_x.at<short>(cur_y, cur_x);
				//	short gy = s_y.at<short>(cur_y, cur_x);
				//	if (abs(gx) > 0 || abs(gy) > 0)
				//	{
				//		float grad_x = 0.0f, grad_y = 0.0f;
				//		NormalGrad((int)gx, (int)gy, grad_x, grad_y);
				//		segContScore += abs(grad_x * r_grad[idx].x + grad_y * r_grad[idx].y);
				//	}
				//}
				partial_score += segContScore / segIdx[index].size();
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
					v_TransLen[j] = v_TransLen_[j];
				}
				//reses.push_back(matchRes);
			}
		}
	}
}
//========================================================================================

//匹配====================================================================================
void Match(const Mat &s_x, const Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<uint>>& segIdx,
	const vector<Point2f>& contNormals, cv::Point center, float minScore, float angle, MatchRes& reses, vector<int>& transLen_down, vector<int>& out_transLen)
{
	int segNum = segIdx.size();
	int maxW = center.x + 10, maxH = center.y + 10;
	float NormGreediness = ((1 - 0.9 * minScore) / (1 - 0.9)) / segNum;
	float anMinScore = 1 - minScore, NormMinScore = minScore / segNum;

	vector<int> v_TransLen_(segNum);
	for (int y = center.y - 10; y < maxH; ++y)
	{
		for (int x = center.x - 10; x < maxW; ++x)
		{
			float partial_score = 0.0f, score = 0.0f;
			for (int index = 0; index < segNum; index++)
			{
				int sum_i = index + 1;
				float segContScore = 0.0f;
				//平移部分==================
				for (int transLen = -2; transLen <= 2; transLen += 1)
				{
					float segContScore_t = 0.0f;
					vector<Point2f> tranContour;
					TranslationContour(r_coord, segIdx[index], contNormals[index], tranContour, transLen + transLen_down[index]);
					for (int i = 0; i < tranContour.size(); ++i)
					{
						uint idx = segIdx[index][i];
						int cur_x = x + tranContour[i].x;
						int cur_y = y + tranContour[i].y;
						if (cur_x < 2 || cur_y < 2 || cur_x > s_x.cols || cur_y > s_x.rows)
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
						v_TransLen_[index] = transLen + transLen_down[index];
					}
				}
				//===============================
				//for (int i = 0; i < segIdx[index].size(); ++i)
				//{
				//	uint idx = segIdx[index][i];
				//	int cur_x = x + r_coord[idx].x;
				//	int cur_y = y + r_coord[idx].y;
				//	if (cur_x < 2 || cur_y < 2 || cur_x > maxW || cur_y > maxH)
				//		continue;
				//	short gx = s_x.at<short>(cur_y, cur_x);
				//	short gy = s_y.at<short>(cur_y, cur_x);
				//	if (abs(gx) > 0 || abs(gy) > 0)
				//	{
				//		float grad_x = 0.0f, grad_y = 0.0f;
				//		NormalGrad((int)gx, (int)gy, grad_x, grad_y);
				//		segContScore += abs(grad_x * r_grad[idx].x + grad_y * r_grad[idx].y);
				//	}
				//}
				partial_score += segContScore / segIdx[index].size();
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
					out_transLen[j] = v_TransLen_[j];
				}
			}
		}
	}
}
//========================================================================================

//根据中心点位置索引======================================================================
void GetIndex(LocalDeforModelInfo& up_, LocalDeforModelInfo& down_, vector<int>& transLen_up, vector<int>& transLen_down)
{
	if (transLen_up.size() != up_.segContIdx.size())
		return;
	vector<Point2f> gravitys(transLen_up.size());
	for (int i = 0; i < transLen_up.size(); ++i)
	{
		float sum_x = 0.0f, sum_y = 0.0f;
		for (int j = 0; j < up_.segContIdx[i].size(); ++j)
		{
			sum_x += (up_.coord[up_.segContIdx[i][j]].x);
			sum_y += (up_.coord[up_.segContIdx[i][j]].y);
		}
		gravitys[i].x = sum_x / up_.segContIdx[i].size();
		gravitys[i].y = sum_y / up_.segContIdx[i].size();
	}
	int cont_num = down_.segContIdx.size();
	transLen_down.resize(cont_num);

	Mat source = cv::Mat(gravitys).reshape(1);
	for (int i = 0; i < cont_num; ++i)
	{
		float sum_x = 0.0f, sum_y = 0.0f;
		for (int j = 0; j < down_.segContIdx[i].size(); ++j)
		{
			sum_x += (down_.coord[down_.segContIdx[i][j]].x);
			sum_y += (down_.coord[down_.segContIdx[i][j]].y);
		}
		/**KD树knn查询**/
		vector<float> vecQuery(2);//存放 查询点 的容器（本例都是vector类型）
		vecQuery[0] = 0.5 * sum_x / down_.segContIdx[i].size(); //查询点x坐标
		vecQuery[1] = 0.5 * sum_y / down_.segContIdx[i].size(); //查询点y坐标

		//Point2f gravity_(sum_x / down_.segContIdx[i].size(), sum_y / down_.segContIdx[i].size());

		cv::flann::KDTreeIndexParams indexParams(2);
		cv::flann::Index kdtree(source, indexParams); //此部分建立kd-tree索引同上例，故不做详细叙述

		/**预设knnSearch所需参数及容器**/
		vector<int> vecIndex(1);//存放返回的点索引
		vector<float> vecDist(1);//存放距离
		cv::flann::SearchParams params(32);//设置knnSearch搜索参数
		kdtree.knnSearch(vecQuery, vecIndex, vecDist, 1, params);
		transLen_down[i] =  2 * transLen_up[vecIndex[0]];
	}
}
//========================================================================================

//匹配====================================================================================
void LocalDeforModelMatch(Mat &srcImg, LocalDeforModel* &model)
{
	const int pyr_n = model->pyrNumber - 1;
	vector<Mat> imgPry;
	get_pyr_image(srcImg, imgPry, pyr_n + 1);
	double t3 = getTickCount();
	float angStep = model->angStep > 1 ? model->angStep : 1;
	float angleStep_ = angStep * pow(2, pyr_n);

	int angNum = (model->endAng - model->startAng) / angleStep_ + 1;
	//顶层匹配
	Mat sobel_x, sobel_y;
	Sobel(imgPry[pyr_n], sobel_x, CV_16SC1, 1, 0, 3);
	Sobel(imgPry[pyr_n], sobel_y, CV_16SC1, 0, 1, 3);
	vector<Point2f>& top_coord = model->v_LocalDeforModel[pyr_n].coord;
	vector<Point2f>& top_grad = model->v_LocalDeforModel[pyr_n].grad;
	vector<vector<uint>>& segContIdx = model->v_LocalDeforModel[pyr_n].segContIdx;
	vector<vector<MatchRes>> mulMatchRes(angNum);
	vector<int> v_TransLen(model->v_LocalDeforModel[pyr_n].segContIdx.size());
	//计算轮廓的法向量用于后面的平移
	vector<Point2f> contNormals;
	ComputeContourNormal(top_coord, segContIdx, contNormals);
//#pragma omp parallel for
	MatchRes reses;
	for (int i = 0; i < angNum; ++i)
	{
		float angle = model->startAng + i * angleStep_;
		vector<Point2f> r_coord, r_grad;
		RotateCoordGrad(top_coord, top_grad, r_coord, r_grad, angle);
		TopMatch(sobel_x, sobel_y, r_coord, r_grad, segContIdx, contNormals, model->minScore, angle, reses, v_TransLen);
		//mulMatchRes[i] = reses;
	}

	Mat img1;
	cvtColor(imgPry[pyr_n], img1, COLOR_GRAY2BGR);
	vector<Point2f> r_coord, r_grad;
	RotateCoordGrad(top_coord, top_grad, r_coord, r_grad, reses.angle);
	vector<Point2f> r_t_coord(r_coord.size());
	for (int i = 0; i < segContIdx.size(); ++i)
	{
		vector<Point2f> tranContour;
		TranslationContour(r_coord, segContIdx[i], contNormals[i], tranContour, v_TransLen[i]);
		for (int j = 0; j < tranContour.size(); ++j)
		{
			r_t_coord[segContIdx[i][j]] = tranContour[j];
		}
	}
	draw_contours(img1, r_t_coord, Point2f(reses.c_x, reses.c_y));

	vector<int> out_transLen(0);
	out_transLen.resize(v_TransLen.size());
	for (int i = 0; i < v_TransLen.size(); ++i)
	{
		out_transLen[i] = v_TransLen[i];
	}
	for (int pyr_num_ = pyr_n - 1;  pyr_num_ > -1; --pyr_num_)
	{		
		vector<int> transLen_down;
		GetIndex(model->v_LocalDeforModel[pyr_num_ +1], model->v_LocalDeforModel[pyr_num_], out_transLen, transLen_down);
		Mat s_x_other, s_y_ohter;
		Sobel(imgPry[pyr_num_], s_x_other, CV_16SC1, 1, 0, 3);
		Sobel(imgPry[pyr_num_], s_y_ohter, CV_16SC1, 0, 1, 3);
		angleStep_ /= 2;
		vector<Point2f>& other_coord = model->v_LocalDeforModel[pyr_num_].coord;
		vector<Point2f>& other_grad = model->v_LocalDeforModel[pyr_num_].grad;
		vector<vector<uint>>& segContIdx_ohter = model->v_LocalDeforModel[pyr_num_].segContIdx;
		vector<Point2f> contNormals_other;
		ComputeContourNormal(other_coord, segContIdx_ohter, contNormals_other);
		cv::Point center(2.0 * reses.c_x, 2.0 * reses.c_y);
		float start_angle = reses.angle;
		reses.init();
		out_transLen.resize(model->v_LocalDeforModel[pyr_num_].segContIdx.size());
		for (int i = -2; i <= 2; ++i)
		{
			float angle = start_angle + i * angleStep_;
			vector<Point2f> r_coord, r_grad;
			RotateCoordGrad(other_coord, other_grad, r_coord, r_grad, angle);
			Match(s_x_other, s_y_ohter, r_coord, r_grad, segContIdx_ohter, contNormals_other, center, model->minScore, angle, reses, transLen_down, out_transLen);
		}
		Mat img1_other;
		cvtColor(imgPry[pyr_num_], img1_other, COLOR_GRAY2BGR);
		vector<Point2f> r_coord_other, r_grad_ohter;
		RotateCoordGrad(other_coord, other_grad, r_coord_other, r_grad_ohter, reses.angle);
		vector<Point2f> r_t_coord_ohter(r_coord_other.size());
		for (int i = 0; i < segContIdx_ohter.size(); ++i)
		{
			vector<Point2f> tranContour;
			TranslationContour(r_coord_other, segContIdx_ohter[i], contNormals_other[i], tranContour, out_transLen[i]);
			for (int j = 0; j < tranContour.size(); ++j)
			{
				r_t_coord_ohter[segContIdx_ohter[i][j]] = tranContour[j];
			}
		}
		draw_contours(img1_other, r_t_coord_ohter, Point2f(reses.c_x, reses.c_y));
	}
	return;
}
//========================================================================================

void LocalDeforModelTest()
{
	string imgPath = "LocalModelTest.bmp";
	Mat modImg = imread(imgPath, 0);
	LocalDeforModel *model = new LocalDeforModel;

	SPAPLEMODELINFO shapeModelInfo;
	shapeModelInfo.pyrNumber = 5;
	shapeModelInfo.lowVal = 100;
	shapeModelInfo.highVal = 200;
	shapeModelInfo.step = 1;
	shapeModelInfo.angStep = 1;
	shapeModelInfo.endAng = 30;
	shapeModelInfo.startAng = -30;
	CreateLocalDeforableModel(modImg, model, shapeModelInfo);

	Mat resizeImg;
	cv::resize(modImg, resizeImg, cv::Size(modImg.cols * 0.9, modImg.rows*0.7));

	Mat rotMat = getRotationMatrix2D(Point2f(resizeImg.cols * 0.5, resizeImg.rows*0.5), 163, 1);
	Mat rotImg;
	cv::warpAffine(resizeImg, rotImg, rotMat, resizeImg.size());

	Mat testImg = imread("Test1.bmp", 0);
	LocalDeforModelMatch(rotImg, model);
}