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

	vector<Mat> imgPry;
	get_pyr_image(modImg, imgPry, shapeModelInfo.pyrNumber);

	for (int i = 0; i < imgPry.size(); i++)
	{
		vector<Point> v_Coord_;
		ExtractModelContour(imgPry[i], shapeModelInfo, v_Coord_);
		if (v_Coord_.size() < 1)
			break;
		vector<Point2f> v_Coord, v_Grad;
		vector<float> v_Amplitude;
		ExtractModelInfo(imgPry[i], v_Coord_, v_Coord, v_Grad, v_Amplitude);	

		vector<Point2f> v_RedCoord, v_RedGrad;
		ReduceMatchPoint(v_Coord, v_Grad, v_Amplitude, v_RedCoord, v_RedGrad, shapeModelInfo.step);
		//聚类
		LocalDeforModelInfo localDeforModelInfo;
		GetKNearestPoint(v_RedCoord, v_RedGrad, localDeforModelInfo);
		//对模板打标签
		LabelContour(localDeforModelInfo);
		//计算重心
		ComputeSegContGravity(localDeforModelInfo);
		model->models.push_back(localDeforModelInfo);
		Mat colorImg;
		cvtColor(imgPry[i], colorImg, COLOR_GRAY2BGR);
		//for (int i = 0; i < localDeforModelInfo.segContIdx.size(); ++i)
		//{
		//	draw_contours(colorImg, localDeforModelInfo.coord, localDeforModelInfo.segContIdx[i], localDeforModelInfo.gravity);
		//}
		draw_contours(colorImg, localDeforModelInfo.coord, localDeforModelInfo.gravity);
		model->pyrNum++;
	}
	//轮廓从上层到下层的映射索引
	GetMapIndex(*model);
	//计算自轮廓的方向向量
	ComputeSegContourVec(*model);
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
			const vector<uint>& segCont = models_.segContIdx[j];
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
			models_.normals_[j].z = -(models_.normals_[j].x * line_[2] + models_.normals_[j].y * line_[3]);
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
			float sum_x = 0.0f, sum_y = 0.0f;
			for (size_t k = 0; k < up_.segContIdx[j].size(); ++k)
			{
				sum_x += (up_.coord[up_.segContIdx[j][k]].x);
				sum_y += (up_.coord[up_.segContIdx[j][k]].y);
			}
			gravitys_up[j].x = sum_x / up_.segContIdx[j].size();
			gravitys_up[j].y = sum_y / up_.segContIdx[j].size();
		}

		//knn最近邻搜索
		LocalDeforModelInfo& down_ = localDeforModel.models[i - 1];
		Mat source = cv::Mat(gravitys_up).reshape(1);
		down_.segContMapIdx.resize(down_.segContIdx.size());
		for (size_t j = 0; j < down_.segContIdx.size(); ++j)
		{
			float sum_x = 0.0f, sum_y = 0.0f;
			for (size_t k = 0; k < down_.segContIdx[j].size(); ++k)
			{
				sum_x += (down_.coord[down_.segContIdx[j][k]].x);
				sum_y += (down_.coord[down_.segContIdx[j][k]].y);
			}
			/**KD树knn查询**/
			vector<float> vecQuery(2);//存放查询点
			vecQuery[0] = sum_x / down_.segContIdx[j].size() * 0.5f - 2.0f; //查询点x坐标
			vecQuery[1] = sum_y / down_.segContIdx[j].size() * 0.5f - 2.0f; //查询点y坐标

			cv::flann::KDTreeIndexParams indexParams(2);
			cv::flann::Index kdtree(source, indexParams);

			vector<int> vecIndex(1);//存放返回的点索引
			vector<float> vecDist(1);//存放距离
			cv::flann::SearchParams params(32);//设置knnSearch搜索参数
			kdtree.knnSearch(vecQuery, vecIndex, vecDist, 1, params);
			down_.segContMapIdx[j] = vecIndex[0];
		}
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
		fitLine(fitLinePoint, line_, DIST_L2, 0, 0.1, 0.01);
		normals[i].x = -line_[1];
		normals[i].y = line_[0];
		cv::Point2f ref_p = fitLinePoint[0];
		float cosVal = ref_p.x * normals[i].x + ref_p.y * normals[i].y;
		if (cosVal > 0)
		{
			normals[i].x = -normals[i].x;
			normals[i].y = -normals[i].y;
		}
	}
}
//========================================================================================

//移动轮廓================================================================================
void TranslationContour(const vector<Point2f>& contour, const vector<uint>& contIdx, 
	const Point3f& normal_, vector<Point2f>& tranContour, int transLen)
{
	if (tranContour.size() != contIdx.size())
		tranContour.resize(contIdx.size());
	for (size_t i = 0; i < contIdx.size(); ++i)
	{
		tranContour[i].x = contour[contIdx[i]].x + transLen * normal_.x;
		tranContour[i].y = contour[contIdx[i]].y + transLen * normal_.y;
	}
}
//========================================================================================

//顶层匹配================================================================================
void TopMatch(const Mat &s_x, const Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, 
	const vector<vector<uint>>& segIdx, const vector<Point3f>& normals_, float minScore, float angle, LocalMatchRes& reses)
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
				for (int transLen = -10; transLen <= 10; transLen += 1)
				{
					if (transLen >= normals_[index].z)
						break;
					//计算重心点到轮廓的距离
					float segContScore_t = 0.0f;
					vector<Point2f> tranContour;
					TranslationContour(r_coord, segIdx[index], normals_[index], tranContour, transLen);
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
				reses.score = score;
				reses.c_x = x;
				reses.c_y = y;
				reses.angle = angle;
				for (size_t j = 0; j < segNum; ++j)
				{
					reses.translates[j] = v_TransLen_[j];
				}
				//reses.push_back(matchRes);
			}
		}
	}
}
//========================================================================================

//匹配====================================================================================
void Match(const Mat &image, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<uint>>& segIdx, 
	const vector<Point3f>& normals_, cv::Point center, float minScore, float angle, vector<int>& transLen_down, LocalMatchRes& reses)
{
	int segNum = segIdx.size();
	int maxW = center.x + 5, maxH = center.y + 5;
	float NormGreediness = ((1 - 0.9 * minScore) / (1 - 0.9)) / segNum;
	float anMinScore = 1 - minScore, NormMinScore = minScore / segNum;

	vector<int> v_TransLen_(segNum);
	for (int y = center.y - 5; y < maxH; ++y)
	{
		for (int x = center.x - 5; x < maxW; ++x)
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
					TranslationContour(r_coord, segIdx[index], normals_[index], tranContour, transLen + transLen_down[index]);
					for (int i = 0; i < tranContour.size(); ++i)
					{
						uint idx = segIdx[index][i];
						int cur_x = x + tranContour[i].x;
						int cur_y = y + tranContour[i].y;
						if (cur_x < 2 || cur_y < 2 || cur_x > image.cols || cur_y > image.rows)
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
					reses.translates[j] = v_TransLen_[j] + transLen_down[j];
				}
			}
		}
	}
}
//========================================================================================

//旋转方向向量============================================================================
void RotContourVec(const vector<Point3f>& srcVec, vector<Point3f>& dstVec, float rotAng)
{
	float rotRad = rotAng / 180 * CV_PI;
	float sinVal = sin(rotRad);
	float cosVal = cos(rotRad);
	if (dstVec.size() != srcVec.size())
		dstVec.resize(srcVec.size());
	for (int i = 0; i < dstVec.size(); ++i)
	{
		dstVec[i].x = srcVec[i].x * cosVal - srcVec[i].y * sinVal;
		dstVec[i].y = srcVec[i].y * cosVal + srcVec[i].x * sinVal;
		dstVec[i].z = srcVec[i].z;
	}
}
//========================================================================================

//上层映射到下层==========================================================================
void UpMapToDown(LocalDeforModelInfo& up_, LocalDeforModelInfo& down_, vector<int>& transLen_up, vector<int>& transLen_down)
{
	vector<cv::Mat> sources(up_.segContIdx.size());
	for (size_t i = 0; i < up_.segContIdx.size(); ++i)
	{
		vector<Point2f> v_source(up_.segContIdx[i].size());
		for (size_t j = 0; j < up_.segContIdx[i].size(); ++j)
		{
			v_source[j] =  2.0f * up_.coord[up_.segContIdx[i][j]];
		}
		sources[i] = cv::Mat(v_source).reshape(1);
	}

	transLen_down.resize(down_.segContIdx.size());
	for (size_t i = 0; i < down_.segContIdx.size(); ++i)
	{
		vector<int> index_(up_.segContIdx.size(), 0);
		for (size_t j = 0; j < down_.segContIdx[i].size(); ++j)
		{
			/**KD树knn查询**/
			vector<float> vecQuery(2);//存放查询点
			vecQuery[0] = down_.coord[down_.segContIdx[i][j]].x; //查询点x坐标
			vecQuery[1] = down_.coord[down_.segContIdx[i][j]].y; //查询点y坐标
			int max_idx = 0;
			float min_dist = 1e8f;;
			for (size_t k = 0; k < sources.size(); ++k)
			{
				cv::flann::KDTreeIndexParams indexParams(2);
				cv::flann::Index kdtree(sources[k], indexParams);
				vector<int> vecIndex(1);//存放返回的点索引
				vector<float> vecDist(1);//存放距离
				cv::flann::SearchParams params(32);//设置knnSearch搜索参数
				kdtree.knnSearch(vecQuery, vecIndex, vecDist, 1, params);
				if (min_dist > vecDist[0])
				{
					min_dist = vecDist[0];
					max_idx = vecIndex[0];
				}
			}
			index_[max_idx]++;
		}
		int max_idx = 0;
		for (size_t j = 0; j < index_.size(); ++j)
		{
			max_idx = max_idx > index_[j] ? max_idx : index_[j];
		}
		transLen_down[i] = 2 * transLen_up[max_idx];
	}
}
//========================================================================================

//获取平移量==============================================================================
void GetTranslation(vector<int>& segContMapIdx, vector<int>& transLen_up, vector<int>& transLen_down)
{
	if (transLen_down.size() != segContMapIdx.size())
		transLen_down.resize(segContMapIdx.size());
	for (size_t i = 0; i < segContMapIdx.size(); ++i)
	{
		transLen_down[i] = 2 * transLen_up[segContMapIdx[i]];
	}
}
//========================================================================================

//匹配====================================================================================
void LocalDeforModelMatch(Mat &srcImg, LocalDeforModel* &model)
{
	const int pyr_n = model->pyrNum - 1;
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
	vector<vector<MatchRes>> mulMatchRes(angNum);

	vector<LocalMatchRes> reses_(angNum);
	//计算轮廓的法向量用于后面的平移
#pragma omp parallel for
	for (int i = 0; i < angNum; ++i)
	{
		reses_[i].translates.resize(model->models[pyr_n].segContIdx.size());
		float angle = model->startAng + i * angleStep_;
		vector<Point2f> r_coord, r_grad;
		RotateCoordGrad(model->models[pyr_n].coord, model->models[pyr_n].grad, r_coord, r_grad, angle);
		vector<Point3f> normals_;
		RotContourVec(model->models[pyr_n].normals_, normals_, angle);
		TopMatch(sobel_x, sobel_y, r_coord, r_grad, model->models[pyr_n].segContIdx, normals_, model->minScore, angle, reses_[i]);
	}
	std::stable_sort(reses_.begin(), reses_.end());
	LocalMatchRes res = reses_[0];

	Mat img1;
	cvtColor(imgPry[pyr_n], img1, COLOR_GRAY2BGR);
	vector<Point2f> r_coord, r_grad;
	RotateCoordGrad(model->models[pyr_n].coord, model->models[pyr_n].grad, r_coord, r_grad, res.angle);
	vector<Point2f> r_t_coord(r_coord.size());
	vector<Point3f> normals_;
	RotContourVec(model->models[pyr_n].normals_, normals_, res.angle);
	for (int i = 0; i < model->models[pyr_n].segContIdx.size(); ++i)
	{
		vector<Point2f> tranContour;
		TranslationContour(r_coord, model->models[pyr_n].segContIdx[i], normals_[i], tranContour, res.translates[i]);
		for (int j = 0; j < tranContour.size(); ++j)
		{
			r_t_coord[model->models[pyr_n].segContIdx[i][j]] = tranContour[j];
		}
	}
	draw_contours(img1, r_t_coord, Point2f(res.c_x, res.c_y));

	for (int pyr_num_ = pyr_n - 1;  pyr_num_ > -1; --pyr_num_)
	{	
		reses_.clear();
		reses_.resize(5);
		vector<int> transLen_down;
		GetTranslation(model->models[pyr_num_].segContMapIdx, res.translates, transLen_down);
		angleStep_ /= 2;
		cv::Point center(2.0 * res.c_x, 2.0 * res.c_y);
		float start_angle = res.angle;
		res.reset();
#pragma omp parallel for
		for (int i = -2; i <= 2; ++i)
		{
			reses_[i + 2].translates.resize(model->models[pyr_num_].segContIdx.size());
			float angle = start_angle + i * angleStep_;
			vector<Point2f> r_coord, r_grad;
			RotateCoordGrad(model->models[pyr_num_].coord, model->models[pyr_num_].grad, r_coord, r_grad, angle);
			vector<Point3f> contNormals;
			RotContourVec(model->models[pyr_num_].normals_, contNormals, angle);
			Match(imgPry[pyr_num_], r_coord, r_grad, model->models[pyr_num_].segContIdx, contNormals, center, model->minScore, angle, transLen_down, reses_[i + 2]);
		}
		std::stable_sort(reses_.begin(), reses_.end());
		res = reses_[0];
		Mat img1_other;
		cvtColor(imgPry[pyr_num_], img1_other, COLOR_GRAY2BGR);
		vector<Point2f> r_coord_other, r_grad_ohter;
		RotateCoordGrad(model->models[pyr_num_].coord, model->models[pyr_num_].grad, r_coord_other, r_grad_ohter, res.angle);
		vector<Point3f> lines_0;
		RotContourVec(model->models[pyr_num_].normals_, lines_0, res.angle);
		vector<Point2f> r_t_coord_ohter(r_coord_other.size());
		for (int i = 0; i < model->models[pyr_num_].segContIdx.size(); ++i)
		{
			vector<Point2f> tranContour;
			TranslationContour(r_coord_other, model->models[pyr_num_].segContIdx[i], lines_0[i], tranContour, res.translates[i]);
			for (int j = 0; j < tranContour.size(); ++j)
			{
				r_t_coord_ohter[model->models[pyr_num_].segContIdx[i][j]] = tranContour[j];
			}
		}
		draw_contours(img1_other, r_t_coord_ohter, Point2f(res.c_x, res.c_y));
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
	shapeModelInfo.pyrNumber = 6;
	shapeModelInfo.lowVal = 30;
	shapeModelInfo.highVal = 200;
	shapeModelInfo.step = 1;
	shapeModelInfo.angStep = 1;
	shapeModelInfo.startAng = -180;
	shapeModelInfo.endAng = 180;
	CreateLocalDeforableModel(modImg, model, shapeModelInfo);

	Mat resizeImg;
	cv::resize(modImg, resizeImg, cv::Size(modImg.cols * 0.6, modImg.rows * 0.6));

	Mat rotMat = getRotationMatrix2D(Point2f(resizeImg.cols * 0.45, resizeImg.rows * 0.45), 53, 1);
	Mat rotImg;
	cv::warpAffine(resizeImg, rotImg, rotMat, resizeImg.size());

	Mat testImg = imread("Test1.bmp", 0);
	LocalDeforModelMatch(rotImg, model);
}