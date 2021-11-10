#include "../../include/FitShapeFile/Img_FitCircle.h"
#include "../../include/FitShapeFile/ComputeModels.h"
#include "../../include/BaseOprFile/MathOpr.h"

//随机一致采样算法计算园========================================================================
void Img_RANSACFitCircle(NB_Array2D pts, Circle2D& circle, vector<int>& inliners, double thres)
{
	if (pts.size() < 3)
		return;
	int best_model_p = 0;
	double P = 0.99;  //模型存在的概率
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	for (int i = 0; i < maxEpo; ++i)
	{
		int effetPoints = 0;
		//随机选择三个点计算园---注意：这里可能需要特殊处理防止点相同
		int index_1 = rand() % size;
		int index_2 = rand() % size;
		int index_3 = rand() % size;
		Circle2D circle_;
		Img_ThreePtsComputeCicle(pts[index_1], pts[index_2], pts[index_3], circle_);
		//计算局内点的个数
		for (int j = 0; j < size; ++j)
		{
			double diff_x = pts[j].x - circle_.x;
			double diff_y = pts[j].y - circle_.y;
			float dist = std::sqrt(diff_x * diff_x + diff_y * diff_y);
			effetPoints += abs(dist - circle_.r) < thres ? 1 : 0;
		}
		//获取最优模型，并根据概率修改迭代次数
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			circle = circle_;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
		if (best_model_p > 0.5 * size)
		{
			circle = circle_;
			break;
		}
	}
	//提取局内点
	if (inliners.size() != 0)
		inliners.resize(0);
	inliners.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		double diff_x = pts[i].x - circle.x;
		double diff_y = pts[i].y - circle.y;
		double dist = std::sqrt(diff_x * diff_x + diff_y * diff_y);
		if (abs(dist - circle.r) < thres)
			inliners.push_back(i);
	}
}
//==============================================================================================

//最小二乘法拟合园==============================================================================
void Img_OLSFitCircle(NB_Array2D pts, vector<double>& weights, Circle2D& circle)
{
	double w_sum = 0.0;
	double w_x_sum = 0.0;
	double w_y_sum = 0.0;
	double w_x2y2_sum = 0.0;
	for (int i = 0; i < pts.size(); ++i)
	{
		w_sum += weights[i];
		w_x_sum += weights[i] * pts[i].x;
		w_y_sum += weights[i] * pts[i].y;
		w_x2y2_sum += weights[i] * (pts[i].x * pts[i].x + pts[i].y * pts[i].y);
	}
	w_sum = 1.0 / std::max(w_sum, EPS);
	double w_x_mean = w_x_sum * w_sum;
	double w_y_mean = w_y_sum * w_sum;
	double w_x2y2_mean = w_x2y2_sum * w_sum;

	Mat A(2, 2, CV_64FC1, cv::Scalar(0));
	Mat B(2, 1, CV_64FC1, cv::Scalar(0));
	double* pA = A.ptr<double>(0);
	double* pB = B.ptr<double>(0);
	for (int i = 0; i < pts.size(); ++i)
	{
		double x_ = pts[i].x - w_x_mean;
		double y_ = pts[i].y - w_y_mean;
		pA[0] += weights[i] * x_ * x_;
		pA[1] += weights[i] * x_ * y_;
		pA[3] += weights[i] * y_ * y_;

		double r_ = pts[i].x * pts[i].x + pts[i].y * pts[i].y - w_x2y2_mean;
		pB[0] -= weights[i] * x_ * r_;
		pB[1] -= weights[i] * y_ * r_;
	}
	pA[2] = pA[1];
	
	Mat C = (A.inv()) * B;
	double* pC = C.ptr<double>(0);
	circle.x = -pC[0] / 2.0;
	circle.y = -pC[1] / 2.0;
	double c = -(pC[0] * w_x_mean + pC[1] * w_y_mean + w_x2y2_mean);
	circle.r = std::sqrt(std::max(circle.x * circle.x + circle.y * circle.y - c, EPS));
}
//==============================================================================================

//huber计算权重=================================================================================
void Img_HuberCircleWeights(NB_Array2D pts, Circle2D& circle, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double diff_x = pts[i].x - circle.x;
		double diff_y = pts[i].y - circle.y;
		double distance = std::sqrt(max(diff_x * diff_x + diff_y * diff_y, EPS));
		distance = abs(distance - circle.r);
		if (distance <= tao)
		{
			weights[i] = 1;
		}
		else
		{
			weights[i] = tao / distance;
		}
	}
}
//==============================================================================================

//Tukey计算权重================================================================================
void Img_TukeyCircleWeights(NB_Array2D pts, Circle2D& circle, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		double diff_x = pts[i].x - circle.x;
		double diff_y = pts[i].y - circle.y;
		double distance = std::sqrt(max(diff_x * diff_x + diff_y * diff_y, EPS));
		distance = abs(distance - circle.r);
		dists[i] = distance;
	}
	//求限制条件tao
	vector<double> disttanceSort = dists;
	sort(disttanceSort.begin(), disttanceSort.end());
	double tao = disttanceSort[(disttanceSort.size() - 1) / 2] / 0.6745 * 2;

	//更新权重
	for (int i = 0; i < dists.size(); ++i)
	{
		if (dists[i] <= tao)
		{
			double d_tao = dists[i] / tao;
			weights[i] = std::pow((1 - d_tao * d_tao), 2);
		}
		else weights[i] = 0;
	}
}
//==============================================================================================

//拟合园========================================================================================
void Img_FitCircle(NB_Array2D pts, Circle2D& circle, int k, NB_MODEL_FIT_METHOD method)
{
	vector<double> weights(pts.size(), 1);

	Img_OLSFitCircle(pts, weights, circle);
	if (method == NB_MODEL_FIT_METHOD::OLS_FIT)
	{
		return;
	}
	else
	{
		for (int i = 0; i < k; ++i)
		{
			switch (method)
			{
			case HUBER_FIT:
				Img_HuberCircleWeights(pts, circle, weights);
				break;
			case TUKEY_FIT:
				Img_TukeyCircleWeights(pts, circle, weights);
				break;
			default:
				break;
			}
			Img_OLSFitCircle(pts, weights, circle);
		}
	}
}
//==============================================================================================

//二维圆拟合测试==============================================================================
void Img_FitCircleTest()
{
	string imgPath = "C:/Users/Administrator/Desktop/testimage/8.bmp";
	cv::Mat srcImg = cv::imread(imgPath, 0);
	cv::Mat binImg;
	cv::threshold(srcImg, binImg, 10, 255, ThresholdTypes::THRESH_BINARY_INV);
	vector<vector<cv::Point>> contours;
	cv::findContours(binImg, contours, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_NONE);

	Mat colorImg;
	cv::cvtColor(srcImg, colorImg, cv::COLOR_GRAY2BGR);

	vector<cv::Point2f> pts(contours.size());
	for (int i = 0; i < contours.size(); ++i)
	{
		int len = contours[i].size();
		float sum_x = 0.0f, sum_y = 0.0f;
		for (int j = 0; j < len; ++j)
		{
			sum_x += contours[i][j].x;
			sum_y += contours[i][j].y;
		}
		pts[i].x = sum_x / len;
		pts[i].y = sum_y / len;
	}

	Circle2D circle;
	vector<int> inliners;
	//Img_RANSACFitCircle(pts, circle, inliners, 0.2);
	Img_FitCircle(pts, circle, 5, NB_MODEL_FIT_METHOD::HUBER_FIT);
	cv::circle(colorImg, cv::Point(circle.x, circle.y), circle.r, cv::Scalar(0, 255, 0), 2);
	for (int i = 0; i < inliners.size(); ++i)
	{
		cv::line(colorImg, pts[inliners[i]], pts[inliners[i]], cv::Scalar(0, 0, 255), 5);
	}
}
//============================================================================================

