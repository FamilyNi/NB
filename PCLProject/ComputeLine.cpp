#include "ComputeLine.h"
#include "ContourOpr.h"
#include "ContourOpr.cpp"

//两点计算直线==================================================================================
template <typename T>
void Img_TwoPtsComputeLine(T& pt1, T& pt2, cv::Vec3d& line)
{
	double vec_x = pt2.x - pt1.x;
	double vec_y = pt2.y - pt1.y;
	double norm_ = 1 / std::sqrt(vec_x * vec_x + vec_y * vec_y);
	line[0] = -vec_y * norm_;
	line[1] = vec_x * norm_;
	line[2] = -(line[0] * pt1.x + line[1] * pt1.y);
}
//==============================================================================================

//随机一致采样算法计算直线======================================================================
template <typename T1, typename T2>
void Img_RANSACComputeLine(vector<T1>& pts, T2& line, vector<T1>& inlinerPts, double thres)
{
	if (pts.size() < 2)
		return;
	int best_model_p = 0;
	double P = 0.99;  //模型存在的概率
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	for (int i = 0; i < maxEpo; ++i)
	{
		int effetPoints = 0;
		//随机选择两个点计算直线---注意：这里可能需要特殊处理防止点相同
		int index_1 = rand() % size;
		int index_2 = rand() % size;
		T2 line_;
		Img_TwoPtsComputeLine(pts[index_1], pts[index_2], line_);

		//计算局内点的个数
		for (int j = 0; j < size; ++j)
		{
			float dist = abs(line_[0] * pts[j].x + line_[1] * pts[j].y + line_[2]);
			effetPoints += dist < thres ? 1 : 0;
		}
		//获取最优模型，并根据概率修改迭代次数
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			line = line_;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
		if (best_model_p > 0.5 * size)
		{
			line = line_;
			break;
		}
	}

	//提取局内点
	if (inlinerPts.size() != 0)
		inlinerPts.resize(0);
	inlinerPts.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		if (abs(line[0] * pts[i].x + line[1] * pts[i].y + line[2]) < thres)
			inlinerPts.push_back(pts[i]);
	}
}
//==============================================================================================

//最小二乘法拟合直线============================================================================
template <typename T1, typename T2>
void Img_OLSFitLine(vector<T1>& pts, vector<double>& weights, T2& line)
{
	double w_sum = 0.0;
	double w_x_sum = 0.0;
	double w_y_sum = 0.0;
	for (int i = 0; i < weights.size(); ++i)
	{
		w_sum += weights[i];
		w_x_sum += weights[i] * pts[i].x;
		w_y_sum += weights[i] * pts[i].y;
	}
	w_sum = 1.0 / std::max(w_sum, EPS);
	double w_x_mean = w_x_sum * w_sum;
	double w_y_mean = w_y_sum * w_sum;
	Mat A(2, 2, CV_64FC1, cv::Scalar(0));
	double* pA = A.ptr<double>(0);
	for (int i = 0; i < pts.size(); ++i)
	{
		double x_ = pts[i].x - w_x_mean;
		double y_ = pts[i].y - w_y_mean;
		pA[0] += weights[i] * x_ * x_;
		pA[1] += weights[i] * x_ * y_;
		pA[3] += weights[i] * y_ * y_;
	}
	pA[2] = pA[1];
	Mat eigenVal, eigenVec;
	eigenNonSymmetric(A, eigenVal, eigenVec);
	double* pEigenVec = eigenVec.ptr<double>(1);
	line[0] = pEigenVec[0];
	line[1] = pEigenVec[1];
	line[2] = -(w_x_mean * line[0] + w_y_mean * line[1]);
}
//==============================================================================================

//Huber计算权重=================================================================================
template <typename T1, typename T2>
void Img_HuberLineWeights(vector<T1>& pts, T2& line, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = abs(pts[i].x * line[0] + pts[i].y * line[1] + line[2]);
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
template <typename T1, typename T2>
void Img_TukeyLineWeights(vector<T1>& pts, T2& line, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = abs(pts[i].x * line[0] + pts[i].y * line[1] + line[2]);
		dists[i] = distance;
	}
	vector<double> disttanceSort = dists;
	sort(disttanceSort.begin(), disttanceSort.end());
	double tao = disttanceSort[(disttanceSort.size() - 1) / 2] / 0.6745 * 2;

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

//直线拟合======================================================================================
template <typename T1, typename T2>
void Img_FitLine(vector<T1>& pts, T2& line, int k, NB_MODEL_FIT_METHOD method)
{
	vector<double> weights(pts.size(), 1);
	Img_OLSFitLine(pts, weights, line);
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
				Img_HuberLineWeights(pts, line, weights);
				break;
			case TUKEY_FIT:
				Img_TukeyLineWeights(pts, line, weights);
				break;
			default:
				break;
			}
			Img_OLSFitLine(pts, weights, line);
		}
	}
}
//==============================================================================================

void LineTest()
{
	string imgPath = "C:/Users/Administrator/Desktop/testimage/7.bmp";
	cv::Mat srcImg = cv::imread(imgPath, 0);
	cv::Mat binImg;
	cv::threshold(srcImg, binImg, 10, 255, ThresholdTypes::THRESH_BINARY_INV);
	vector<vector<cv::Point>> contours;
	cv::findContours(binImg, contours, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_NONE);

	Mat colorImg;
	cv::cvtColor(srcImg, colorImg, cv::COLOR_GRAY2BGR);
	//for (int i = 0; i < contours.size(); ++i)
	//{
	//	cv::drawContours(colorImg, contours, i, cv::Scalar(0, 255, 0), 2);
	//}

	vector<cv::Point2f> pts(contours.size());
	for (int i = 0; i < contours.size(); ++i)
	{
		int len = contours[i].size();
		if (len == 0)
			return;
		float sum_x = 0.0f, sum_y = 0.0f;
		for (int j = 0; j < len; ++j)
		{
			sum_x += contours[i][j].x;
			sum_y += contours[i][j].y;
		}
		pts[i].x = sum_x / len;
		pts[i].y = sum_y / len;
	}

	cv::Vec3d line;
	vector<cv::Point2f> inlinerPts;
	//Img_FitLine(pts, line, 15, NB_MODEL_FIT_METHOD::TUKEY_FIT);
	Img_RANSACComputeLine(pts, line, inlinerPts, 0.1);
	cv::Point s_pt, e_pt;
	s_pt.x = 35; s_pt.y = -(line[2] + 35 * line[0]) / line[1];
	e_pt.x = 800; e_pt.y = -(line[2] + 800 * line[0]) / line[1];

	vector<cv::Point2f> smoothContour;
	Img_SmoothContour(pts, smoothContour, 9, 0.01);
	for (int i = 0; i < pts.size(); ++i)
	{
		cv::line(colorImg, pts[i], pts[i], cv::Scalar(0, 0, 255), 3);

		cv::line(colorImg, smoothContour[i], smoothContour[i], cv::Scalar(0, 255, 0), 3);
		//cv::drawContours(colorImg, lines, i, cv::Scalar(0, 0, 255), 1);
	}
	cv::line(colorImg, s_pt, e_pt, cv::Scalar(0, 255, 0), 3);
}