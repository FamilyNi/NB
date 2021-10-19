#include "ComputeEllipse.h"
#include "DrawShape.h"

//椭圆方程标准化========================================================================
template <typename T>
void Img_EllipseNormalization(T& ellipse_, T& normEllipse)
{
	normEllipse[2] = -0.5*atan2(ellipse_[1], ellipse_[2] - ellipse_[0]);

	double A = ellipse_[0];
	double B = ellipse_[1] / 2.0;
	double C = ellipse_[2];
	double D = ellipse_[3] / 2.0;
	double E = ellipse_[4] / 2.0;
	double F = ellipse_[5];
	double tmp1 = B * B - A * C;
	double tmp2 = sqrt((A - C)*(A - C) + 4 * B * B);
	double tmp3 = A * E * E + C * D * D + F * B * B - 2.0 * B * D * E - A * C * F;
	
	normEllipse[3] = sqrt(2 * tmp3 / (tmp1 * (tmp2 - A - C)));
	normEllipse[4] = sqrt(2 * tmp3 / (tmp1 * (-tmp2 - A - C)));

	normEllipse[0] = (C * D - B * E) / tmp1;
	normEllipse[1] = (A * E - B * D) / tmp1;

	if (normEllipse[3] < normEllipse[4])
	{
		double temp = normEllipse[3];
		normEllipse[3] = normEllipse[4];
		normEllipse[4] = temp;
		normEllipse[2] += M_PI_2;
	}
}
//======================================================================================

//点到椭圆的距离--超简单版，不建议采用==================================================
template <typename T1, typename T2>
void Img_PtsToEllipseDist(T1& pt, T2& ellipse, double& dist)
{
	double cosVal = std::cos(-ellipse[2]);
	double sinVal = std::sin(-ellipse[2]);
	double x_ = pt.x - ellipse[0];
	double y_ = pt.y - ellipse[1];
	double x = cosVal * x_ - sinVal * y_;
	double y = cosVal * y_ + sinVal * x_;
	double k = y / x;
	double a_2 = ellipse[3] * ellipse[3];
	double b_2 = ellipse[4] * ellipse[4];
	double coeff = a_2 * b_2 / (b_2 + a_2 * k * k);
	double x0 = -std::sqrt(coeff);
	double y0 = k * x0;
	double dist1 = std::sqrt(pow(x - x0, 2) + pow(y - y0, 2));

	x0 = std::sqrt(coeff);
	y0 = k * x0;
	double dist2 = std::sqrt(pow(x - x0, 2) + pow(y - y0, 2));
	dist = dist1 < dist2 ? dist1 : dist2;
}
//======================================================================================

//六点求椭圆============================================================================
template <typename T1, typename T2>
void Img_SixPtsComputeEllipse(vector<T1>& pts, T2& ellipse)
{
	if (pts.size() < 6)
		return;
	Mat coeffMat(cv::Size(6, 6), CV_64FC1, cv::Scalar(1.0f));
	double* pCoeff = coeffMat.ptr<double>();
	for (int i = 0; i < pts.size(); ++i)
	{
		int idx = 6 * i;
		double x = pts[i].x, y = pts[i].y;
		pCoeff[idx] = x * x; pCoeff[idx + 1] = x * y; pCoeff[idx + 2] = y * y;
		pCoeff[idx+3] = x; pCoeff[idx + 4] = y;
	}
	SVD::solveZ(coeffMat, ellipse);
}
//======================================================================================

//随机一致采样算法计算椭圆圆============================================================
template <typename T1, typename T2>
void Img_RANSACComputeCircle(vector<T1>& pts, T2& ellipse, vector<T1>& inlinerPts, double thres)
{
	if (pts.size() < 6)
		return;
	int best_model_p = 0;
	double P = 0.99;  //模型存在的概率
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	vector<T1> pts_(6);
	for (int i = 0; i < maxEpo; ++i)
	{
		int effetPoints = 0;
		//随机选择六个个点计算椭圆---注意：这里可能需要特殊处理防止点相同
		pts_[0] = pts[rand() % size]; pts_[1] = pts[rand() % size];	pts_[2] = pts[rand() % size];
		pts_[3] = pts[rand() % size]; pts_[4] = pts[rand() % size];	pts_[5] = pts[rand() % size];
		T2 ellipse_, normEllipse;
		Img_SixPtsComputeEllipse(pts_, ellipse_);
		Img_EllipseNormalization(ellipse_, normEllipse);
		//计算局内点的个数
		for (int j = 0; j < size; ++j)
		{ 
			double dist = 0.0f;
			Img_PtsToEllipseDist(pts[j], normEllipse, dist);
			effetPoints += dist < thres ? 1 : 0;
		}
		//获取最优模型，并根据概率修改迭代次数
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			ellipse = normEllipse;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
		if (best_model_p > 0.5 * size)
		{
			ellipse = normEllipse;
			break;
		}
	}
	//提取局内点
	if (inlinerPts.size() != 0)
		inlinerPts.resize(0);
	inlinerPts.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		double dist = 0.0f;
		Img_PtsToEllipseDist(pts[i], ellipse, dist);
		if (dist < thres)
			inlinerPts.push_back(pts[i]);
	}
}
//======================================================================================

//最小二乘法拟合椭圆====================================================================
template <typename T1, typename T2>
void Img_OLSFitEllipse(vector<T1>& pts, vector<double>& weights, T2& ellipse)
{
	if (pts.size() < 6)
		return;
	Mat C(3, 3, CV_64FC1, cv::Scalar(0));
	C.at<double>(0, 2) = -2;
	C.at<double>(1,1) = 1;
	C.at<double>(2, 0) = -2;
	int pts_num = pts.size();

	Mat S1(3, 3, CV_64FC1, cv::Scalar(0));
	Mat S2(3, 3, CV_64FC1, cv::Scalar(0));
	Mat S3(3, 3, CV_64FC1, cv::Scalar(0));
	Mat S4(3, 3, CV_64FC1, cv::Scalar(0));
	double* pS1 = S1.ptr<double>(0);
	double* pS2 = S2.ptr<double>(0);
	double* pS4 = S4.ptr<double>(0);
	for (int i = 0; i < pts_num; ++i)
	{
		double w = weights[i];
		double x = pts[i].x, y = pts[i].y;

		double x_2 = x * x, y_2 = y * y, xy = x * y;
		double w_x_2 = w * x_2, w_y_2 = w * y_2, w_xy = w * xy;

		pS1[0] += w_x_2 * x_2; pS1[1] += w_x_2 * xy; pS1[2] += w_x_2 * y_2;
		pS1[4] += w_xy * xy; pS1[5] += w_xy * y_2; pS1[8] += w_y_2 * y_2;

		pS2[0] += w_x_2 * x; pS2[1] += w_x_2 * y; pS2[2] += w_x_2;
		pS2[3] += w_xy * x; pS2[4] += w_xy * y; pS2[5] += w_xy;
		pS2[6] += w_y_2 * x; pS2[7] += w_y_2 * y; pS2[8] += w_y_2;
		
		pS4[0] += w_x_2; pS4[1] += w_xy; pS4[2] += w * x;
		pS4[4] += w_y_2; pS4[5] += w * y;	pS4[8] += w;
	}
	pS1[6] = pS1[2]; pS1[7] = pS1[5]; pS1[3] = pS1[1];
	pS4[6] = pS4[2]; pS4[7] = pS4[5]; pS4[3] = pS4[1];
	S3 = S2.t();
	Mat M = (C.inv()) * (S1 - S2 * (S4.inv()) * S3);

	//判断M是否为对称矩阵
	Mat dstMat;
	cv::compare(M, M.t(), dstMat, 0);

	cv::Mat eigenVal, eigenVec;
	if(cv::countNonZero(dstMat) != 9)
		cv::eigenNonSymmetric(M, eigenVal, eigenVec);
	else
		cv::eigen(M, eigenVal, eigenVec);

	Mat a1(3, 1, CV_64FC1, cv::Scalar(0));
	double* pA1 = a1.ptr<double>(0);
	double* pEigenVec = eigenVec.ptr<double>(2);
	vector<double> A(6);
	for (int i = 0; i < 3; ++i)
	{
		pA1[i] = pEigenVec[i];
		ellipse[i] = pEigenVec[i];
	}

	Mat a2 = (-S4.inv()) * S3 * a1;
	double* pA2 = a2.ptr<double>(0);
	for (int i = 0; i < 3; ++i)
	{
		ellipse[i + 3] = pA2[i];
	}
}
//======================================================================================

//Huber计算权重=========================================================================
template <typename T1, typename T2>
void Img_HuberEllipseWeights(vector<T1>& pts, T2& ellipse, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = 0.0;
		Img_PtsToEllipseDist(pts[i], ellipse, distance);
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
//======================================================================================

//Turkey计算权重========================================================================
template <typename T1, typename T2>
void Img_TukeyEllipseWeights(vector<T1>& pts, T2& ellipse, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		Img_PtsToEllipseDist(pts[i], ellipse, dists[i]);
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
//======================================================================================

//拟合椭圆==============================================================================
template <typename T1, typename T2>
void Img_FitEllipse(vector<T1>& pts, T2& ellipse, int k, NB_MODEL_FIT_METHOD method)
{
	T2 ellipse_;
	vector<double> weights(pts.size(), 1);

	Img_OLSFitEllipse(pts, weights, ellipse_);
	Img_EllipseNormalization(ellipse_, ellipse);
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
				Img_HuberEllipseWeights(pts, ellipse, weights);
				break;
			case TUKEY_FIT:
				Img_TukeyEllipseWeights(pts, ellipse, weights);
				break;
			default:
				break;
			}
			Img_OLSFitEllipse(pts, weights, ellipse_);
			Img_EllipseNormalization(ellipse_, ellipse);

			Mat ellipseImg(cv::Size(1152, 648), CV_8UC1, cv::Scalar(255));
			cv::Point2d center(ellipse[0], ellipse[1]);
			Img_DrawEllipse(ellipseImg, center, ellipse[2], ellipse[3], ellipse[4], 0.2);
		}
	}
}
//======================================================================================


void Img_EllipseTest()
{
	string imgPath = "C:/Users/Administrator/Desktop/testimage/椭圆.bmp";
	cv::Mat srcImg = cv::imread(imgPath, 0);
	cv::Mat binImg;
	cv::threshold(srcImg, binImg, 10, 255, ThresholdTypes::THRESH_BINARY_INV);
	vector<vector<cv::Point>> contours;
	cv::findContours(binImg, contours, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_NONE);

	Mat colorImg;
	cv::cvtColor(srcImg, colorImg, cv::COLOR_GRAY2BGR);

	vector<cv::Point> pts(contours.size());
	for (int i = 0; i < contours.size(); ++i)
	{
		int len = contours[i].size();
		if (len == 0)
			return;
		double sum_x = 0.0, sum_y = 0.0;
		for (int j = 0; j < len; ++j)
		{
			sum_x += contours[i][j].x;
			sum_y += contours[i][j].y;
		}
		pts[i].x = sum_x / len;
		pts[i].y = sum_y / len;
	}

	cv::Vec6d ellipse;
	//vector<cv::Point> inlinerPts;
	//Img_RANSACComputeCircle(pts, ellipse, inlinerPts, 1);
	Img_FitEllipse(pts, ellipse, 5, NB_MODEL_FIT_METHOD::TUKEY_FIT);

	//for (int i = 0; i < inlinerPts.size(); ++i)
	//{
	//	cv::line(colorImg, inlinerPts[i], inlinerPts[i], cv::Scalar(0, 0, 255), 2);

	//}

	cv::RotatedRect rect = cv::fitEllipse(pts);

	Mat ellipseImg(srcImg.size(), CV_8UC1, cv::Scalar(255));
	cv::Point2d center(ellipse[0], ellipse[1]);
	Img_DrawEllipse(ellipseImg, center, ellipse[2], ellipse[3], ellipse[4], 0.2);
}