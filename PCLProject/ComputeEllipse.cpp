#include "ComputeEllipse.h"
#include "DrawShape.h"

//椭圆方程标准化========================================================================
void Img_EllipseNormalization(cv::Vec6d& ellipse_, cv::Vec6d& normEllipse)
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
template <typename T>
void Img_PtsToEllipseDist(T& pt, cv::Vec6d& ellipse, double& dist)
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

//最小二乘法拟合椭圆====================================================================
template <typename T>
void Img_OLSFitEllipse(vector<T>& pts, vector<double>& weights, cv::Vec6d& ellipse)
{
	if (pts.size() < 6)
		return;
	Mat C(3, 3, CV_64FC1, cv::Scalar(0));
	C.at<double>(0, 2) = -2;
	C.at<double>(1,1) = 1;
	C.at<double>(2, 0) = -2;
	int pts_num = pts.size();

	Mat invC = C.inv();
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
		double x = w * pts[i].x;
		double y = w * pts[i].y;
		double x_2 = x * x;
		double y_2 = y * y;
		double xy = x * y;
		pS1[0] += x_2 * x_2; pS1[1] += x_2 * xy; pS1[2] += x_2 * y_2;
		pS1[3] += xy * x_2, pS1[4] += xy * xy; pS1[5] += xy * y_2;
		pS1[6] += y_2 * x_2; pS1[7] += y_2 * xy; pS1[8] += y_2 * y_2;

		pS2[0] += x_2 * x; pS2[1] += x_2 * y; pS2[2] += x_2 * w; 
		pS2[3] += xy * x; pS2[4] += xy * y; pS2[5] += xy * w;
		pS2[6] += y_2 * x; pS2[7] += y_2 * y; pS2[8] += y_2 * w;
		
		pS4[0] += x_2; pS4[1] += xy; pS4[2] += x * w;
		pS4[3] += xy; pS4[4] += y_2; pS4[5] += y * w;
		pS4[6] += w * x; pS4[7] += w * y; pS4[8] += w * w;
	}
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
template <typename T>
void Img_HuberEllipseWeights(vector<T>& pts, cv::Vec6d& ellipse, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = 0.0;
		Img_PtsToEllipseDist(pts[i], ellipse, distance);
		cout << distance << endl;
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
template <typename T>
void Img_TurkeyEllipseWeights(vector<T>& pts, cv::Vec6d& ellipse, vector<double>& weights)
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
template <typename T>
void Img_FitEllipse(vector<T>& pts, cv::Vec6d& ellipse, int k, NB_MODEL_FIT_METHOD method)
{
	cv::Vec6d ellipse_;
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
			case TURKEY_FIT:
				Img_TurkeyEllipseWeights(pts, ellipse, weights);
				break;
			default:
				break;
			}
			Img_OLSFitEllipse(pts, weights, ellipse_);
			Img_EllipseNormalization(ellipse_, ellipse);
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

	for (int i = 0; i < pts.size(); ++i)
	{
		cv::line(colorImg, pts[i], pts[i], cv::Scalar(0, 0, 255), 2);

	}

	cv::Vec6d line;
	Img_FitEllipse(pts, line, 5, NB_MODEL_FIT_METHOD::TURKEY_FIT);

	cv::RotatedRect rect = cv::fitEllipse(pts);

	Mat ellipseImg(srcImg.size(), CV_8UC1, cv::Scalar(255));
	cv::Point2d center(line[0], line[1]);
	Img_DrawEllipse(ellipseImg, center, line[2], line[3], line[4], 0.2);
	//cv::Point2d center(rect.center);
	//Img_DrawEllipse(ellipseImg, center, rect.angle / 180 * CV_PI, rect.size.width / 2.0, rect.size.height / 2.0, 0.2);
}