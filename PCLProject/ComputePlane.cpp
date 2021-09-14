#include "ComputePlane.h"
#include "MathOpr.h"

//三点计算平面==================================================================================
template <typename T>
void PC_ThreePtsComputePlane(T& pt1, T& pt2, T& pt3, cv::Vec4d& plane)
{
	P_XYZ nor_1(pt1.x - pt2.x, pt1.y - pt2.y, pt1.z - pt2.z);
	P_XYZ nor_2(pt1.x - pt3.x, pt1.y - pt3.y, pt1.z - pt3.z);
	P_XYZ norm(0, 0, 0);
	VecCross_PC(nor_1, nor_2, norm);
	if (abs(norm.x) < EPS && abs(norm.y) < EPS && abs(norm.z) < EPS)
		return;
	Normal_PC(norm);
	plane[0] = norm.x; plane[1] = norm.y; plane[2] = norm.z;
	plane[3] = -(plane[0] * pt1.x + plane[1] * pt1.y + plane[2] * pt1.z);
}
//==============================================================================================

//最小二乘法拟合平面============================================================================
template <typename T>
void PC_OLSFitPlane(vector<T>& pts, vector<double>& weights, cv::Vec4d& plane)
{
	if (pts.size() < 3)
		return;
	double w_sum = 0.0;
	double w_x_sum = 0.0;
	double w_y_sum = 0.0;
	double w_z_sum = 0.0;
	for (int i = 0; i < pts.size(); ++i)
	{
		w_sum += weights[i];
		w_x_sum += weights[i] * pts[i].x;
		w_y_sum += weights[i] * pts[i].y;
		w_z_sum += weights[i] * pts[i].z;
	}
	w_sum = 1.0 / std::max(w_sum, EPS);
	double w_x_mean = w_x_sum * w_sum;
	double w_y_mean = w_y_sum * w_sum;
	double w_z_mean = w_z_sum * w_sum;

	cv::Mat A(3, 3, CV_64FC1, cv::Scalar(0));
	double *pA = A.ptr<double>(0);
	for (int i = 0; i < pts.size(); ++i)
	{
		double x_ = pts[i].x - w_x_mean;
		double y_ = pts[i].y - w_y_mean;
		double z_ = pts[i].z - w_z_mean;
		pA[0] += weights[i] * x_ * x_;
		pA[4] += weights[i] * y_ * y_;
		pA[8] += weights[i] * z_ * z_;
		pA[1] += weights[i] * x_ * y_;
		pA[2] += weights[i] * x_ * z_;
		pA[5] += weights[i] * y_ * z_;
	}
	pA[3] = pA[1]; pA[6] = pA[2]; pA[7] = pA[5];
	cv::Mat eigenVal, eigenVec;
	cv::eigen(A, eigenVal, eigenVec);
	double* pEigenVec = eigenVec.ptr<double>(2);
	for (int i = 0; i < 3; ++i)
		plane[i] = pEigenVec[i];
	plane[3] = -(plane[0] * w_x_mean + plane[1] * w_y_mean + plane[2] * w_z_mean);
}
//==============================================================================================

//Huber计算权重=================================================================================
template <typename T>
void Img_HuberPlaneWeights(vector<T>& pts, cv::Vec4d& plane, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = abs(pts[i].x * plane[0] + pts[i].y * plane[1] + pts[i].z * plane[2] + plane[3]);
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

//Turkey计算权重================================================================================
template <typename T>
void Img_TurkeyPlaneWeights(vector<T>& pts, cv::Vec4d& plane, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = abs(pts[i].x * plane[0] + pts[i].y * plane[1] + pts[i].z * plane[2] + plane[3]);
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

//直线拟合======================================================================================
template <typename T>
void Img_FitLine(vector<T>& pts, cv::Vec4d& plane, int k, NB_MODEL_FIT_METHOD method)
{
	vector<double> weights(pts.size(), 1);
	if (method == NB_MODEL_FIT_METHOD::OLS_FIT)
	{
		PC_OLSFitPlane(pts, weights, circle);
		return;
	}

	for (int i = 0; i < k; ++i)
	{
		PC_OLSFitPlane(pts, weights, circle);
		switch (method)
		{
		case HUBER_FIT:
			Img_HuberPlaneWeights(pts, circle, weights);
			break;
		case TURKEY_FIT:
			Img_TurkeyPlaneWeights(pts, circle, weights);
			break;
		default:
			break;
		}
	}
}
//==============================================================================================