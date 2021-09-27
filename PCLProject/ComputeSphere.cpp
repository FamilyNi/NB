#include "ComputeSphere.h"

//四点计算球====================================================================================
template <typename T>
void PC_FourPtsComputeSphere(vector<T>& pts, cv::Vec4d& sphere)
{
	if (pts.size() != 4)
		return;
	cv::Mat XYZ(cv::Size(3, 3), CV_64FC1, cv::Scalar(0));
	double* pXYZ = XYZ.ptr<double>();
	cv::Mat m(cv::Size(1, 3), CV_64FC1, cv::Scalar(0));
	double* pM = m.ptr<double>();
	for (int i = 0; i < pts.size() - 1; ++i)
	{
		int idx = 3 * i;
		pXYZ[idx] = pts[i].x - pts[i + 1].x;
		pXYZ[idx + 1] = pts[i].y - pts[i + 1].y;
		pXYZ[idx + 2] = pts[i].z - pts[i + 1].z;

		double pt0_d = pts[i].x * pts[i].x + pts[i].y * pts[i].y + pts[i].z * pts[i].z;
		double pt1_d = pts[i + 1].x * pts[i + 1].x + pts[i + 1].y * pts[i + 1].y + pts[i + 1].z * pts[i + 1].z;
		pM[i] = (pt0_d - pt1_d) / 2.0;
	}

	cv::Mat center = (XYZ.inv()) * m;
	sphere[0] = center.ptr<double>(0)[0];
	sphere[1] = center.ptr<double>(0)[1];
	sphere[2] = center.ptr<double>(0)[2];
	double diff_x = pts[0].x - sphere[0];
	double diff_y = pts[0].y - sphere[1];
	double diff_z = pts[0].z - sphere[2];
	sphere[3] = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
	return;
}
//==============================================================================================

//最小二乘法拟合球==============================================================================
template <typename T>
void PC_OLSFitSphere(vector<T>& pts, vector<double>& weights, cv::Vec4d& sphere)
{
	double w_sum = 0.0;
	double w_x_sum = 0.0;
	double w_y_sum = 0.0;
	double w_z_sum = 0.0;
	double w_x2y2z2_sum = 0.0;
	for (int i = 0; i < pts.size(); ++i)
	{
		w_sum += weights[i];
		w_x_sum += weights[i] * pts[i].x;
		w_y_sum += weights[i] * pts[i].y;
		w_z_sum += weights[i] * pts[i].z;
		w_x2y2z2_sum += weights[i] * (pts[i].x * pts[i].x + pts[i].y * pts[i].y + pts[i].z * pts[i].z);
	}
	w_sum = 1.0 / std::max(w_sum, EPS);
	double w_x_mean = w_x_sum * w_sum;
	double w_y_mean = w_y_sum * w_sum;
	double w_z_mean = w_z_sum * w_sum;
	double w_x2y2z2_mean = w_x2y2z2_sum * w_sum;

	Mat A(3, 3, CV_64FC1, cv::Scalar(0));
	Mat B(3, 1, CV_64FC1, cv::Scalar(0));
	double* pA = A.ptr<double>(0);
	double* pB = B.ptr<double>(0);
	for (int i = 0; i < pts.size(); ++i)
	{
		double x_ = pts[i].x - w_x_mean;
		double y_ = pts[i].y - w_y_mean;
		double z_ = pts[i].z - w_z_mean;
		pA[0] += weights[i] * x_ * x_;
		pA[1] += weights[i] * x_ * y_;
		pA[2] += weights[i] * x_ * z_;
		pA[4] += weights[i] * y_ * y_;
		pA[5] += weights[i] * y_ * z_;
		pA[8] += weights[i] * z_ * z_;

		double r_ = pts[i].x * pts[i].x + pts[i].y * pts[i].y + pts[i].z * pts[i].z - w_x2y2z2_mean;
		pB[0] -= weights[i] * x_ * r_;
		pB[1] -= weights[i] * y_ * r_;
		pB[2] -= weights[i] * z_ * r_;
	}
	pA[3] = pA[1]; pA[6] = pA[2]; pA[7] = pA[5];

	Mat C = (A.inv()) * B;
	double* pC = C.ptr<double>(0);
	sphere[0] = -pC[0] / 2.0;
	sphere[1] = -pC[1] / 2.0;
	sphere[2] = -pC[2] / 2.0;
	double c = -(pC[0] * w_x_mean + pC[1] * w_y_mean + pC[2] * w_z_mean + w_x2y2z2_mean);
	sphere[3] = std::sqrt(std::max(sphere[0] * sphere[0] + sphere[1] * sphere[1] + sphere[2] * sphere[2] - c, EPS));
}
//==============================================================================================

//huber计算权重=================================================================================
template <typename T>
void PC_HuberSphereWeights(vector<T>& pts, cv::Vec4d& sphere, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double diff_x = pts[i].x - sphere[0];
		double diff_y = pts[i].y - sphere[1];
		double diff_z = pts[i].z - sphere[2];
		double distance = std::sqrt(max(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z, EPS));
		distance = abs(distance - sphere[3]);
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
void PC_TurkeySphereWeights(vector<T>& pts, cv::Vec4d& sphere, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		double diff_x = pts[i].x - sphere[0];
		double diff_y = pts[i].y - sphere[1];
		double diff_z = pts[i].z - sphere[2];
		double distance = std::sqrt(max(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z, EPS));
		distance = abs(distance - sphere[3]);
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

//拟合球========================================================================================
template <typename T>
void PC_FitSphere(vector<T>& pts, cv::Vec4d& sphere, int k, NB_MODEL_FIT_METHOD method)
{
	vector<double> weights(pts.size(), 1);
	PC_OLSFitSphere(pts, weights, sphere);
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
				PC_HuberSphereWeights(pts, sphere, weights);
				break;
			case TURKEY_FIT:
				PC_TurkeySphereWeights(pts, sphere, weights);
				break;
			default:
				break;
			}
			PC_OLSFitSphere(pts, weights, sphere);
		}
	}
}
//==============================================================================================

void PC_SphereTest()
{
	PC_XYZ::Ptr srcPC(new PC_XYZ);
	pcl::io::loadPLYFile("C:/Users/Administrator/Desktop/testimage/噪声球.ply", *srcPC);

	vector<P_XYZ> pts(srcPC->points.size());
	for (int i = 0; i < srcPC->points.size(); ++i)
	{
		pts[i] = srcPC->points[i];
	}

	cv::Vec4d sphere;
	PC_FitSphere(pts, sphere, 5, NB_MODEL_FIT_METHOD::TURKEY_FIT);
}