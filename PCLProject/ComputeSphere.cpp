#include "ComputeSphere.h"
#include "DrawShape.h"

//点到园的距离==================================================================================
template <typename T1, typename T2>
void PC_PtToShpereDist(T1& pt, T2& sphere, double& dist)
{
	double diff_x = pt.x - sphere[0];
	double diff_y = pt.y - sphere[1];
	double diff_z = pt.z - sphere[2];
	dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
	dist = abs(dist - sphere[3]);
}
//==============================================================================================

//四点计算球====================================================================================
template <typename T1, typename T2>
void PC_FourPtsComputeSphere(vector<T1>& pts, T2& sphere)
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
}
//==============================================================================================

//随机一致采样算法计算球========================================================================
template <typename T1, typename T2>
void PC_RANSACComputeSphere(vector<T1>& pts, T2& sphere, vector<T1>& inlinerPts, double thres)
{
	if (pts.size() < 6)
		return;
	int best_model_p = 0;
	double P = 0.99;  //模型存在的概率
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	vector<T1> pts_(4);
	for (int i = 0; i < maxEpo; ++i)
	{
		int effetPoints = 0;
		//随机选择六个个点计算椭圆
		pts_[0] = pts[rand() % size]; pts_[1] = pts[rand() % size];
		pts_[2] = pts[rand() % size]; pts_[3] = pts[rand() % size];
		T2 sphere_;
		PC_FourPtsComputeSphere(pts_, sphere_);
		//计算局内点的个数
		for (int j = 0; j < size; ++j)
		{
			double dist = 0.0;
			PC_PtToShpereDist(pts[j], sphere_, dist);
			effetPoints += dist < thres ? 1 : 0;
		}
		//获取最优模型，并根据概率修改迭代次数
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			sphere = sphere_;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
		if (best_model_p > 0.5 * size)
		{
			sphere = sphere_;
			break;
		}
	}
	//提取局内点
	if (inlinerPts.size() != 0)
		inlinerPts.resize(0);
	inlinerPts.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		double dist = 0.0;
		PC_PtToShpereDist(pts[i], sphere, dist);
		if (dist < thres)
			inlinerPts.push_back(pts[i]);
	}
}
//==============================================================================================

//最小二乘法拟合球==============================================================================
template <typename T1, typename T2>
void PC_OLSFitSphere(vector<T1>& pts, vector<double>& weights, T2& sphere)
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

//Huber计算权重=================================================================================
template <typename T1, typename T2>
void PC_HuberSphereWeights(vector<T1>& pts, T2& sphere, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double dist = 0.0;
		PC_PtToShpereDist(pts[i], sphere, dist);;
		if (dist <= tao)
		{
			weights[i] = 1;
		}
		else
		{
			weights[i] = tao / dist;
		}
	}
}
//==============================================================================================

//Tukey计算权重================================================================================
template <typename T1, typename T2>
void PC_TukeySphereWeights(vector<T1>& pts, T2& sphere, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = 0.0;
		PC_PtToShpereDist(pts[i], sphere, distance);
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
template <typename T1, typename T2>
void PC_FitSphere(vector<T1>& pts, T2& sphere, int k, NB_MODEL_FIT_METHOD method)
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
			case TUKEY_FIT:
				PC_TukeySphereWeights(pts, sphere, weights);
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
	PC_FitSphere(pts, sphere, 5, NB_MODEL_FIT_METHOD::TUKEY_FIT);
	//vector<P_XYZ> inlinerPts;
	//PC_RANSACComputeSphere(pts, sphere, inlinerPts, 0.2);

	PC_XYZ::Ptr spherePC(new PC_XYZ);
	P_XYZ center(sphere[0], sphere[1], sphere[2]);
	PC_DrawSphere(spherePC, center, sphere[3], 0.1);

	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC, 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(srcPC, red, "srcPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "srcPC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> write(spherePC, 255, 255, 255); //设置点云颜色
	viewer.addPointCloud(spherePC, write, "spherePC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "spherePC");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}