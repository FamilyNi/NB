#include "ComputePlane.h"
#include "MathOpr.h"
#include "MathOpr.cpp"

//三点计算平面==================================================================================
template <typename T1, typename T2>
void PC_ThreePtsComputePlane(T1& pt1, T1& pt2, T1& pt3, T2& plane)
{
	P_XYZ nor_1(pt1.x - pt2.x, pt1.y - pt2.y, pt1.z - pt2.z);
	P_XYZ nor_2(pt1.x - pt3.x, pt1.y - pt3.y, pt1.z - pt3.z);
	P_XYZ norm(0, 0, 0);
	PC_VecCross(nor_1, nor_2, norm, true);
	if (abs(norm.x) < EPS && abs(norm.y) < EPS && abs(norm.z) < EPS)
		return;
	plane[0] = norm.x; plane[1] = norm.y; plane[2] = norm.z;
	plane[3] = -(plane[0] * pt1.x + plane[1] * pt1.y + plane[2] * pt1.z);
}
//==============================================================================================

//随机一致采样算法计算平面======================================================================
template <typename T1, typename T2>
void PC_RANSACComputePlane(vector<T1>& pts, T2& plane, vector<T1>& inlinerPts, double thres)
{
	if (pts.size() < 6)
		return;
	int best_model_p = 0;
	double P = 0.99;  //模型存在的概率
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	vector<T1> pts_(3);
	for (int i = 0; i < maxEpo; ++i)
	{
		int effetPoints = 0;
		//随机选择六个个点计算平面
		pts_[0] = pts[rand() % size]; pts_[1] = pts[rand() % size];	pts_[2] = pts[rand() % size];
		T2 plane_;
		PC_ThreePtsComputePlane(pts_[0], pts_[1], pts_[2], plane_);
		//计算局内点的个数
		for (int j = 0; j < size; ++j)
		{
			double dist = 0.0;
			PC_PtToPlaneDist(pts[i], plane_, dist);
			effetPoints += dist < thres ? 1 : 0;
		}
		//获取最优模型，并根据概率修改迭代次数
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			plane = plane_;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
		if (best_model_p > 0.5 * size)
		{
			plane = plane_;
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
		PC_PtToPlaneDist(pts[i], plane, dist);
		if (dist < thres)
			inlinerPts.push_back(pts[i]);
	}
}
//==============================================================================================

//最小二乘法拟合平面============================================================================
template <typename T1, typename T2>
void PC_OLSFitPlane(vector<T1>& pts, vector<double>& weights, T2& plane)
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
template <typename T1, typename T2>
void PC_HuberPlaneWeights(vector<T1>& pts, T2& plane, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = 0.0;
		PC_PtToPlaneDist(pts[i], plane, distance);
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
void PC_TukeyPlaneWeights(vector<T1>& pts, T2& plane, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = 0.0;
		PC_PtToPlaneDist(pts[i], plane, distance);
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

//平面拟合======================================================================================
template <typename T1, typename T2>
void PC_FitPlane(vector<T1>& pts, T2& plane, int k, NB_MODEL_FIT_METHOD method)
{
	vector<double> weights(pts.size(), 1);
	PC_OLSFitPlane(pts, weights, plane);
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
				PC_HuberPlaneWeights(pts, plane, weights);
				break;
			case TUKEY_FIT:
				PC_TukeyPlaneWeights(pts, plane, weights);
				break;
			default:
				break;
			}
			PC_OLSFitPlane(pts, weights, plane);
		}
	}
}
//==============================================================================================


void PC_PlaneTest()
{
	PC_XYZ::Ptr srcPC(new PC_XYZ);
	pcl::io::loadPLYFile("C:/Users/Administrator/Desktop/testimage/噪声平面.ply", *srcPC);

	vector<P_XYZ> pts(srcPC->points.size());
	for (int i = 0; i < srcPC->points.size(); ++i)
	{
		pts[i] = srcPC->points[i];
	}
	cv::Vec4d plane;
	vector<P_XYZ> inlinerPts;
	PC_RANSACComputePlane(pts, plane, inlinerPts, 0.01);

	PC_XYZ::Ptr inlinerPC(new PC_XYZ);
	inlinerPC->points.resize(inlinerPts.size());
	for (int i = 0; i < inlinerPts.size(); ++i)
	{
		inlinerPC->points[i] = inlinerPts[i];
	}
	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> write(inlinerPC, 255, 255, 255); //设置点云颜色
	viewer.addPointCloud(inlinerPC, write, "inlinerPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inlinerPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
	//PC_FitPlane(pts, plane, 5, NB_MODEL_FIT_METHOD::HUBER_FIT);
}