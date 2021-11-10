#include "../../include/FitShapeFile/PC_FitSphere.h"
#include "../../include/FitShapeFile/ComputeModels.h"
#include "../../include/BaseOprFile/MathOpr.h"

//随机一致采样算法计算球========================================================================
void PC_RANSACFitSphere(NB_Array3D pts, Sphere3D& sphere, vector<int>& inliners, double thres)
{
	if (pts.size() < 6)
		return;
	int best_model_p = 0;
	double P = 0.99;  //模型存在的概率
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	vector<Point3d> pts_(4);
	for (int i = 0; i < maxEpo; ++i)
	{
		int effetPoints = 0;
		//随机选择四个点计算球---注意：这里可能需要特殊处理防止点相同
		pts_[0] = pts[rand() % size]; pts_[1] = pts[rand() % size];
		pts_[2] = pts[rand() % size]; pts_[3] = pts[rand() % size];
		Sphere3D sphere_;
		PC_FourPtsComputeSphere(pts_, sphere_);
		//计算局内点的个数
		for (int j = 0; j < size; ++j)
		{
			double dist = 0.0;
			PC_PtToCircleDist(pts[j], sphere_, dist);
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
	if (inliners.size() != 0)
		inliners.resize(0);
	inliners.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		double dist = 0.0;
		PC_PtToCircleDist(pts[i], sphere, dist);
		if (dist < thres)
			inliners.push_back(i);
	}
}
//==============================================================================================

//最小二乘法拟合球==============================================================================
void PC_OLSFitSphere(NB_Array3D pts, vector<double>& weights, Sphere3D& sphere)
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
	sphere.x = -pC[0] / 2.0;
	sphere.y = -pC[1] / 2.0;
	sphere.z = -pC[2] / 2.0;
	double c = -(pC[0] * w_x_mean + pC[1] * w_y_mean + pC[2] * w_z_mean + w_x2y2z2_mean);
	sphere.r = std::sqrt(sphere.x * sphere.x + sphere.y * sphere.y + sphere.z * sphere.z - c);
}
//==============================================================================================

//Huber计算权重=================================================================================
void PC_HuberSphereWeights(NB_Array3D pts, Sphere3D& sphere, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double dist = 0.0;
		PC_PtToCircleDist(pts[i], sphere, dist);;
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
void PC_TukeySphereWeights(NB_Array3D pts, Sphere3D& sphere, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = 0.0;
		PC_PtToCircleDist(pts[i], sphere, distance);
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
void PC_FitSphere(NB_Array3D pts, Sphere3D& sphere, int k, NB_MODEL_FIT_METHOD method)
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

//空间求拟合测试==============================================================================
void PC_FitSphereTest()
{
	PC_XYZ::Ptr srcPC(new PC_XYZ);
	pcl::io::loadPLYFile("C:/Users/Administrator/Desktop/testimage/噪声球.ply", *srcPC);

	vector<P_XYZ> pts(srcPC->points.size());
	for (int i = 0; i < srcPC->points.size(); ++i)
	{
		pts[i] = srcPC->points[i];
	}
	std::random_shuffle(pts.begin(), pts.end());
	Sphere3D sphere;
	//PC_FitSphere(pts, sphere, 5, NB_MODEL_FIT_METHOD::OLS_FIT);

	vector<int> inliners;
	PC_RANSACFitSphere(pts, sphere, inliners, 0.2);
	PC_XYZ::Ptr inlinerPC(new PC_XYZ);
	inlinerPC->points.resize(inliners.size());
	for (int i = 0; i < inliners.size(); ++i)
	{
		inlinerPC->points[i] = pts[inliners[i]];
	}

	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC, 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(srcPC, red, "srcPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "srcPC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> write(inlinerPC, 255, 255, 255); //设置点云颜色
	viewer.addPointCloud(inlinerPC, write, "spherePC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "spherePC");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
//============================================================================================