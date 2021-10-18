#include "Compute3DCircle.h"
#include "ComputePlane.h"
#include "ComputePlane.cpp"

//点到圆的距离====================================================================================
template <typename T1, typename T2>
void PC_PtToCircleDist(T1& pt, T2& circle, double& dist)
{
	double diff_x = pt.x - circle[0];
	double diff_y = pt.y - circle[1];
	double diff_z = pt.z - circle[2];
	dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
	dist = abs(dist - circle[3]);
}
//================================================================================================

//最小二乘法拟合空间空间园========================================================================
template <typename T1, typename T2>
void PC_OLSFit3DCircle(vector<T1>& pts, vector<double>& weights, T2& circle)
{
	cv::Vec4d plane;
	PC_OLSFitPlane(pts, weights, plane);

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

	double a = plane[0], b = plane[1], c = plane[2];
	Mat A(3, 3, CV_64FC1, cv::Scalar(0));
	Mat B(3, 1, CV_64FC1, cv::Scalar(0));
	double* pA = A.ptr<double>(0);
	double* pB = B.ptr<double>(0);
	for (int i = 0; i < pts.size(); ++i)
	{
		double x = pts[i].x, y = pts[i].y, z = pts[i].z, w = weights[i];
		double x_ = x - w_x_mean, y_ = y - w_y_mean, z_ = z - w_z_mean;
		pA[0] += w * (x_ * x_ + 0.25 * a * a);
		pA[1] += w * (x_ * y_ + 0.25 * a * b);
		pA[2] += w * (x_ * z_ + 0.25 * c * a);
		pA[4] += w * (y_ * y_ + 0.25 * b * b);
		pA[5] += w * (y_ * z_ + 0.25 * b * c);
		pA[8] += w * (z_ * z_ + 0.25 * c * c);

		double r_ = pts[i].x * pts[i].x + pts[i].y * pts[i].y + pts[i].z * pts[i].z - w_x2y2z2_mean;
		pB[0] -= w * (x_ * r_ + 0.5 * a * a * x + 0.5 * a * b * y + 0.5 * a * c * z);
		pB[1] -= w * (y_ * r_ + 0.5 * a * b * x + 0.5 * b * b * y + 0.5 * b * c * z);
		pB[2] -= w * (z_ * r_ + 0.5 * a * c * x + 0.5 * b * c * y + 0.5 * c * c * z);
	}
	pA[3] = pA[1]; pA[6] = pA[2]; pA[7] = pA[5];

	Mat C = (A.inv()) * B;
	double* pC = C.ptr<double>(0);
	circle[0] = -pC[0] / 2.0;
	circle[1] = -pC[1] / 2.0;
	circle[2] = -pC[2] / 2.0;
	double c_ = -(pC[0] * w_x_mean + pC[1] * w_y_mean + pC[2] * w_z_mean + w_x2y2z2_mean);
	circle[3] = std::sqrt(std::max(circle[0] * circle[0] + circle[1] * circle[1] + circle[2] * circle[2] - c_, EPS));
	circle[4] = a; circle[5] = b; circle[6] = c;
}
//================================================================================================

//Huber计算权重===================================================================================
template <typename T1, typename T2>
void PC_HuberCircleWeights(vector<T1>& pts, T2& circle, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double dist = 0.0;
		PC_PtToCircleDist(pts[i], circle, dist);;
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
//================================================================================================

//Tukey计算权重===================================================================================
template <typename T1, typename T2>
void PC_TukeyCircleWeights(vector<T1>& pts, T2& circle, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = 0.0;
		PC_PtToCircleDist(pts[i], circle, distance);
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
//================================================================================================

//拟合球==========================================================================================
template <typename T1, typename T2>
void PC_FitSphere(vector<T1>& pts, T2& circle, int k, NB_MODEL_FIT_METHOD method)
{
	vector<double> weights(pts.size(), 1);
	PC_OLSFit3DCircle(pts, weights, circle);
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
				PC_HuberCircleWeights(pts, circle, weights);
				break;
			case TUKEY_FIT:
				PC_TukeyCircleWeights(pts, circle, weights);
				break;
			default:
				break;
			}
			PC_OLSFit3DCircle(pts, weights, circle);
		}
	}
}
//================================================================================================

void PC_CircleTest()
{
	PC_XYZ::Ptr srcPC(new PC_XYZ);
	pcl::io::loadPLYFile("C:/Users/Administrator/Desktop/testimage/噪声圆.ply", *srcPC);

	vector<P_XYZ> pts(srcPC->points.size());
	for (int i = 0; i < srcPC->points.size(); ++i)
	{
		pts[i] = srcPC->points[i];
	}

	vector<double> circle(7,0.0);
	vector<double> weights(pts.size(), 1.0);
	PC_FitSphere(pts, circle, 5, NB_MODEL_FIT_METHOD::TUKEY_FIT);
	//vector<P_XYZ> inlinerPts;
	//PC_RANSACComputeSphere(pts, sphere, inlinerPts, 0.2);

	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC, 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(srcPC, red, "srcPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "srcPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}