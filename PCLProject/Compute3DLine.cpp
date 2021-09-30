#include "Compute3DLine.h"
#include "MathOpr.h"

//空间两点求直线==================================================================================
template <typename T>
void PC_OLSFit3DLine(T& pt1, T& pt2, cv::Vec6d& line)
{
	line[0] = pt1.x - pt2.x;
	line[1] = pt1.y - pt2.y;
	line[2] = pt1.z - pt2.z;
	double norm_ = 1.0 / std::max(std::sqrt(line[0] * line[0] + line[1] * line[1] + line[2] * line[2]), EPS);
	line[0] *= norm_; line[1] *= norm_; line[2] *= norm_;
	line[3] = pt1.x; line[4] = pt1.y; line[5] = pt1.y;
}
//================================================================================================

//最小二乘法拟合空间直线==========================================================================
template <typename T>
void PC_OLSFit3DLine(vector<T>& pts, vector<double>& weights, cv::Vec6d& line)
{
	double w_sum = 0.0, w_x_sum = 0.0, w_y_sum = 0.0, w_z_sum = 0.0;
	double w_xy_sum = 0.0, w_yz_sum = 0.0, w_zx_sum = 0.0;
	for (int i = 0; i < pts.size(); ++i)
	{
		double w = weights[i], x = pts[i].x, y = pts[i].y, z = pts[i].z;
		w_x_sum += w * x; w_y_sum += w * y; w_z_sum += w * z; w_sum += w;
	}
	w_sum = 1.0 / std::max(w_sum, EPS);
	double w_x_mean = w_x_sum * w_sum;
	double w_y_mean = w_y_sum * w_sum;
	double w_z_mean = w_z_sum * w_sum;

	Mat A(3, 3, CV_64FC1, cv::Scalar(0));
	double* pA = A.ptr<double>(0);
	for (int i = 0; i < pts.size(); ++i)
	{
		double w = weights[i], x = pts[i].x, y = pts[i].y, z = pts[i].z;
		double x_ = x - w_x_mean;
		double y_ = y - w_y_mean;
		double z_ = z - w_z_mean;

		pA[0] += w * (y_ * y_ + z_ * z_);
		pA[1] -= w * x_ * y_;
		pA[2] -= w * z_ * x_;
		pA[4] += w * (x_ * x_ + z_ * z_);
		pA[5] -= w * y_ * z_;
		pA[8] += w * (x_ * x_ + y_ * y_);
	}
	pA[3] = pA[1]; pA[6] = pA[2]; pA[7] = pA[5];

	cv::Mat eigenVal, eigenVec;
	cv::eigen(A, eigenVal, eigenVec);
	double* pEigenVec = eigenVec.ptr<double>(2);
	for (int i = 0; i < 3; ++i)
		line[i] = pEigenVec[i];
	line[3] = w_x_mean; line[4] = w_y_mean; line[5] = w_z_mean;
}
//================================================================================================

//Huber计算权重===================================================================================
template <typename T>
void PC_Huber3DLineWeights(vector<T>& pts, cv::Vec6d& line, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = 0.0;
		PC_PtToLineDist(pts[i], line, distance);
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
//================================================================================================

//Turkey计算权重==================================================================================
template <typename T>
void PC_Turkey3DLineWeights(vector<T>& pts, cv::Vec6d& line, vector<double>& weights)
{
	vector<double> dists(pts.size(), 0.0);
	for (int i = 0; i < pts.size(); ++i)
	{
		PC_PtToLineDist(pts[i], line, dists[i]);
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

//空间直线拟合====================================================================================
template <typename T>
void PC_Fit3DLine(vector<T>& pts, cv::Vec6d& line, int k, NB_MODEL_FIT_METHOD method)
{
	vector<double> weights(pts.size(), 1);
	PC_OLSFit3DLine(pts, weights, line);
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
				PC_Huber3DLineWeights(pts, line, weights);
				break;
			case TURKEY_FIT:
				PC_Turkey3DLineWeights(pts, line, weights);
				break;
			default:
				break;
			}
			PC_OLSFit3DLine(pts, weights, line);
		}
	}
}
//==============================================================================================

void PC_3DLineTest()
{
	PC_XYZ::Ptr srcPC(new PC_XYZ);
	pcl::io::loadPLYFile("C:/Users/Administrator/Desktop/testimage/噪声直线.ply", *srcPC);

	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> write(srcPC, 255, 255, 255); //设置点云颜色
	viewer.addPointCloud(srcPC, write, "srcPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "srcPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}

	vector<P_XYZ> pts(srcPC->points.size());
	for (int i = 0; i < srcPC->points.size(); ++i)
	{
		pts[i] = srcPC->points[i];
	}
	cv::Vec6d line;
	PC_Fit3DLine(pts, line, 5, NB_MODEL_FIT_METHOD::TURKEY_FIT);
}