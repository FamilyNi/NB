#include "PC_UTILS.h"

//计算点云重心=======================================================
void ComputePCGravity(PC_XYZ::Ptr &srcPC, P_XYZ &gravity)
{
	if (srcPC->empty())
	{
		gravity.x = 0;
		gravity.y = 0;
		gravity.z = 0;
		return;
	}
	float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
	int num_points = srcPC->points.size();
	for (int i = 0; i < num_points; ++i)
	{
		sum_x += srcPC->points[i].x;
		sum_y += srcPC->points[i].y;
		sum_z += srcPC->points[i].z;
	}
	gravity.x = sum_x / num_points;
	gravity.y = sum_y / num_points;
	gravity.z = sum_z / num_points;;
}
//===================================================================

//将点云投影到XY二维平面=============================================
void PC_ProjectToXY(PC_XYZ::Ptr &srcPC, cv::Mat &xyPlane)
{
	P_XYZ min_pt, max_pt;
	int scale = 1;
	pcl::getMinMax3D(*srcPC, min_pt, max_pt);
	int imgW = (max_pt.x - min_pt.x) * scale + 10;
	int imgH = (max_pt.y - min_pt.y) * scale + 10;
	int z_Scalar = 255 / (max_pt.z - min_pt.z);

	xyPlane = cv::Mat(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	uchar* pImagXY = xyPlane.ptr<uchar>();
	for (int i = 0; i < srcPC->points.size(); ++i)
	{
		P_XYZ& p_ = srcPC->points[i];
		int index_x = (p_.x - min_pt.x) * scale + 5;
		int index_y = (p_.y - min_pt.y) * scale + 5;
		pImagXY[index_y * imgW + index_x] = (p_.z - min_pt.z) * z_Scalar;
	}
}
//===================================================================

//计算点云的法向量===================================================
void ComputePCNormal(PC_XYZ::Ptr &srcPC, PC_N::Ptr &normals, float radius)
{
	if (srcPC->empty())
		return;
	//计算法线
	pcl::NormalEstimation<P_XYZ, pcl::Normal> normal_est;
	normal_est.setInputCloud(srcPC);
	normal_est.setRadiusSearch(radius);
	pcl::search::KdTree<P_XYZ>::Ptr kdtree(new pcl::search::KdTree<P_XYZ>);
	normal_est.setSearchMethod(kdtree);
	normal_est.compute(*normals);
}
//===================================================================

//计算协方差矩阵=====================================================
void ComputeCovMat(PC_XYZ::Ptr &pc, cv::Mat &covMat, P_XYZ &gravity)
{
	if (pc->empty() || covMat.size() != cv::Size(3, 3))
		return;
	double sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
	size_t point_num = pc->points.size();

	for (size_t i = 0; i < point_num; ++i)
	{
		sum_x += pc->points[i].x; sum_y += pc->points[i].y;	sum_z += pc->points[i].z;
	}
	double mean_x = sum_x / point_num;
	double mean_y = sum_y / point_num;
	double mean_z = sum_z / point_num;

	vector<double> ori_x(point_num), ori_y(point_num), ori_z(point_num);
	for (int i = 0; i < point_num; ++i)
	{
		ori_x[i] = pc->points[i].x - mean_x;
		ori_y[i] = pc->points[i].y - mean_y; 
		ori_z[i] = pc->points[i].z - mean_z;
	}
	double *pCovMat = covMat.ptr<double>(0);
	for (int i = 0; i < point_num; ++i)
	{
		pCovMat[0] += (ori_x[i] * ori_x[i]);
		pCovMat[4] += (ori_y[i] * ori_y[i]);
		pCovMat[8] += (ori_z[i] * ori_z[i]);

		pCovMat[1] += (ori_x[i] * ori_y[i]);
		pCovMat[5] += (ori_y[i] * ori_z[i]);
		pCovMat[2] += (ori_z[i] * ori_x[i]);
	}
	pCovMat[3] = pCovMat[1]; pCovMat[6] = pCovMat[2]; pCovMat[7] = pCovMat[5];
	gravity.x = mean_x; gravity.y = mean_y; gravity.z = mean_z;
}
//===================================================================