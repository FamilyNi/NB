#include "PointCloudOpr.h"

//计算点云的最小包围盒==============================================================
void PC_ComputeOBB(const PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& obb)
{
	Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*srcPC, pcaCentroid);
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*srcPC, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();

	Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity();
	transMat.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();;
	transMat.block<3, 1>(0, 3) = -1.0f * transMat.block<3, 3>(0, 0) * (pcaCentroid.head<3>());

	PC_XYZ::Ptr transPC(new PC_XYZ);
	pcl::transformPointCloud(*srcPC, *transPC, transMat);

	P_XYZ min_p, max_p;
	pcl::getMinMax3D(*transPC, min_p, max_p);
	PC_XYZ::Ptr tran_box(new PC_XYZ);
	tran_box->points.resize(8);
	tran_box->points[0] = min_p;
	tran_box->points[1] = { max_p.x, min_p.y,  min_p.z };
	tran_box->points[2] = { max_p.x, max_p.y,  min_p.z };
	tran_box->points[3] = { min_p.x, max_p.y,  min_p.z };

	tran_box->points[4] = { min_p.x, min_p.y,  max_p.z };
	tran_box->points[5] = { max_p.x, min_p.y,  max_p.z };
	tran_box->points[6] = max_p;
	tran_box->points[7] = { min_p.x, max_p.y,  max_p.z };
	pcl::transformPointCloud(*tran_box, *obb, transMat.inverse());
}
//==================================================================================

//提取点云==========================================================================
void PC_ExtractPC(const PC_XYZ::Ptr& srcPC, vector<int>& indexes, PC_XYZ::Ptr& dstPC)
{
	uint size_src = srcPC->points.size();
	uint size_idx = indexes.size();
	if (size_idx == 0 || size_src == 0 || size_src < size_idx)
		return;
	dstPC->points.resize(size_idx);
	for (uint i = 0; i < size_idx; ++i)
	{
		dstPC->points[i] = srcPC->points[indexes[i]];
	}
}
//===================================================================================

//点云直线投影平滑===================================================================
void PC_LineProjSmooth(const PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, int size, float thresVal)
{
	size_t length = srcPC->points.size();
	KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC);
	vector<cv::Point3f> cvp_(size);
	dstPC->points.resize(length);
	for (int i = 0; i < length; ++i)
	{
		vector<int> PIdx(0);
		vector<float> PDist(0);
		P_XYZ& p_ = srcPC->points[i];
		kdtree.nearestKSearch(p_, size, PIdx, PDist);
		for (int j = 0; j < size; ++j)
		{
			cvp_[j].x = srcPC->points[PIdx[j]].x;
			cvp_[j].y = srcPC->points[PIdx[j]].y;
			cvp_[j].z = srcPC->points[PIdx[j]].z;
		}
		cv::Vec6f line;
		cv::fitLine(cvp_, line, cv::DIST_HUBER, 0, 0.01, 0.01);
		float scale = p_.x * line[0] + p_.y * line[1] + p_.z * line[2] -
			(line[3] * line[0] + line[4] * line[1] + line[5] * line[2]);
		P_XYZ v_p;
		v_p.x = line[3] + scale * line[0]; v_p.y = line[4] + scale * line[1]; v_p.z = line[5] + scale * line[2];
		float dist = v_p.x * p_.x + v_p.y * p_.y + v_p.z * p_.z;
		if (dist > thresVal)
			dstPC->points[i] = v_p;
		else
			dstPC->points[i] = p_;
	}
}
//===================================================================================

//计算点云的重心=====================================================================
void PC_GetPCGravity(PC_XYZ::Ptr &srcPC, P_XYZ &gravity)
{
	int point_num = srcPC->points.size();
	if (point_num == 0)
		return;
	float sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
	for (int i = 0; i < point_num; ++i)
	{
		sum_x += srcPC->points[i].x;
		sum_y += srcPC->points[i].y;
		sum_z += srcPC->points[i].z;
	}
	gravity.x = sum_x / point_num;
	gravity.y = sum_y / point_num;
	gravity.z = sum_z / point_num;
}
//===================================================================================

//沿法线方向缩放点云=================================================================
void PC_ScalePCBaseNormal(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &normal_p, PC_XYZ::Ptr &dstPC, float scale)
{
	int length = srcPC->points.size();
	dstPC->points.resize(length);
	if (normal_p->points.size() != length || dstPC->points.size() != length)
		return;
	for (int i = 0; i < length; ++i)
	{
		if (isnan(normal_p->points[i].x))
		{
			dstPC->points[i].x = NAN;
		}
		else
		{
			dstPC->points[i].x = srcPC->points[i].x + normal_p->points[i].x *  scale;
			dstPC->points[i].y = srcPC->points[i].y + normal_p->points[i].y *  scale;
			dstPC->points[i].z = srcPC->points[i].z + normal_p->points[i].z *  scale;
		}
	}
}
//===================================================================================

//将点云投影到XY二维平面=============================================================
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
//===================================================================================

//计算点云的法向量===================================================================
void PC_ComputePCNormal(PC_XYZ::Ptr &srcPC, PC_N::Ptr &normals, float radius)
{
	if (srcPC->empty())
		return;
	pcl::NormalEstimation<P_XYZ, pcl::Normal> normal_est;
	normal_est.setInputCloud(srcPC);
	normal_est.setRadiusSearch(radius);
	pcl::search::KdTree<P_XYZ>::Ptr kdtree(new pcl::search::KdTree<P_XYZ>);
	normal_est.setSearchMethod(kdtree);
	normal_est.compute(*normals);
}
//===================================================================================

//叉乘计算法向量=====================================================================
void PC_MulCrossCalNormal(P_XYZ &p_1, P_XYZ &p_2, P_XYZ &normal_p)
{
	normal_p.x = p_1.y * p_2.z - p_1.z * p_2.y;
	normal_p.y = p_1.z * p_2.x - p_1.x * p_2.z;
	normal_p.z = p_1.x * p_2.y - p_1.y * p_2.x;
	float norm_ = std::sqrtf(normal_p.x*normal_p.x + normal_p.y*normal_p.y + normal_p.z * normal_p.z);
	normal_p.x /= norm_;
	normal_p.y /= norm_;
	normal_p.z /= norm_;
}
//===================================================================================

//计算点云的协方差矩阵===============================================================
void PC_ComputeCovMat(PC_XYZ::Ptr& pc, Mat& covMat, P_XYZ& gravity)
{
	if (pc->empty())
		return;
	if (covMat.size() != cv::Size(3, 3))
		covMat = Mat(cv::Size(3, 3), CV_32FC1, cv::Scalar(0));
	int point_num = pc->points.size();

	PC_GetPCGravity(pc, gravity);

	vector<float> ori_x(point_num), ori_y(point_num), ori_z(point_num);
	for (int i = 0; i < point_num; ++i)
	{
		P_XYZ& p_ = pc->points[i];
		ori_x[i] = p_.x - gravity.x;
		ori_y[i] = p_.y - gravity.y;
		ori_z[i] = p_.z - gravity.z;
	}

	float *pCovMat = covMat.ptr<float>(0);
	for (int i = 0; i < point_num; ++i)
	{
		pCovMat[0] += (ori_x[i] * ori_x[i]);
		pCovMat[4] += (ori_y[i] * ori_y[i]);
		pCovMat[8] += (ori_z[i] * ori_z[i]);

		pCovMat[1] += (ori_x[i] * ori_y[i]);
		pCovMat[5] += (ori_y[i] * ori_z[i]);
		pCovMat[2] += (ori_z[i] * ori_x[i]);
	}
	pCovMat[3] = pCovMat[1];
	pCovMat[6] = pCovMat[2];
	pCovMat[7] = pCovMat[5];
}
//===================================================================================
