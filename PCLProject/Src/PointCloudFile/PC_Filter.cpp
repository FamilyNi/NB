#include "../../include/PointCloudFile/PC_Filter.h"

//体素滤波---下采样==================================================================
void PC_VoxelGrid(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, float leafSize)
{
	if (srcPC->empty())
		return;
	VoxelGrid<P_XYZ> vg;
	vg.setInputCloud(srcPC);
	vg.setLeafSize(leafSize, leafSize, leafSize);
	vg.filter(*dstPC);
	if (dstPC->empty())
		return;
}
//===================================================================================

//直通滤波===========================================================================
void PC_PassFilter(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, const string mode, double minVal, double maxVal)
{
	if (srcPC->empty())
		return;
	if (mode != "x" && mode != "y" && mode != "z")
		return;
	PassThrough<P_XYZ> pt;
	pt.setInputCloud(srcPC);
	pt.setFilterFieldName(mode);
	pt.setFilterLimits(minVal, maxVal);
	pt.filter(*dstPC);
}
//===================================================================================

//半径剔除===========================================================================
void PC_RadiusOutlierRemoval(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, double radius, int minNeighborNum)
{
	if (srcPC->empty())
		return;
	RadiusOutlierRemoval<P_XYZ> ror;
	ror.setInputCloud(srcPC);
	ror.setRadiusSearch(radius);
	ror.setMinNeighborsInRadius(minNeighborNum);
	ror.filter(*dstPC);
}
//===================================================================================

//移动最小二乘法=====================================================================
void PC_MLSFilter(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, float radius, int order)
{
	if (srcPC->points.size() == 0)
		return;
	pcl::search::KdTree<P_XYZ>::Ptr tree(new pcl::search::KdTree<P_XYZ>);
	pcl::MovingLeastSquares<P_XYZ, P_XYZ> mls;
	mls.setComputeNormals(false);
	mls.setInputCloud(srcPC);
	mls.setPolynomialFit(true);
	mls.setPolynomialOrder(order);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(radius);
	mls.process(*dstPC);
}
//===================================================================================

//平面投影滤波=======================================================================
void PC_ProjectFilter(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, float v_x, float v_y, float v_z)
{
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	coefficients->values.resize(4);
	coefficients->values[0] = v_x;
	coefficients->values[1] = v_y;
	coefficients->values[2] = v_z;
	coefficients->values[3] = 10;

	pcl::ProjectInliers<P_XYZ> proj;
	proj.setModelType(pcl::SACMODEL_SPHERE);
	proj.setInputCloud(srcPC);
	proj.setModelCoefficients(coefficients);
	proj.filter(*dstPC);
}
//===================================================================================

//导向滤波===========================================================================
void PC_GuideFilter(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& dstPC, double radius, double lamda)
{
	KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC);
	int pts_num = srcPC->points.size();
	dstPC->points.resize(pts_num);
	for (int i = 0; i < pts_num; ++i)
	{
		vector<int> PIdx;
		vector<float> PDist;
		P_XYZ& src_p = srcPC->points[i];
		kdtree.radiusSearch(srcPC->points[i], radius, PIdx, PDist);
		P_XYZ& dst_p = dstPC->points[i];
		if (PIdx.size() > 1)
		{
			float inv_ = 1.0f / (float)PIdx.size();
			float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
			float sum_xx = 0.0f, sum_yy = 0.0f, sum_zz = 0.0f;
			for (int j = 0; j < PIdx.size(); ++j)
			{
				P_XYZ& p_ = srcPC->points[PIdx[j]];
				sum_x += p_.x; sum_y += p_.y; sum_z += p_.z;
				sum_xx = p_.x * p_.x; sum_yy = p_.y * p_.y; sum_zz = p_.z * p_.z;
			}
			float mean_x = sum_x * inv_, mean_xx = sum_xx * inv_;
			float mean_y = sum_y * inv_, mean_yy = sum_yy * inv_;
			float mean_z = sum_z * inv_, mean_zz = sum_zz * inv_;
			float a_x = mean_xx - mean_x * mean_x;
			float a_y = mean_yy - mean_y * mean_y;
			float a_z = mean_zz - mean_z * mean_z;
			a_x /= (a_x + lamda); a_y /= (a_y + lamda); a_z /= (a_z + lamda);
			float b_x = mean_x - a_x * mean_x;
			float b_y = mean_y - a_y * mean_y;
			float b_z = mean_z - a_z * mean_z;

			dst_p = { a_x * src_p.x + b_x, a_y * src_p.y + b_y, a_z * src_p.z + b_z };
		}
		else
		{
			dst_p = src_p;
		}
	}
}
//===================================================================================


void PC_FitlerTest()
{
	string pc_rotpath = "C:/Users/Administrator/Desktop/testimage/噪声球.ply";
	PC_XYZ::Ptr srcPC(new PC_XYZ);
	pcl::io::loadPLYFile(pc_rotpath, *srcPC);

	//PC_XYZ::Ptr v_srcPC(new PC_XYZ);
	//PC_VoxelGrid(srcPC, v_srcPC, 0.5f);

	PC_XYZ::Ptr dstPC(new PC_XYZ);
	PC_GuideFilter(srcPC, dstPC, 7, 0.5);

	pcl::visualization::PCLVisualizer viewer;
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> white(srcPC, 255, 255, 255);
	viewer.addPointCloud(srcPC, white, "srcPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "srcPC");
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(dstPC, 255, 0, 0);
	viewer.addPointCloud(dstPC, red, "dstPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dstPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}