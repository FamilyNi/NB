#include "PC_Filter.h"

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


void PC_FitlerTest()
{
	float v_x = 0; 
	float v_y = 0;
	float v_z = 1;

	PC_XYZ::Ptr srcPC(new PC_XYZ);
	string path = "G:/JC_Config/整体点云/样品2/PC.ply";
	pcl::io::loadPLYFile(path, *srcPC);
	PC_XYZ::Ptr dstPC(new PC_XYZ);

	PC_XYZ::Ptr v_srcPC(new PC_XYZ);
	PC_VoxelGrid(srcPC, v_srcPC, 1.6f);

	PC_ProjectFilter(v_srcPC, dstPC, v_x, v_y, v_z);

	pcl::visualization::PCLVisualizer viewer;
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> white(v_srcPC, 255, 255, 255);
	viewer.addPointCloud(v_srcPC, white, "v_srcPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "v_srcPC");
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(dstPC, 255, 0, 0);
	viewer.addPointCloud(dstPC, red, "dstPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dstPC");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}