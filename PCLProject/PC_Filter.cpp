#include "PC_Filter.h"

//体素滤波---下采样==================================================================
int PC_VoxelGrid(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, float leafSize)
{
	if (srcPC->empty())
		return 1;
	VoxelGrid<P_XYZ> vg;
	vg.setInputCloud(srcPC);
	vg.setLeafSize(leafSize, leafSize, leafSize);
	vg.filter(*dstPC);
	if (dstPC->empty())
		return 2;
	return 0;
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

//邻域密度约束的动态标准差阈值离群点检测模型=========================================
void NeighbourMove(PC_XYZ::Ptr &srcPC, vector<uint> &index_p, uint k, float outcoef, float incoef)
{
	size_t length = srcPC->points.size();
	vector<float> r_(length, 0.0f);
	vector<float> meanDist(length, 0.0f);
	pcl::KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC);

	float sum_r = 0.0f, sum_dist = 0.0f;
	for (size_t i = 0; i < length; ++i)
	{
		vector<int> PIdx;
		vector<float> DistIdx;
		P_XYZ& ref_p = srcPC->points[i];
		kdtree.nearestKSearch(ref_p, k, PIdx, DistIdx);
		size_t len_ = PIdx.size();
		//求每个点邻域的平均距离、重心
		float sum_dist = 0.0f, sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;;
		for (size_t j = 0; j < len_; ++j)
		{
			sum_dist += std::sqrt(DistIdx[j]);
			P_XYZ p_ = srcPC->points[PIdx[j]];
			sum_x += p_.x; sum_y += p_.y; sum_z += p_.z;
		}
		meanDist[i] = sum_dist / len_;
		float c_pi_x = sum_x / len_;
		float c_pi_y = sum_y / len_;
		float c_pi_z = sum_z / len_;
		//求调整后的重心
		float sum_adjx = 0.0f, sum_adjy = 0.0f, sum_adjz = 0.0f, adj_coeff = 0.0f;
		for (size_t j = 0; j < len_; ++j)
		{
			P_XYZ& p_ = srcPC->points[PIdx[j]];
			float diff_x = p_.x - c_pi_x;
			float diff_y = p_.y - c_pi_y;
			float diff_z = p_.z - c_pi_z;
			float dist = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
			float coeff = std::expf(dist / 2.0f);
			sum_adjx += coeff * p_.x; sum_adjy += coeff * p_.y; sum_adjz += coeff * p_.z;
			adj_coeff += coeff;
		}
		float c_adj_x = sum_adjx / adj_coeff - c_pi_x;
		float c_adj_y = sum_adjy / adj_coeff - c_pi_y;
		float c_adj_z = sum_adjz / adj_coeff - c_pi_z;
		//计算点云密度
		for (size_t j = 0; j < len_; ++j)
		{
			P_XYZ& p_ = srcPC->points[PIdx[j]];
			float diff_x = p_.x - c_adj_x;
			float diff_y = p_.y - c_adj_y;
			float diff_z = p_.z - c_adj_z;
			float dist = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
			r_[i] = r_[i] > dist ? r_[i] : dist;
		}
		r_[i] = 1 / (std::max(r_[i] * std::sqrt(r_[i]), 1e-8f));
		sum_r += r_[i];
		sum_dist += meanDist[i];
	}
	//计算平均密度、平均距离
	float mean_dist = sum_dist / length;
	float mean_r = sum_r / length;
	//计算方差
	float delta = 0.0f;
	vector<float> diff_dist_(length, 0);
	for (size_t i = 0; i < length; ++i)
	{
		diff_dist_[i] = fabs(meanDist[i] - mean_dist);
		delta += diff_dist_[i] * diff_dist_[i];
	}
	delta = std::sqrt(delta / (length - 1));

	//进行滤波
	for (size_t i = 0; i < length; ++i)
	{
		float r_1 = mean_r - r_[i];
		float r_2 = mean_r + r_[i];
		float l_out = outcoef - r_1 / r_2;
		float l_in = incoef - abs(r_1) / (r_2);
		if (diff_dist_[i] < l_out * delta/* && diff_dist_[i] < l_in * delta*/)
			index_p.push_back(i);
	}

	PC_XYZ::Ptr dstPC(new PC_XYZ);
	dstPC->points.resize(index_p.size());
	for (int m = 0; m < index_p.size(); ++m)
	{
		dstPC->points[m] = srcPC->points[index_p[m]];
	}
	//pcl::visualization::PCLVisualizer viewer;
	////显示轨迹
	//pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> white(dstPC, 255, 255, 255); //设置点云颜色
	//viewer.addPointCloud(dstPC, white, "dstPC");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "dstPC");
	//pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC, 255, 0, 0); //设置点云颜色
	//viewer.addPointCloud(srcPC, red, "srcPC");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "srcPC");
	//while (!viewer.wasStopped())
	//{
	//	viewer.spinOnce();
	//}
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
	ReadPointCloud(path, srcPC);
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