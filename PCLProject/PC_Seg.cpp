#include "PC_Seg.h"
#include "PC_Filter.h"

//随机采样一致性的点云分割==========================================================
void PC_RANSACSeg(PC_XYZ::Ptr &srcPC, PC_XYZ::Ptr &dstPC, int mode, float thresVal)
{
	if (mode > 16)
		return;
	if (srcPC->empty())
		return;

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

	pcl::SACSegmentation<P_XYZ> seg;

	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

	seg.setOptimizeCoefficients(true);
	seg.setModelType(mode);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(1000);
	seg.setDistanceThreshold(thresVal);
	seg.setInputCloud(srcPC);
	seg.segment(*inliers, *coefficients);

	pcl::ExtractIndices<P_XYZ> extract;
	extract.setInputCloud(srcPC);
	extract.setIndices(inliers);
	extract.setNegative(false);
	extract.filter(*dstPC);
}
//==================================================================================

//基于欧式距离分割方法==============================================================
void PC_EuclideanSeg(PC_XYZ::Ptr &srcPC, std::vector<P_IDX> clusters, float distThresVal)
{
	if (srcPC->empty())
		return;
	pcl::search::KdTree<P_XYZ>::Ptr kdtree(new pcl::search::KdTree<P_XYZ>);
	kdtree->setInputCloud(srcPC);
	pcl::EuclideanClusterExtraction<P_XYZ> clustering;
	clustering.setClusterTolerance(distThresVal);
	clustering.setMinClusterSize(1);
	clustering.setMaxClusterSize(10000000);
	clustering.setSearchMethod(kdtree);
	clustering.setInputCloud(srcPC);
	clustering.extract(clusters);
}
//==================================================================================

//区域生长实现点云分割==============================================================
void PC_RegionGrowing(PC_XYZ::Ptr &srcPC, std::vector<vector<uint>> &indexs, float radius)
{
	size_t length = srcPC->points.size();
	vector<bool> isLabeled(length, 0);

	pcl::KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC);

	vector<bool> isSand(length, false);
	for (size_t i = 0; i < length; ++i)
	{
		if (isSand[i])
		{
			continue;
		}
		queue<uint> local_sandp;
		vector<uint> cluster(0);
		cluster.push_back(i);
		local_sandp.push(i);
		isLabeled[i] = true;
		isSand[i] = true;
		float ref_z = srcPC->points[i].z;
		while (!local_sandp.empty())
		{
			P_XYZ& ref_p = srcPC->points[local_sandp.front()];
			vector<int> PIdx(0);
			vector<float> DistIdx(0);
			kdtree.radiusSearch(ref_p, radius, PIdx, DistIdx);
			for (size_t j = 1; j < PIdx.size(); ++j)
			{
				float z_ = abs(ref_z - srcPC->points[PIdx[j]].z);
				if (z_ < 2.0f)
				{
					if (!isLabeled[PIdx[j]])
					{
						cluster.push_back(PIdx[j]);
						isLabeled[PIdx[j]] = true;
					}
					if (!isSand[PIdx[j]])
					{
						local_sandp.push(PIdx[j]);
						isSand[PIdx[j]] = true;
					}
				}
			}
			local_sandp.pop();
		}
		indexs.push_back(cluster);

		PC_XYZ::Ptr dstPC(new PC_XYZ);
		dstPC->points.resize(cluster.size());
		for (int m = 0; m < cluster.size(); ++m)
		{
			dstPC->points[m] = srcPC->points[cluster[m]];
		}
	}
	uint sum_ = 0;
	for (size_t i = 0; i < indexs.size(); ++i)
	{
		sum_ += indexs[i].size();
	}
}
//====================================================================================

//DBSCAN分割==========================================================================
void DBSCANSeg(PC_XYZ::Ptr &srcPC, vector<vector<uint>> &indexs, float radius, int p_number)
{
	size_t length = srcPC->points.size();
	vector<bool> isCore(length, false);
	vector<bool> isLabeled(length, false);

	pcl::KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC);
	//将点云中的点分为核心与非核心
	vector<uint> nonCore;
	for (size_t i = 0; i < length; ++i)
	{
		vector<int> PIdx;
		vector<float> DistIdx;
		kdtree.radiusSearch(srcPC->points[i], radius, PIdx, DistIdx);
		if (PIdx.size() > p_number)
			isCore[i] = true;
	}

	//对核心点进行聚类
	vector<bool> isSand(length, false);
	for (size_t i = 0; i < length; ++i)
	{
		if (isSand[i])
			continue;
		if (isCore[i])
		{
			queue<uint> sand_p;
			vector<uint> index;
			sand_p.push(i);
			isSand[i] = true;
			index.push_back(i);
			while (!sand_p.empty())
			{
				vector<int> PIdx;
				vector<float> DistIdx;
				kdtree.radiusSearch(srcPC->points[sand_p.front()], radius, PIdx, DistIdx);
				for (size_t j = 1; j < PIdx.size(); ++j)
				{
					if (isCore[PIdx[j]] && !isSand[PIdx[j]])
					{
						sand_p.push(PIdx[j]);
						isSand[PIdx[j]] = true;
					}
					if (!isLabeled[PIdx[j]])
					{
						index.push_back(PIdx[j]);
						isLabeled[PIdx[j]] = true;
					}
				}
				sand_p.pop();
			}
			indexs.push_back(index);
		}
	}
	//对噪声进行聚类
	for (size_t i = 0; i < length; ++i)
	{
		if (isLabeled[i])
			continue;
		queue<uint> sand_p;
		vector<uint> index;
		sand_p.push(i);
		index.push_back(i);
		while (!sand_p.empty())
		{
			vector<int> PIdx;
			vector<float> DistIdx;
			kdtree.radiusSearch(srcPC->points[sand_p.front()], radius, PIdx, DistIdx);
			for (size_t j = 0; j < PIdx.size(); ++j)
			{
				sand_p.push(PIdx[j]);
				if (!isLabeled[PIdx[j]])
				{
					index.push_back(PIdx[j]);
					isLabeled[i] = true;
				}
			}
			sand_p.pop();
		}
		indexs.push_back(index);
	}
}
//===================================================================================

//Different Of Normal分割============================================================
void DONSeg(PC_XYZ::Ptr &srcPC, float large_r, float small_r, float thresVal)
{
	size_t length = srcPC->points.size();
	pcl::search::Search<P_XYZ>::Ptr tree;
	pcl::NormalEstimation<P_XYZ, P_N> ne;
	ne.setInputCloud(srcPC);
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(small_r);
	PC_N::Ptr normals_small_scale(new PC_N);
	ne.compute(*normals_small_scale);

	PC_N::Ptr normals_large_scale(new PC_N);
	ne.setRadiusSearch(large_r);
	ne.compute(*normals_large_scale);

	PC_XYZ::Ptr dstPC(new PC_XYZ);
	for (size_t i = 0; i < length; ++i)
	{
		P_N l_pn = normals_large_scale->points[i];
		P_N s_pn = normals_small_scale->points[i];
		float diff_x = std::fabs(abs(l_pn.normal_x) - abs(s_pn.normal_x)) * 0.5;
		float diff_y = std::fabs(abs(l_pn.normal_y) - abs(s_pn.normal_y)) * 0.5;
		float diff_z = std::fabs(abs(l_pn.normal_z) - abs(s_pn.normal_z)) * 0.5;
		if (diff_x > thresVal || diff_y > thresVal || diff_z > thresVal)
		{
			dstPC->points.push_back(srcPC->points[i]);
		}
	}
	pcl::visualization::PCLVisualizer viewer;
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> white(srcPC, 255, 255, 255);
	viewer.addPointCloud(srcPC, white, "srcPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "srcPC");
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(dstPC, 255, 0, 0);
	viewer.addPointCloud(dstPC, red, "dstPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dstPC");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
//===================================================================================

/*点云分割测试程序*/
void PC_SegTest()
{
	PC_XYZ::Ptr srcPC(new PC_XYZ);
	string path = "G:/JC_Config/整体点云/样品2/PC.ply";
	pcl::io::loadPLYFile(path, *srcPC);
	PC_XYZ::Ptr v_srcPC(new PC_XYZ);
	PC_VoxelGrid(srcPC, v_srcPC, 1.6f);
	float large_r = 20.0f;
	float small_r = 2.0f;
	float thresVal = 0.86f;
	DONSeg(v_srcPC, large_r, small_r, thresVal);
}