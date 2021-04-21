#include "PC_Seg.h"

//区域生长实现点云胡分割==============================================================
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
		pcl::visualization::PCLVisualizer viewer;
		//显示轨迹
		pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> white(dstPC, 255, 255, 255); //设置点云颜色
		viewer.addPointCloud(dstPC, white, "dstPC");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "dstPC");
		pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC, 255, 0, 0); //设置点云颜色
		viewer.addPointCloud(srcPC, red, "srcPC");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "srcPC");
		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
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
void DBSCANSeg(PC_XYZ::Ptr &srcPC, vector<vector<uint>> &indexs, float radius, uint p_number)
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

/*点云分割测试程序*/
void PC_SegTest()
{
}