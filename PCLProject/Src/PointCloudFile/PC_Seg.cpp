#include "../../include/PointCloudFile/PC_Filter.h"
#include "../../include/PointCloudFile/PointCloudOpr.h"
#include "../../include/PointCloudFile/PC_Seg.h"

//�������һ���Եĵ��Ʒָ�==========================================================
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

//����ŷʽ����ָ��==============================================================
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

//DBSCAN�ָ�==========================================================================
void PC_DBSCANSeg(PC_XYZ::Ptr& srcPC, vector<vector<int>>& indexs,
	double radius, int n, int minGroup, int maxGroup)
{
	int length = srcPC->points.size();

	//-1��ʾ�õ��Ѿ����ࡢ0��ʾ�õ�Ϊ�������Ҳ�Ϊ���ĵ㡢1��ʾ�õ�Ϊ���ĵ�---ֻ�к��ĵ���ܳ�Ϊ���ӵ�
	vector<int> isLabeled(length, 0);
	pcl::KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC);
	//�������еĵ��Ϊ������Ǻ���
#pragma omp parallel for
	for (int i = 0; i < length; ++i)
	{
		vector<int> PIdx;
		vector<float> DistIdx;
		kdtree.radiusSearch(srcPC->points[i], radius, PIdx, DistIdx);
		if (PIdx.size() > n)
		{
			isLabeled[i] = 1;
		}
	}

	//����
	queue<int> sands;
	for (int i = 0; i < isLabeled.size(); ++i)
	{
		if (isLabeled[i] == 1)
		{
			sands.push(i);
		}
		else
			continue;
		vector<int> index(0);
		while (!sands.empty())
		{
			vector<int> PIdx;
			vector<float> DistIdx;
			kdtree.radiusSearch(srcPC->points[sands.front()], radius, PIdx, DistIdx);
			for (int i = 0; i < PIdx.size(); ++i)
			{
				int idx = PIdx[i];
				if (isLabeled[idx] > -1)
				{
					index.push_back(idx);
					if (isLabeled[idx] == 1)
					{
						sands.push(idx);
					}
					isLabeled[idx] = -1;
				}
			}
			sands.pop();
		}
		if (index.size() > minGroup && index.size() < maxGroup)
			indexs.push_back(index);
	}
}
//===================================================================================

//Different Of Normal�ָ�============================================================
void PC_DONSeg(PC_XYZ::Ptr &srcPC, vector<int>& indexs, double large_r, double small_r, double thresVal)
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

	indexs.reserve(length);
	for (size_t i = 0; i < length; ++i)
	{
		P_N l_pn = normals_large_scale->points[i];
		P_N s_pn = normals_small_scale->points[i];
		float diff_x = std::fabs(abs(l_pn.normal_x) - abs(s_pn.normal_x)) * 0.5;
		float diff_y = std::fabs(abs(l_pn.normal_y) - abs(s_pn.normal_y)) * 0.5;
		float diff_z = std::fabs(abs(l_pn.normal_z) - abs(s_pn.normal_z)) * 0.5;
		if (diff_x > thresVal || diff_y > thresVal || diff_z > thresVal)
		{
			indexs.push_back(i);
		}
	}
}
//===================================================================================

//����ƽ��ָ�=======================================================================
void PC_SegBaseOnPlane(PC_XYZ::Ptr& srcPC, Plane3D& plane, vector<int>& index, double thresVal, int orit)
{
	index.reserve(srcPC->points.size());
	for (int i = 0; i < srcPC->points.size(); ++i)
	{
		P_XYZ& p = srcPC->points[i];
		float dist = p.x * plane.a + p.y * plane.b + p.z * plane.c + plane.d;
		if (dist < thresVal && orit == 0)
		{
			index.push_back(i);
		}
		if (dist > thresVal && orit == 1)
		{
			index.push_back(i);
		}
	}
}
//===================================================================================

//�������ʷָ���Ʒָ�===============================================================
void PC_CurvatureSeg(PC_XYZ::Ptr &srcPC, PC_N::Ptr& normals, vector<int>& indexs, double H_Thres, double L_Thres)
{
	int pts_nun = srcPC->points.size();
	indexs.reserve(pts_nun);
	for (int i = 0; i < pts_nun; ++i)
	{
		double curvature = normals->points[i].curvature;
		if (curvature > L_Thres && curvature < H_Thres)
		{
			indexs.push_back(i);
		}
	}
}
//===================================================================================

/*���Ʒָ���Գ���*/
void PC_SegTest()
{
	PC_XYZ::Ptr srcPC(new PC_XYZ);
	string path = "C:/Users/Administrator/Desktop/testimage/���1.ply";
	pcl::io::loadPLYFile(path, *srcPC);
	PC_XYZ::Ptr downSrcPC(new PC_XYZ);
	PC_VoxelGrid(srcPC, downSrcPC, 0.2);
	vector<vector<int>> indexs;
	PC_DBSCANSeg(downSrcPC, indexs, 0.5, 5, 0, 10000000);

	//PC_XYZ::Ptr dstPC(new PC_XYZ);
	//PC_ExtractIdxPC(downSrcPC, dstPC, indexs[3]);

	for (int i = 0; i < indexs.size(); ++i)
	{
		PC_XYZ::Ptr dstPC(new PC_XYZ);
		PC_ExtractPC(downSrcPC, indexs[i], dstPC);
		pcl::visualization::PCLVisualizer viewer;
		viewer.addCoordinateSystem(10);
		//��ʾ�켣
		pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(downSrcPC, 255, 0, 0); //���õ�����ɫ
		viewer.addPointCloud(downSrcPC, red, "downSrcPC");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "downSrcPC");

		pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> write(dstPC, 255, 255, 255); //���õ�����ɫ
		viewer.addPointCloud(dstPC, write, "dstPC");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "dstPC");
		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
		}
	}

	//PC_XYZ::Ptr v_srcPC(new PC_XYZ);
	//PC_VoxelGrid(srcPC, v_srcPC, 1.6f);
	//float large_r = 20.0f;
	//float small_r = 2.0f;
	//float thresVal = 0.86f;
	//DONSeg(v_srcPC, large_r, small_r, thresVal);
}