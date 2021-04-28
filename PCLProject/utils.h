#pragma once
#include <iostream>
#include <string>
#include <conio.h>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/kdtree/kdtree_flann.h>  
#include <pcl/kdtree/flann.h>  
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/flann_search.h>  
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/console/time.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>  
#include <pcl/sample_consensus/model_types.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

const float EPS = 1e-8f;

typedef struct
{
	double a;
	double b;
	double c;
	double d;
}Plane3D;

using namespace std;
using namespace pcl;

typedef pcl::PointCloud<pcl::PointXYZ> PC_XYZ;
typedef pcl::PointXYZ P_XYZ;
typedef pcl::PointCloud<pcl::PointNormal> PC_XYZN;
typedef pcl::PointCloud<pcl::Normal> PC_N;
typedef pcl::PointNormal P_XYZN;
typedef pcl::PointCloud<pcl::PointXYZI> PC_XYZI;
typedef pcl::PointXYZI P_XYZI;
typedef pcl::Normal P_N;


inline void ReadPointCloud(string &path, PC_XYZ::Ptr &cloud)
{
	if (pcl::io::loadPLYFile<P_XYZ>(path, *cloud) == -1) {
		PCL_ERROR("Couldnot read file.\n");
		system("pause");
		return;
	}
}

//ʮ����ת������
inline void DecToBin(const int dec_num, vector<bool>& bin)
{
	int a = dec_num;
	int index = 0;
	int length = bin.size() - 1;
	while (a != 0)
	{
		if (index > length)
			break;
		bin[index] = a % 2;
		a /= 2;
		index++;
	}
}

//������תʮ����
inline void BinToDec(const vector<bool>& bin, int& dec_num)
{
	dec_num = 0;
	for (size_t i = 0; i < bin.size(); ++i)
	{
		dec_num += bin[i] * std::pow(2, (int)i);
	}
}

//������ת������
inline void BinToGrayCode(const vector<bool>& bin, vector<bool>& grayCode)
{
	int len = bin.size();
	grayCode.resize(len);
	for (int i = 0; i < len - 1; ++i)
	{
		grayCode[i] = bin[i] ^ bin[i + 1];
	}
	grayCode[len - 1] = bin[len - 1];
}

//������ת������
inline void GrayCodeToBin(const vector<bool>& grayCode, vector<bool>& bin)
{
	int len = grayCode.size();
	bin.resize(len);
	bin[len - 1] = grayCode[len - 1];
	for (int i = len-2; i > -1; --i)
	{
		bin[i] = grayCode[i] ^ bin[i + 1];
	}
}