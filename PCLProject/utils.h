#pragma once

/*OpenCV版本：4.5.3
  PCL版本：1.9.1
*/

#include <iostream>
#include <string>
#include <conio.h>
#include <vector>
#include <random> 

#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/kdtree/kdtree_flann.h>  
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>

#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

#include <pcl/common/transforms.h>
#include <pcl/common/common.h>  
#include <pcl/sample_consensus/model_types.h>
#include <pcl/registration/icp.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#ifdef NB
#define NB_API _declspec(dllexport )
#endif

const double EPS = 1e-8;

enum NB_MODEL_FIT_METHOD {
	OLS_FIT = 0,
	HUBER_FIT = 1,
	TUKEY_FIT = 2
};

using namespace std;
using namespace pcl;
using namespace cv;

typedef pcl::PointCloud<pcl::PointXYZ> PC_XYZ;
typedef pcl::PointXYZ P_XYZ;
typedef pcl::PointCloud<pcl::PointNormal> PC_XYZN;
typedef pcl::PointCloud<pcl::Normal> PC_N;
typedef pcl::PointNormal P_XYZN;
typedef pcl::PointCloud<pcl::PointXYZI> PC_XYZI;
typedef pcl::PointXYZI P_XYZI;
typedef pcl::Normal P_N;
typedef pcl::PointIndices P_IDX;

//直线
typedef struct Line2D
{
	double a;
	double b;
	double c;
}Line2D;

//圆
typedef struct Circle2D
{
	double x;
	double y;
	double r;
}Circle2D;

//椭圆-----中心坐标：(x, y)，与x轴夹角：angle，长短轴：(a, b)
typedef struct Ellipse2D
{
	double x;
	double y;
	double angle;
	double a;
	double b;
}Ellipse2D;

//平面
typedef struct Plane3D
{
	double a;
	double b;
	double c;
	double d;
}Plane3D;

//空间园----中心坐标：(x, y, z)， 半径：r，原所在平面法向量：(a, b, c)
typedef struct Circle3D
{
	double x;
	double y;
	double z;
	double r;
	double a;
	double b;
	double c;
}Circle3D;

//球
typedef struct Sphere3D
{
	double x;
	double y;
	double z;
	double r;
}Sphere3D;

//空间直线----直线的方向向量：(a, b, c)，直线上的坐标：(x, y, z)
typedef struct Line3D
{
	double a;
	double b;
	double c;
	double x;
	double y;
	double z;
}Line3D;

class NB_Array2D
{
public:
	template<typename _Tp> NB_Array2D(const _Tp& vec)
	{
		m_Size = vec.size();
		m_vData.resize(m_Size);
		//这个地方弄成传指针，而不是拷贝数据（差评）
		for (int i = 0; i < m_Size; ++i)
		{
			m_vData[i].x = vec[i].x; m_vData[i].y = vec[i].y;
		}
	}
	Point2d& operator[](int n)
	{
		return m_vData[n];
	}
	int size()
	{
		return m_Size;
	}

private:
	vector<Point2d> m_vData;
	int m_Size;
};

class NB_Array3D
{
public:
	template<typename _Tp> NB_Array3D(const _Tp& vec)
	{
		m_Size = vec.size();
		m_vData.resize(m_Size);
		//这个地方弄成传指针，而不是拷贝数据（差评）
		for (int i = 0; i < m_Size; ++i)
		{
			m_vData[i].x = vec[i].x; m_vData[i].y = vec[i].y; m_vData[i].z = vec[i].z;
		}
	}
	Point3d& operator[](int n)
	{
		return m_vData[n];
	}
	int size()
	{
		return m_Size;
	}

private:
	vector<Point3d> m_vData;
	int m_Size;
};
