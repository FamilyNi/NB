#pragma once

/*OpenCV°æ±¾£º4.5.3
  PCL°æ±¾£º1.9.1
*/

#include <iostream>
#include <string>
#include <conio.h>
#include <vector>

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
	TURKEY_FIT = 2
};

struct Plane3D
{
	double a;
	double b;
	double c;
	double d;
	Plane3D() :a(0), b(0), c(0), d(0)
	{}
};

struct Sphere
{
	double c_x;
	double c_y;
	double c_z;
	double r;
	Sphere() :c_x(0), c_y(0), c_z(0), r(0)
	{}
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
