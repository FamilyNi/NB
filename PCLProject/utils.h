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

const float EPS = 1e-8f;

typedef struct
{
	double a;
	double b;
	double c;
	double d;
}Plane3D;

typedef struct
{
	double c_x;
	double c_y;
	double c_z;
	double r;
}Sphere;


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
