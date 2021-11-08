#include "TestFitShape.h"
#include "DrawShape.h"

#include "Img_FitEllipse.h"

//#include "PC_FitLine.h"
//#include "PC_FitLine.cpp"

//#include "PC_FitPlane.h"
//#include "PC_FitPlane.cpp"

//#include "PC_FitCircle.h"
//#include "PC_FitCircle.cpp"
//
//#include "PC_FitSphere.h"
//#include "PC_FitSphere.cpp"


//空间三维圆拟合测试==========================================================================
//void PC_FitCircleTest()
//{
//	PC_XYZ::Ptr srcPC(new PC_XYZ);
//	pcl::io::loadPLYFile("C:/Users/Administrator/Desktop/testimage/八分之一噪声圆.ply", *srcPC);
//
//	vector<P_XYZ> pts(srcPC->points.size());
//	for (int i = 0; i < srcPC->points.size(); ++i)
//	{
//		pts[i] = srcPC->points[i];
//	}
//	std::random_shuffle(pts.begin(), pts.end());
//	vector<double> circle(7, 0.0);
//	vector<P_XYZ> inlinerPts;
//	PC_RANSACFitCircle(pts, circle, inlinerPts, 0.2);
//
//	pcl::visualization::PCLVisualizer viewer;
//	viewer.addCoordinateSystem(10);
//	//显示轨迹
//	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC, 255, 0, 0); //设置点云颜色
//	viewer.addPointCloud(srcPC, red, "srcPC");
//	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "srcPC");
//
//	while (!viewer.wasStopped())
//	{
//		viewer.spinOnce();
//	}
//}
//============================================================================================

