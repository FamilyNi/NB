// PCLProject.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "utils.h"
#include "PC_Filter.h"
#include "PC_Seg.h"
#include "JC_Calibrate.h"
#include "PPFMatch.h"
#include "GrayCode.h"
#include "WaveLet.h"
#include "LocalDeforableModel.h"
#include "DrawShape.h"
#include "MathOpr.h"
#include "SiftMatch.h"
#include "ContourOpr.h"
#include "ShapeMatch.h"
#include "LBPfeatures.h"
#include "ImageFilter.h"
#include "ImageEnhance.h"
#include "ImageSeg.h"
#include "GrabEdges.h"
#include "ComputeLine.h"
#include "ComputeCircle.h"
#include "ComputeSphere.h"
#include "ComputePlane.h"
#include "Compute3DLine.h"

int main(int argc, char *argv[])
{
	PC_3DLineTest();

	PC_XYZ::Ptr linePC(new PC_XYZ);
	cv::Vec6d line(5, 15, 7.65, 6.25, 3.65, 7.65);
	PC_DrawLine(linePC, -10, 25, -15, 15, 21, 41, line, 3);

	vector<cv::Point3f> pts(linePC->points.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		pts[i] = cv::Point3f(linePC->points[i].x, linePC->points[i].y, linePC->points[i].z);
	}
	cv::Vec6f line_;
	cv::fitLine(pts, line_, cv::DIST_L2, 0, 0.01, 0.01);

	PC_XYZ::Ptr noisePC(new PC_XYZ);
	PC_AddNoise(linePC, noisePC, 3, 3);
	pcl::io::savePLYFile("C:/Users/Administrator/Desktop/testimage/噪声直线.ply", *noisePC);
	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> write(noisePC, 255, 255, 255); //设置点云颜色
	viewer.addPointCloud(noisePC, write, "linePC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "linePC");

	//pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(noise, 255, 0, 0); //设置点云颜色
	//viewer.addPointCloud(noise, red, "noise");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "noise");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}

	//ImgSegTest();
	string imgPath = "C:/Users/Administrator/Desktop/灰度图.bmp";
	cv::Mat srcImg = cv::imread(imgPath, 0);
	cv::Mat gaussImg;
	cv::GaussianBlur(srcImg, gaussImg, cv::Size(7, 7), 1.41, 1.41);

	return (0);
}
