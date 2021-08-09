// PCLProject.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "utils.h"
#include "PC_Filter.h"
#include "FitModel.h"
#include "PC_Seg.h"
#include "JC_Calibrate.h"
#include "PPFMatch.h"
#include "GrayCode.h"
#include "WaveLet.h"
#include "LocalDeforableModel.h"
#include "DrawShape.h"
#include "MathOpr.h"
#include "SiftMatch.h"

int main(int argc, char *argv[])
{
	PC_XYZ::Ptr ellipsoid(new PC_XYZ);
	P_XYZ center(0, 0, 0);
	DrawEllipsoid(ellipsoid, center, 20, 50, 10, 0.1);

	pcl::visualization::PCLVisualizer viewer;
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(ellipsoid, 255, 0, 0);
	viewer.addPointCloud(ellipsoid, red, "ellipsoid");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ellipsoid");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
	return (0);
}
