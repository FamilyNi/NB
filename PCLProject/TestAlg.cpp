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
#include "ContourOpr.h"
#include "LBPfeatures.h"
#include "ImageEnhance.h"
#include "ComputePlane.h"
#include "ImageFilter.h"
#include "ComputeLine.h"
#include "Compute3DCircle.h"

int main(int argc, char *argv[])
{
	PC_CircleTest();
	//PC_XYZ::Ptr planePC(new PC_XYZ);
	//cv::Vec6d plane;
	//plane[0] = 13;
	//plane[1] = 5;
	//plane[2] = 18;
	//plane[3] = 29;
	//plane[4] = 36;
	//plane[5] = 65;
	//PC_DrawPlane(planePC, plane, 50, 60, 0.2);

	//PC_XYZ::Ptr noisePC(new PC_XYZ);
	//PC_AddNoise(planePC, noisePC, 7, 5);

	//pcl::io::savePLYFile("C:/Users/Administrator/Desktop/testimage/噪声平面.ply", *noisePC);

	//pcl::visualization::PCLVisualizer viewer;
	//viewer.addCoordinateSystem(10);
	////显示轨迹
	//pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(noisePC, 255, 0, 0); //设置点云颜色
	//viewer.addPointCloud(noisePC, red, "noisePC");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "noisePC");
	//while (!viewer.wasStopped())
	//{
	//	viewer.spinOnce();
	//}
	return (0);
}
