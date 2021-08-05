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

int main(int argc, char *argv[])
{
	PC_XYZ::Ptr sphere(new PC_XYZ);
	P_XYZ center = { 100.69, 3000.2,150.69 };
	DrawSphere(sphere, center, 20.63, 0.1);
	Sphere sphere_param;
	PC_OLSFitSphere(sphere, sphere_param);

	LocalDeforModelTest();
	return (0);
}
