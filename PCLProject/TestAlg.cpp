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
	//Mat rotMat = getRotationMatrix2D(cv::Point2f(700, 450), 69, 1);
	//Mat srcImg = imread("1.jpg", IMREAD_GRAYSCALE);
	//Mat t_img;
	//warpAffine(srcImg, t_img, rotMat, srcImg.size());
	//cv::imwrite("1_t.png", t_img);
	SiftPtTest();

	LocalDeforModelTest();
	return (0);
}
