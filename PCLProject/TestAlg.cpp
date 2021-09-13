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
#include "ContourOpr.h"
#include "ShapeMatch.h"
#include "LBPfeatures.h"
#include "ImageFilter.h"
#include "ImageEnhance.h"
#include "ImageSeg.h"
#include "GrabEdges.h"
#include "ComputeLine.h"
#include "ComputeCircle.h"

int main(int argc, char *argv[])
{
	CircleTest();
	//ImgSegTest();
	string imgPath = "C:/Users/Administrator/Desktop/1.bmp";
	cv::Mat srcImg = cv::imread(imgPath, 0);

	//第一种思路
	vector<cv::Point> circle;
	cv::Point center(285, 472);
	cv::Point center_(290, 153);
	Img_GrabEdgesRect(srcImg, circle, center, center_, 200, 5, 2, 10, IMG_GRABEDGEMODE::IMG_EDGE_ABSOLUTE, 4, 0);


	Mat colorImg;
	cv::cvtColor(srcImg, colorImg, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < circle.size(); ++i)
	{
		cv::line(colorImg, circle[i], circle[i], cv::Scalar(0, 255, 0), 2);
		//cv::drawContours(colorImg, lines, i, cv::Scalar(0, 0, 255), 1);
	}

	ImgSegTest();
	return (0);
}
