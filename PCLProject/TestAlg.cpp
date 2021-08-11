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

int main(int argc, char *argv[])
{
	LocalDeforModelTest();

	Mat image1 = cv::imread("E:/4.jpg", 0);

	//Mat binImg;
	//cv::threshold(image1, binImg, 0, 255, THRESH_OTSU);

	//Mat kenerl = getStructuringElement(MORPH_RECT, cv::Size(5, 5));
	//Mat erodeImg;
	//erode(binImg, erodeImg, kenerl);

	Mat guassImg;
	GaussianBlur(image1, guassImg, cv::Size(5, 5), 1.02, 1.02);

	Mat cannyImg;
	cv::Canny(guassImg, cannyImg, 30, 100);

	Mat thres;
	cv::adaptiveThreshold(guassImg, thres, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, -5);
	//LocalDeforModelTest();

	vector<vector<cv::Point>> contours;
	cv::findContours(thres, contours, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
	int maxIndex = 0;
	GetMaxAreaContour(contours, maxIndex);
	cv::Mat mask(thres.size(), CV_8UC1, cv::Scalar(0));
	cv::fillConvexPoly(mask, contours[maxIndex], cv::Scalar(255));

	Mat kenerl = getStructuringElement(MORPH_RECT, cv::Size(5, 5));
	Mat erodeImg;
	erode(mask, erodeImg, kenerl, cv::Point(-1, -1),2);

	Mat roi;
	cv::multiply(erodeImg, thres, roi);

	vector<Vec4i> lines;
	HoughLinesP(roi, lines, 1, CV_PI / 180, 50, 400, 200);

	Mat colorImg;
	cv::cvtColor(image1, colorImg, COLOR_GRAY2BGR);
	//vector<Point> contour(2 * lines.size());
	//for (int i = 0; i < lines.size(); ++i)
	//{
	//	cv::Point pt1(lines[i][0], lines[i][1]);
	//	cv::Point pt2(lines[i][2], lines[i][3]);
	//	cv::line(colorImg, pt1, pt2, Scalar(0, 0, 255), 2);
	//	contour[2 * i] = pt1;
	//	contour[2 * i + 1] = pt2;
	//}

	//筛选直线
	vector<Vec4i> lines__;
	vector<bool> flags(lines.size(), false);
	for (int i = 0; i < lines.size(); ++i)
	{
		if (flags[i])
			continue;
		float a1 = lines[i][2] - lines[i][0];
		float b1 = lines[i][3] - lines[i][1];
		float length1 = std::sqrt(a1 * a1 + b1 * b1);
		a1 /= length1;
		b1 /= length1;
		cout << a1 << " " << b1 << endl;
		flags[i] = true;

		int index = i;
		for (int j = 0; j < lines.size(); ++j)
		{
			if (flags[j])
				continue;
			float a2 = lines[j][2] - lines[j][0];
			float b2 = lines[j][3] - lines[j][1];
			float length2 = std::sqrt(a2 * a2 + b2 * b2);
			a2 /= length2;
			b2 /= length2;
			float cosVal = abs(a1 * a2 + b1 * b2);
			
			float c = -(-b2 * lines[j][0] + a2 * lines[j][1]);
			float dist = abs(-b2 * lines[i][0] + lines[i][1] * a2 + c);

			if (cosVal > 0.95 && dist < 20) //近似为一条直线
			{
				if (length1 < length2)
				{
					index = j;
					a1 = a2;
					b1 = b2;
					length1 = length2;
				}
				flags[j] = true;
			}
		}
		lines__.push_back(lines[index]);
	}

	for (int i = 0; i < lines__.size(); ++i)
	{
		cv::Point pt1(lines__[i][0], lines__[i][1]);
		cv::Point pt2(lines__[i][2], lines__[i][3]);
		cout << lines__[i] << endl;
		cv::line(colorImg, pt1, pt2, Scalar(0, 0, 255), 2);
	}

	return (0);
}
