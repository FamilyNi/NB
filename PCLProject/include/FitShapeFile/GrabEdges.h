#pragma once
#include "../BaseOprFile/utils.h"

/*说明：
	srcImg：[in]原始图像
	edges：[out]输出的边界
*/

enum IMG_GRABEDGEMODE
{
	IMG_EDGE_LIGHT = 0,
	IMG_EDGE_DARK = 1,
	IMG_EDGE_ABSOLUTE = 2
};

/*以圆弧的方式抓边：
	center：[in]圆弧中心
	r_1、r_2：[in]小圆弧、大圆弧半径 --- r_2 > r_1
	r_step：[in]扫描步长
	startAng、endAng：[in]扫描起始角与终止角度 --- startAng < endAng
	angStep：[in]角度步长
	thresVal：[in]阈值
	mode：[in]边缘模式---IMG_EDGE_LIGHT：取亮边；IMG_EDGE_DARK：取暗边；IMG_EDGE_ABSOLUTE：都取
	ptsNo：[in]要选择的点序号
	scanOrit：[in]扫描方向----0：表示由半径r_2到r_1，1：表示半径r_1到r_2
*/
void Img_GrabEdgesCircle(Mat& srcImg, vector<cv::Point>& edges, cv::Point& center, double r_1, double r_2, double r_step,
	double startAng, double endAng, double angStep, double thresVal, IMG_GRABEDGEMODE mode, int ptsNo, int scanOrit);

/*以矩形的方式抓边
	start_p、end_p：[in]起始点、与终止点
	width：[in]扫描宽度
	step1：[in]点步长
	step2：[in]扫描步长
	thresVal：[in]阈值
	mode：[in]边缘模式---IMG_EDGE_LIGHT：取亮边；IMG_EDGE_DARK：取暗边；IMG_EDGE_ABSOLUTE：都取
	ptsNo：[in]要选择的点序号
	scanOrit：[in]扫描方向
*/
void Img_GrabEdgesRect(Mat& srcImg, vector<cv::Point>& edges, cv::Point& start_p, cv::Point& end_p, int width,
	double step1, double step2, double thresVal, IMG_GRABEDGEMODE mode, int ptsNo, int scanOrit);