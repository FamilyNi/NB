#include "../../include/FitShapeFile/GrabEdges.h"

//以圆弧的方式抓边=================================================================================
void Img_GrabEdgesCircle(Mat& srcImg, vector<cv::Point>& edges, cv::Point& center, double r_1, double r_2, double r_step, 
	double startAng, double endAng, double angStep, double thresVal, IMG_GRABEDGEMODE mode, int ptsNo, int scanOrit)
{
	if (edges.size() != 0)
		edges.resize(0);
	double a_step = std::max(0.017, angStep);
	int pts_num = (endAng - startAng) / a_step + 1;
	edges.reserve(pts_num);
	//确定边缘模式
	switch (mode)
	{
	case IMG_EDGE_LIGHT:
		thresVal = thresVal < 0 ? -thresVal : thresVal;	break;
	case IMG_EDGE_DARK:
		thresVal = thresVal > 0 ? -thresVal : thresVal;	break;
	case IMG_EDGE_ABSOLUTE:
		thresVal = abs(thresVal);break;
	default: break;
	}
	//确定扫描方向
	double r_ = std::max(1.0, r_step);
	r_ = scanOrit == 0 ? r_: -r_;
	//确定半径终止条件以及起始半径
	double strart_r = scanOrit == 0 ? r_1 : r_2;
	double r_1_2 = r_1 * r_1, r_2_2 = r_2 * r_2;

	for (int i = 0; i < pts_num; ++i)
	{
		double angle = startAng + i * angStep;
		double n_x = std::cos(angle);
		double n_y = std::sin(angle);
		double step_x = r_ * n_x;
		double step_y = r_ * n_y;

		double min_x = center.x + strart_r * n_x;
		double min_y = center.y + strart_r * n_y;

		vector<cv::Point> pts;
		pts.reserve(5);
		while (min_x > 0 && min_x < srcImg.cols - 1.0 && min_y > 0 
			&& min_y < srcImg.rows - 1.0 && pts.size() <5)
		{
			double x_ = min_x, y_ = min_y;
			min_x += step_x; min_y += step_y;
			double dist = std::pow(min_x - center.x, 2) + std::pow(min_y - center.y, 2);
			if (dist <= r_2_2 && dist >= r_1_2 && min_x > 0 && min_x < srcImg.cols - 1.0 && min_y > 0 && min_y < srcImg.cols - 1.0)
			{
				int diff_GrayVal = srcImg.at<uchar>(min_y, min_x) - srcImg.at<uchar>(y_, x_);
				switch (mode)
				{
				case IMG_EDGE_LIGHT:
					if (diff_GrayVal > thresVal)
						pts.push_back(cv::Point(min_x, min_y));
					break;
				case IMG_EDGE_DARK:
					if (diff_GrayVal < thresVal)
						pts.push_back(cv::Point(min_x, min_y));
					break;
				case IMG_EDGE_ABSOLUTE:
					if (abs(diff_GrayVal) > thresVal)
						pts.push_back(cv::Point(min_x, min_y));
					break;
				default: break;
				}
			}
		}
		if (pts.size() != 0)
		{
			if (pts.size() > ptsNo)
				edges.push_back(pts[ptsNo - 1]);
			else
				edges.push_back(pts.back());
		}
	}
}
//=================================================================================================

//以矩形的方式抓边=================================================================================
void Img_GrabEdgesRect(Mat& srcImg, vector<cv::Point>& edges, cv::Point& start_p, cv::Point& end_p, int width,
	double step1, double step2, double thresVal, IMG_GRABEDGEMODE mode, int ptsNo, int scanOrit)
{
	if (edges.size() != 0)
		edges.resize(0);
	//确定边缘模式
	switch (mode)
	{
	case IMG_EDGE_LIGHT:
		thresVal = thresVal < 0 ? -thresVal : thresVal;	break;
	case IMG_EDGE_DARK:
		thresVal = thresVal > 0 ? -thresVal : thresVal;	break;
	case IMG_EDGE_ABSOLUTE:
		thresVal = abs(thresVal); break;
	default: break;
	}
	double vec_x = end_p.x - start_p.x;
	double vec_y = end_p.y - start_p.y;
	double norm_dist = std::sqrt(vec_x * vec_x + vec_y * vec_y);
	vec_x /= std::max(norm_dist, EPS);
	vec_y /= std::max(norm_dist, EPS);
	double norm_x = scanOrit == 0 ? -vec_y : vec_y;
	double norm_y = scanOrit == 0 ? vec_x : -vec_x;
	
	double max_wx = norm_x * width;
	double max_wy = norm_y * width;

	double step_x = step2 * norm_x;
	double step_y = step2 * norm_y;

	int pt_num = norm_dist / step1 + 1;
	edges.reserve(pt_num);

	for (int i = 0; i < pt_num; ++i)
	{
		double ref_x = start_p.x + step1 * i * vec_x;
		double ref_y = start_p.y + step1 * i * vec_y;

		double min_x = ref_x - max_wx;
		double min_y = ref_y - max_wy;

		vector<cv::Point> pts;
		pts.reserve(5);
		while (min_x > 0 && min_x < srcImg.cols - 1.0 && min_y > 0
			&& min_y < srcImg.rows - 1.0 && pts.size() < 5)
		{
			double x_ = min_x, y_ = min_y;
			min_x += step_x; min_y += step_y;
			double dist = std::pow(min_x - ref_x, 2) + std::pow(min_y - ref_y, 2);
			if (dist <= width * width && min_x > 0 && min_x < srcImg.cols - 1.0 && min_y > 0 && min_y < srcImg.cols - 1.0)
			{
				int diff_GrayVal = srcImg.at<uchar>(min_y, min_x) - srcImg.at<uchar>(y_, x_);
				switch (mode)
				{
				case IMG_EDGE_LIGHT:
					if (diff_GrayVal > thresVal)
						pts.push_back(cv::Point(min_x, min_y));
					break;
				case IMG_EDGE_DARK:
					if (diff_GrayVal < thresVal)
						pts.push_back(cv::Point(min_x, min_y));
					break;
				case IMG_EDGE_ABSOLUTE:
					if (abs(diff_GrayVal) > thresVal)
						pts.push_back(cv::Point(min_x, min_y));
					break;
				default: break;
				}
			}
		}
		if (pts.size() != 0)
		{
			if (pts.size() > ptsNo)
				edges.push_back(pts[ptsNo - 1]);
			else
				edges.push_back(pts.back());
		}
	}
}
//=================================================================================================