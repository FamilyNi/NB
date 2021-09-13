#include "ComputeCircle.h"

//三点求圆======================================================================================
template <typename T>
void Img_ThreePointComputeCicle(T& pt1, T& pt2, T& pt3, cv::Vec3d& circle)
{
	double B21 = pt2.x * pt2.x + pt2.y * pt2.y - (pt1.x * pt1.x + pt1.y * pt1.y);
	double B32 = pt3.x * pt3.x + pt3.y * pt3.y - (pt2.x * pt2.x + pt2.y * pt2.y);

	double X21 = pt2.x - pt1.x;
	double Y21 = pt2.y - pt1.y;
	double X32 = pt3.x - pt2.x;
	double Y32 = pt3.y - pt2.y;

	circle[0] = 0.5 * (B21 * Y32 - B32 * Y21) / (X21 * Y32 - X32 * Y21);
	circle[1] = 0.5 * (B21 * X32 - B32 * X21) / (Y21 * X32 - Y32 * X21);

	double diff_x = pt1.x - circle[0];
	double diff_y = pt1.y - circle[1];
	circle[2] = std::sqrt(diff_x * diff_x + diff_y * diff_y);
}
//==============================================================================================

//随机一致采样算法计算直线======================================================================
template <typename T>
void Img_RANSACComputeCircle(vector<T>& pts, cv::Vec3d& circle, vector<T>& inlinerPts, double thres)
{
	if (pts.size() < 2)
		return;
	int best_model_p = 0;
	double P = 0.99;  //模型存在的概率
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	for (int i = 0; i < maxEpo; ++i)
	{
		int effetPoints = 0;
		//随机选择两个点，并计算直线
		int index_1 = rand() % size;
		int index_2 = rand() % size;
		int index_3 = rand() % size;
		cv::Vec3d circle_;
		Img_ThreePointComputeCicle(pts[index_1], pts[index_2], pts[index_3], circle_);

		//计算局内点的个数
		for (int j = 0; j < size; ++j)
		{
			float diff_x = pts[j].x - circle_[0];
			float diff_y = pts[j].y - circle_[1];
			float dist = std::sqrt(diff_x * diff_x + diff_y * diff_y);
			effetPoints += abs(dist - circle_[2]) < thres ? 1 : 0;
		}
		//获取最优模型，并根据概率修改迭代次数
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			circle = circle_;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
		if (best_model_p > 0.5 * size)
		{
			circle = circle_;
			break;
		}
	}

	//提取局内点
	if (inlinerPts.size() != 0)
		inlinerPts.resize(0);
	inlinerPts.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		float diff_x = pts[i].x - circle[0];
		float diff_y = pts[i].y - circle[1];
		float dist = std::sqrt(diff_x * diff_x + diff_y * diff_y);
		if (abs(dist - circle[2]) < thres)
			inlinerPts.push_back(pts[i]);
	}
}
//==============================================================================================




void CircleTest()
{
	string imgPath = "C:/Users/Administrator/Desktop/testimage/6.bmp";
	cv::Mat srcImg = cv::imread(imgPath, 0);
	cv::Mat binImg;
	cv::threshold(srcImg, binImg, 10, 255, ThresholdTypes::THRESH_BINARY_INV);
	vector<vector<cv::Point>> contours;
	cv::findContours(binImg, contours, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_NONE);

	Mat colorImg;
	cv::cvtColor(srcImg, colorImg, cv::COLOR_GRAY2BGR);

	vector<cv::Point> pts(contours.size());
	for (int i = 0; i < contours.size(); ++i)
	{
		int len = contours[i].size();
		if (len == 0)
			return;
		float sum_x = 0.0f, sum_y = 0.0f;
		for (int j = 0; j < len; ++j)
		{
			sum_x += contours[i][j].x;
			sum_y += contours[i][j].y;
		}
		pts[i].x = sum_x / len;
		pts[i].y = sum_y / len;
	}

	cv::Vec3d circle;
	vector<Point> inlinerPts;
	Img_RANSACComputeCircle(pts, circle, inlinerPts, 10);
	cv::circle(colorImg, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 2);

	for (int i = 0; i < inlinerPts.size(); ++i)
	{
		cv::line(colorImg, inlinerPts[i], inlinerPts[i], cv::Scalar(0, 0, 255), 5);
	}
}