#include "ComputeLine.h"

//两点计算直线==================================================================================
template <typename T>
void Img_TwoPtsComputeLine(T& pt1, T& pt2, cv::Vec3d& line)
{
	double vec_x = pt2.x - pt1.x;
	double vec_y = pt2.y - pt1.y;
	double norm_ = 1 / std::sqrt(vec_x * vec_x + vec_y * vec_y);
	line[0] = -vec_y * norm_;
	line[1] = vec_x * norm_;
	line[2] = -(line[0] * pt1.x + line[1] * pt1.y);
}
//==============================================================================================

//随机一致采样算法计算直线======================================================================
template <typename T>
void Img_RANSACComputeLine(vector<T>& pts, cv::Vec3d& line, vector<T>& inlinerPts, double thres)
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
		cv::Vec3d line_;
		Img_TwoPtsComputeLine(pts[index_1], pts[index_2], line_);

		//计算局内点的个数
		for (int j = 0; j < size; ++j)
		{
			float dist = abs(line_[0] * pts[j].x + line_[1] * pts[j].y + line_[2]);
			effetPoints += dist < thres ? 1 : 0;
		}
		//获取最优模型，并根据概率修改迭代次数
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			line = line_;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
		if (best_model_p > 0.5 * size)
		{
			line = line_;
			break;
		}
	}

	//提取局内点
	if (inlinerPts.size() != 0)
		inlinerPts.resize(0);
	inlinerPts.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		if (abs(line[0] * pts[i].x + line[1] * pts[i].y + line[2]) < thres)
			inlinerPts.push_back(pts[i]);
	}
}
//==============================================================================================

/*Turkey直线拟合*/
void Img_TurkeyFitLine(vector<cv::Point>& pts, cv::Vec3d& line, int k, double thres)
{
	//初始化权重以及权重坐标
	vector<double> weigths(pts.size(), 1);
	vector<Point> w_pts(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		w_pts[i] = pts[i];
	}

	//进行迭代
	vector<double> pldistance(pts.size());
	cv::Vec4d line_;
	for (int i = 0; i < k; ++i)
	{
		//求权重和、权重坐标各种矩
		//float w_sum = 0.0;
		//float center_x = 0.0;
		//float center_y = 0.0;
		//float center_xy = 0.0;
		//float center_x2 = 0.0;
		//float center_y2 = 0.0;
		//for (int j = 0; j < weigths.size(); ++j)
		//{
		//	w_sum += weigths[j];
		//	center_x += weigths[j] * weights_points[j].x;
		//	center_y += weigths[j] * weights_points[j].y;
		//	center_xy += center_x * weights_points[j].y;
		//	center_x2 += center_x * weights_points[j].x;
		//	center_y2 += center_y * weights_points[j].y;
		//}
		//w_sum = 1.0 / w_sum;

		////求矩阵个元素
		//float u20 = (center_x2 - center_x * center_x * w_sum) / weights_points.size();
		//float u11 = (center_xy - center_x * center_y * w_sum) / weights_points.size();
		//float u02 = (center_y2 - center_y * center_y * w_sum) / weights_points.size();

		//求二阶矩的本征值以及本征向量
		//Mat A = (Mat_<float>(2, 2) << u20, u11, u11, u02);
		//Mat eigenValue, eigenVector;
		//eigenNonSymmetric(A, eigenValue, eigenVector);

		//提取直线的a, b , c值，较小本征值的本征向量一般放在第一列，测试是这样的。
		//float* pEigenVector = eigenVector.ptr<float>(0);
		cv::fitLine(w_pts, line_, DistanceTypes::DIST_L2, 0, 0.01, 0.01);
		//line[0] = pEigenVector[0];
		//line[1] = pEigenVector[2];
		//line[2] = (-center_x * line[0] - center_y * line[1]) * w_sum;

		//求点到直线的距离——这里是原坐标点并不是权重坐标点
		line_[2] = -(line_[2] * (-line_[1]) + line_[3] * line_[0]);
		//float norm = sqrt(line[0] * line[0] + line[1] * line[1]);
		for (int j = 0; j < pts.size(); ++j)
		{
			float distance = abs(pts[j].x * (-line_[1]) + pts[j].y * line_[0] + line_[2]);
			pldistance[j] = distance;
		}
		//求限制条件tao
		vector<double> disttanceSort = pldistance;
		sort(disttanceSort.begin(), disttanceSort.end());
		double tao = disttanceSort[(disttanceSort.size() - 1) / 2] / 0.6745 * 2;

		//更新权重
		for (int j = 0; j < pldistance.size(); ++j)
		{
			if (pldistance[j] <= tao)
				weigths[j] = pow(1 - (pldistance[j] / tao)*(pldistance[j] / tao), 2);
			else weigths[j] = 5.0 / pldistance[j];
		}
		for (int j = 0; j < pldistance.size(); ++j)
		{
			w_pts[j] = pts[j] * weigths[j];
		}
	}
	line[0] = -line_[1];
	line[1] = line_[0];
	line[2] = line_[2];
}


void LineTest()
{
	string imgPath = "C:/Users/Administrator/Desktop/4.bmp";
	cv::Mat srcImg = cv::imread(imgPath, 0);
	cv::Mat binImg;
	cv::threshold(srcImg, binImg, 10, 255, ThresholdTypes::THRESH_BINARY_INV);
	vector<vector<cv::Point>> contours;
	cv::findContours(binImg, contours, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_NONE);

	Mat colorImg;
	cv::cvtColor(srcImg, colorImg, cv::COLOR_GRAY2BGR);
	//for (int i = 0; i < contours.size(); ++i)
	//{
	//	cv::drawContours(colorImg, contours, i, cv::Scalar(0, 255, 0), 2);
	//}

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

	cv::Vec3d line;
	vector<cv::Point> inlinerPts;
	Img_TurkeyFitLine(pts, line, 5, 2);
	cv::Point s_pt, e_pt;
	s_pt.x = 35; s_pt.y = -(line[2] + 35 * line[0]) / line[1];
	e_pt.x = 800; e_pt.y = -(line[2] + 800 * line[0]) / line[1];

	for (int i = 0; i < inlinerPts.size(); ++i)
	{
		cv::line(colorImg, inlinerPts[i], inlinerPts[i], cv::Scalar(0, 0, 255), 10);
		//cv::drawContours(colorImg, lines, i, cv::Scalar(0, 0, 255), 1);
	}
	cv::line(colorImg, s_pt, e_pt, cv::Scalar(0, 255, 0), 3);
}