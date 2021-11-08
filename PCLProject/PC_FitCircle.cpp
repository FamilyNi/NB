#include "PC_FitCircle.h"
#include "PC_FitPlane.h"
#include "MathOpr.h"
//#include "MathOpr.cpp"

//点到圆的距离====================================================================================
//template <typename T1, typename T2>
//void PC_PtToCircleDist(T1& pt, T2& circle, double& dist)
//{
//	double diff_x = pt.x - circle[0];
//	double diff_y = pt.y - circle[1];
//	double diff_z = pt.z - circle[2];
//	dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
//	dist = abs(dist - circle[3]);
//}
//================================================================================================

//三点计算园======================================================================================
//template <typename T1, typename T2>
//void PC_ThreePtsComputeCircle(T1& pt1, T1& pt2, T1& pt3, T2& circle)
//{
//	//首先计算3点所在平面的法向量
//	P_XYZ nor_1(pt1.x - pt2.x, pt1.y - pt2.y, pt1.z - pt2.z);
//	P_XYZ nor_2(pt1.x - pt3.x, pt1.y - pt3.y, pt1.z - pt3.z);
//	P_XYZ norm(0, 0, 0);
//	PC_VecCross(nor_1, nor_2, norm, true);
//
//	//这个地方不想手动解方程组了，交给opencv了
//	Mat A(cv::Size(3, 3), CV_64FC1, cv::Scalar(0));
//	double* pA = A.ptr<double>(0);
//	pA[0] = 2.0 * (pt1.x - pt2.x); pA[1] = 2.0 * (pt1.y - pt2.y); pA[2] = 2.0 * (pt1.z - pt2.z);
//	pA[3] = 2.0 * (pt2.x - pt3.x); pA[4] = 2.0 * (pt2.y - pt3.y); pA[5] = 2.0 * (pt2.z - pt3.z);
//	pA[6] = norm.x; pA[7] = norm.y; pA[8] = norm.z;
//	Mat B(cv::Size(1, 3), CV_64FC1, cv::Scalar(0));
//	double* pB = B.ptr<double>(0); 
//	pB[0] = pt1.x * pt1.x - pt2.x * pt2.x + pt1.y * pt1.y - pt2.y * pt2.y + pt1.z * pt1.z - pt2.z * pt2.z;
//	pB[1] = pt2.x * pt2.x - pt3.x * pt3.x + pt2.y * pt2.y - pt3.y * pt3.y + pt2.z * pt2.z - pt3.z * pt3.z;
//	pB[2] = norm.x * pt1.x + norm.y * pt1.y + norm.z * pt1.z;
//	Mat C = A.inv() * B;
//	double* pC = C.ptr<double>(0);
//	circle[0] = pC[0]; circle[1] = pC[1]; circle[2] = pC[2];
//	double diff_x = pt1.x - pC[0], diff_y = pt1.y - pC[1], diff_z = pt1.z - pC[2];
//	circle[3] = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
//	circle[4] = norm.x; circle[5] = norm.y; circle[6] = norm.z;
//}
//================================================================================================

//随机一致采样算法计算园==========================================================================
//template <typename T1, typename T2>
//void PC_RANSACFitCircle(vector<T1>& pts, T2& circle, vector<T1>& inlinerPts, double thres)
//{
//	if (pts.size() < 3)
//		return;
//	int best_model_p = 0;
//	double P = 0.99;  //模型存在的概率
//	double log_P = log(1 - P);
//	int size = pts.size();
//	int maxEpo = 10000;
//	for (int i = 0; i < maxEpo; ++i)
//	{
//		int effetPoints = 0;
//		//随机选择三个点计算园---注意：这里可能需要特殊处理防止点相同
//		int index_1 = rand() % size;
//		int index_2 = rand() % size;
//		int index_3 = rand() % size;
//		T2 circle_(7);
//		PC_ThreePtsComputeCircle(pts[index_1], pts[index_2], pts[index_3], circle_);
//		//计算局内点的个数
//		for (int j = 0; j < size; ++j)
//		{
//			double dist = 0.0;
//			PC_PtToCircleDist(pts[j], circle_, dist);
//			effetPoints += dist < thres ? 1 : 0;
//		}
//		//获取最优模型，并根据概率修改迭代次数
//		if (best_model_p < effetPoints)
//		{
//			best_model_p = effetPoints;
//			circle = circle_;
//			double t_P = (double)best_model_p / size;
//			double pow_t_p = t_P * t_P * t_P;
//			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
//		}
//		if (best_model_p > 0.5 * size)
//		{
//			circle = circle_;
//			break;
//		}
//	}
//	//提取局内点
//	if (inlinerPts.size() != 0)
//		inlinerPts.resize(0);
//	inlinerPts.reserve(size);
//	for (int i = 0; i < size; ++i)
//	{
//		double dist = 0.0;
//		PC_PtToCircleDist(pts[i], circle, dist);
//		if (dist < thres)
//			inlinerPts.push_back(pts[i]);
//	}
//}
//================================================================================================

//最小二乘法拟合空间空间园========================================================================
//template <typename T1, typename T2>
//void PC_OLSFit3DCircle(NB_Array3D pts, vector<double>& weights, Circle3D& circle)
//{
//	cv::Vec4d plane;
//	PC_OLSFitPlane(pts, weights, plane);
//
//	double w_sum = 0.0;
//	double w_x_sum = 0.0;
//	double w_y_sum = 0.0;
//	double w_z_sum = 0.0;
//	double w_x2y2z2_sum = 0.0;
//	for (int i = 0; i < pts.size(); ++i)
//	{
//		w_sum += weights[i];
//		w_x_sum += weights[i] * pts[i].x;
//		w_y_sum += weights[i] * pts[i].y;
//		w_z_sum += weights[i] * pts[i].z;
//		w_x2y2z2_sum += weights[i] * (pts[i].x * pts[i].x + pts[i].y * pts[i].y + pts[i].z * pts[i].z);
//	}
//	w_sum = 1.0 / std::max(w_sum, EPS);
//	double w_x_mean = w_x_sum * w_sum;
//	double w_y_mean = w_y_sum * w_sum;
//	double w_z_mean = w_z_sum * w_sum;
//	double w_x2y2z2_mean = w_x2y2z2_sum * w_sum;
//
//	double a = plane[0], b = plane[1], c = plane[2];
//	Mat A(3, 3, CV_64FC1, cv::Scalar(0));
//	Mat B(3, 1, CV_64FC1, cv::Scalar(0));
//	double* pA = A.ptr<double>(0);
//	double* pB = B.ptr<double>(0);
//	for (int i = 0; i < pts.size(); ++i)
//	{
//		double x = pts[i].x, y = pts[i].y, z = pts[i].z, w = weights[i];
//		double x_ = x - w_x_mean, y_ = y - w_y_mean, z_ = z - w_z_mean;
//		pA[0] += w * (x_ * x_ + 0.25 * a * a);
//		pA[1] += w * (x_ * y_ + 0.25 * a * b);
//		pA[2] += w * (x_ * z_ + 0.25 * c * a);
//		pA[4] += w * (y_ * y_ + 0.25 * b * b);
//		pA[5] += w * (y_ * z_ + 0.25 * b * c);
//		pA[8] += w * (z_ * z_ + 0.25 * c * c);
//
//		double r_ = pts[i].x * pts[i].x + pts[i].y * pts[i].y + pts[i].z * pts[i].z - w_x2y2z2_mean;
//		pB[0] -= w * (x_ * r_ + 0.5 * a * a * x + 0.5 * a * b * y + 0.5 * a * c * z);
//		pB[1] -= w * (y_ * r_ + 0.5 * a * b * x + 0.5 * b * b * y + 0.5 * b * c * z);
//		pB[2] -= w * (z_ * r_ + 0.5 * a * c * x + 0.5 * b * c * y + 0.5 * c * c * z);
//	}
//	pA[3] = pA[1]; pA[6] = pA[2]; pA[7] = pA[5];
//
//	Mat C = (A.inv()) * B;
//	double* pC = C.ptr<double>(0);
//	circle[0] = -pC[0] / 2.0;
//	circle[1] = -pC[1] / 2.0;
//	circle[2] = -pC[2] / 2.0;
//	double c_ = -(pC[0] * w_x_mean + pC[1] * w_y_mean + pC[2] * w_z_mean + w_x2y2z2_mean);
//	circle[3] = std::sqrt(std::max(circle[0] * circle[0] + circle[1] * circle[1] + circle[2] * circle[2] - c_, EPS));
//	circle[4] = a; circle[5] = b; circle[6] = c;
//}
////================================================================================================
//
////Huber计算权重===================================================================================
//template <typename T1, typename T2>
//void PC_HuberCircleWeights(vector<T1>& pts, T2& circle, vector<double>& weights)
//{
//	double tao = 1.345;
//	for (int i = 0; i < pts.size(); ++i)
//	{
//		double dist = 0.0;
//		PC_PtToCircleDist(pts[i], circle, dist);;
//		if (dist <= tao)
//		{
//			weights[i] = 1;
//		}
//		else
//		{
//			weights[i] = tao / dist;
//		}
//	}
//}
////================================================================================================
//
////Tukey计算权重===================================================================================
//template <typename T1, typename T2>
//void PC_TukeyCircleWeights(vector<T1>& pts, T2& circle, vector<double>& weights)
//{
//	vector<double> dists(pts.size());
//	for (int i = 0; i < pts.size(); ++i)
//	{
//		double distance = 0.0;
//		PC_PtToCircleDist(pts[i], circle, distance);
//		dists[i] = distance;
//	}
//	//求限制条件tao
//	vector<double> disttanceSort = dists;
//	sort(disttanceSort.begin(), disttanceSort.end());
//	double tao = disttanceSort[(disttanceSort.size() - 1) / 2] / 0.6745 * 2;
//
//	//更新权重
//	for (int i = 0; i < dists.size(); ++i)
//	{
//		if (dists[i] <= tao)
//		{
//			double d_tao = dists[i] / tao;
//			weights[i] = std::pow((1 - d_tao * d_tao), 2);
//		}
//		else weights[i] = 0;
//	}
//}
////================================================================================================
//
////拟合球==========================================================================================
//template <typename T1, typename T2>
//void PC_FitCircle(vector<T1>& pts, T2& circle, int k, NB_MODEL_FIT_METHOD method)
//{
//	vector<double> weights(pts.size(), 1);
//	PC_OLSFit3DCircle(pts, weights, circle);
//	if (method == NB_MODEL_FIT_METHOD::OLS_FIT)
//	{
//		return;
//	}
//	else
//	{
//		for (int i = 0; i < k; ++i)
//		{
//			switch (method)
//			{
//			case HUBER_FIT:
//				PC_HuberCircleWeights(pts, circle, weights);
//				break;
//			case TUKEY_FIT:
//				PC_TukeyCircleWeights(pts, circle, weights);
//				break;
//			default:
//				break;
//			}
//			PC_OLSFit3DCircle(pts, weights, circle);
//		}
//	}
//}
//================================================================================================
