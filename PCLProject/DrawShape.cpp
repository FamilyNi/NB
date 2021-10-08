#include "DrawShape.h"
#include "PointCloudOpr.h"
#include "MathOpr.h"

//形状变换==========================================================
void PC_ShapeTrans(PC_XYZ::Ptr& pc, cv::Vec6d& shape, int mode)
{
	double norm_ = std::sqrt(shape[0] * shape[0] +
		shape[1] * shape[1] + shape[2] * shape[2]);
	cv::Point3d norm_1(shape[0] / norm_, shape[1] / norm_, shape[2] / norm_);
	cv::Point3d norm_2(1.0f, 0.0f, 0.0f);
	double rotAng = std::acos(norm_1.x);
	if (mode == 1)
	{
		norm_2 = cv::Point3d(0.0f, 0.0f, 1.0f);
		rotAng = std::acos(norm_1.z);
	}
	cv::Point3d normal_p;
	PC_VecCross(norm_1, norm_2, normal_p, true);
	cv::Mat rotMat;
	RodriguesFormula(normal_p, rotAng, rotMat);
	Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity();
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			transMat(i, j) = rotMat.at<double>(j, i);
		}
	}
	PC_XYZ::Ptr linePC_t(new PC_XYZ);
	transMat(0, 3) = shape[3]; transMat(1, 3) = shape[4]; transMat(2, 3) = shape[5];
	pcl::transformPointCloud(*pc, *pc, transMat);
}
//==================================================================

//绘制直线==========================================================
void PC_DrawLine(PC_XYZ::Ptr& linePC, cv::Vec6d& line, double length, double step)
{
	step = step < 1e-5 ? 1 : step;
	for (double x = -length / 2.0; x <= length / 2.0; x += step)
	{
		linePC->points.push_back({(float)x, 0.0f, 0.0f});
	}
	PC_ShapeTrans(linePC, line, 0);
}
//==================================================================

//绘制平面==========================================================
void PC_DrawPlane(PC_XYZ::Ptr& planePC, cv::Vec6d& plane, double length, double width, double step)
{
	step = step < 1e-5 ? 1 : step;
	for (double x = -length / 2.0; x <= length / 2.0; x += step)
	{
		for (double y = -width / 2.0; y < width / 2.0; y += step)
		{
			planePC->points.push_back({ (float)x,float(y), 0.0f });
		}
	}
	PC_ShapeTrans(planePC, plane, 1);
}
//==================================================================

//绘制球============================================================
void PC_DrawSphere(PC_XYZ::Ptr& spherePC, P_XYZ& center, double raduis, double step)
{
	double step_z = CV_PI / (int(CV_PI / step));
	double step_xy = raduis * step;
	for (double theta = -CV_PI / 2; theta <= CV_PI / 2; theta += step_z)
	{
		double r_xy = raduis * std::cos(theta);
		float z = raduis * std::sin(theta) + center.z;
		double step_xy_ = step_xy / std::max(r_xy, (double)EPS);
		step_xy_ = CV_2PI / (int(CV_2PI / step_xy_));
		for (double alpha = 0; alpha < CV_2PI; alpha += step_xy_)
		{
			float x = r_xy * std::cos(alpha) + center.x;
			float y = r_xy * std::sin(alpha) + center.y;
			spherePC->points.push_back({ x, y, z });
		}
	}
}
//==================================================================

//绘制椭球面========================================================
void PC_DrawEllipsoid(PC_XYZ::Ptr& ellipsoid, P_XYZ& center, double a, double b, double c, double step)
{
	double step_z = CV_PI / (int(CV_PI / step));
	double step_xy = std::min(a, b) * step;
	for (double theta = -CV_PI / 2 + step; theta <= CV_PI / 2; theta += step_z)
	{
		double cosVal = std::cos(theta);
		double r_x = a * cosVal;
		double r_y = b * cosVal;
		float z = c * std::sin(theta) + center.z;

		double step_xy_ = step_xy / std::max(std::min(r_x, r_y), (double)EPS);
		step_xy_ = CV_2PI / (int(CV_2PI / step_xy_));
		for (double alpha = 0; alpha < CV_2PI; alpha += step_xy_)
		{
			float x = r_x * std::cos(alpha) + center.x;
			float y = r_y * std::sin(alpha) + center.y;
			ellipsoid->points.push_back({ x, y, z });
		}
	}
}
//==================================================================

//绘制椭圆==========================================================
void Img_DrawEllipse(Mat& ellipseImg, cv::Point2d& center, double rotAng, double a, double b, double step)
{
	if (ellipseImg.empty())
		return;
	double cosVal = std::cos(rotAng);
	double sinVal = std::sin(rotAng);
	for (double theta = 0; theta < CV_2PI; theta += step)
	{
		cv::Point p_;
		double x = a * std::cos(theta);
		double y = b * std::sin(theta);
		p_.x = cosVal * x - sinVal * y + center.x;
		p_.y = cosVal * y + sinVal * x + center.y;
		cv::line(ellipseImg, p_, p_, cv::Scalar(0), 3);
	}
}
//==================================================================

//添加噪声==========================================================
void PC_AddNoise(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& noisePC, int range, int step)
{
	PC_XYZ::Ptr noise_(new PC_XYZ);
	noise_->points.reserve(srcPC->points.size() / step + 1);
	for (int i = 0; i < srcPC->points.size(); i += step)
	{
		P_XYZ& p = srcPC->points[i];
		float dist_x = rand() % range;
		float dist_y = rand() % range;
		float dist_z = rand() % range;
		int index_x = rand() % 2;
		int index_y = rand() % 2;
		int index_z = rand() % 2;
		dist_x = index_x == 0 ? dist_x : -dist_x;
		dist_y = index_y == 0 ? dist_y : -dist_y;
		dist_z = index_z == 0 ? dist_z : -dist_z;
		noise_->push_back({ p.x + dist_x, p.y + dist_y, p.z + dist_z });
	}
	*noisePC = *srcPC + *noise_;
}
//==================================================================

//测试程序==========================================================
void DrawShapeTest()
{
	PC_XYZ::Ptr linePC(new PC_XYZ);
	cv::Vec6d line;
	line[0] = 12;
	line[1] = 25;
	line[2] = 8;
	line[3] = 106;
	line[4] = 69;
	line[5] = 903;
	PC_DrawLine(linePC, line, 20, 0.5);
}
//==================================================================