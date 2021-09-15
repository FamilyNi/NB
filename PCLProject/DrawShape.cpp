#include "DrawShape.h"

//绘制平面==========================================================
void PC_DrawLine(PC_XYZ::Ptr& linePC, double min_x, double max_x, double min_y,
	double max_y, double min_z, double max_z, cv::Vec6d& line, double step)
{
	double norm = std::sqrt(line[0] * line[0] + line[1] * line[1] + line[2] * line[2]);
	if (abs(norm - 1) > 1e-8)
	{
		line[0] /= norm; line[1] /= norm; line[2] /= norm;
	}
	if (norm < 1e-8)
		return;
	step = step < 1e-5 ? 1 : step;
	double x_ = (min_x + max_x) / 2.0;
	double y_ = (min_y + max_y) / 2.0;
	double z_ = (min_z + max_z) / 2.0;
	for (float x = min_x; x <= max_x; x += step)
	{
		for (float y = min_y; y <= max_y; y += step)
		{
			for (float z = min_z; z <= max_z; z += step)
			{
				P_XYZ p{ x,y,z };
				P_XYZ projPt;
				PC_3DPtProjLinePt(p, line, projPt);
				linePC->points.push_back(projPt);
			}
		}
	}
}
//==================================================================

//绘制平面==========================================================
void PC_DrawPlane(PC_XYZ::Ptr& planePC, double min_x, double max_x, double min_y,
	double max_y, double min_z, double max_z, cv::Vec4d& plane, double step)
{
	double norm = std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
	if (abs(norm - 1) > 1e-8)
	{
		plane[0] /= norm; plane[1] /= norm; plane[2] /= norm;
	}
	if (norm < 1e-8)
		return;
	step = step < 1e-5 ? 1 : step;
	double x_ = (min_x + max_x) / 2.0;
	double y_ = (min_y + max_y) / 2.0;
	double z_ = (min_z + max_z) / 2.0;
	plane[3] = -(plane[0] * x_ + plane[1] * y_ + plane[2] * z_);
	for (float x = min_x; x <= max_x; x += step)
	{
		for(float y = min_y; y <= max_y; y += step)
		{
			for (float z = min_z; z <= max_z; z += step)
			{
				P_XYZ p{ x,y,z };
				P_XYZ projPt;
				PC_PtProjPlanePt(p, plane, projPt);
				planePC->points.push_back(projPt);
			}
		}
	}
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