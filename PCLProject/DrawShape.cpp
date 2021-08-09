#include "DrawShape.h"

//ªÊ÷∆«Ú============================================================
void DrawSphere(PC_XYZ::Ptr& sphere, P_XYZ& center, double raduis, double step)
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
			sphere->points.push_back({ x, y, z });
		}
	}
}
//==================================================================

//ªÊ÷∆Õ÷«Ú√Ê========================================================
void DrawEllipsoid(PC_XYZ::Ptr& ellipsoid, P_XYZ& center, double a, double b, double c, double step)
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