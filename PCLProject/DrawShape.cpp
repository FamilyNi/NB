#include "DrawShape.h"

//ªÊ÷∆«Ú============================================================
void DrawSphere(PC_XYZ::Ptr& sphere, P_XYZ& center, double raduis, double step)
{
	for (double theta = -CV_PI / 2; theta <= CV_PI / 2; theta += step)
	{
		double r_xy = raduis * std::cos(theta);
		float z = raduis * std::sin(theta) + center.z;
		for (double alpha = 0; alpha <= CV_2PI; alpha += step)
		{
			float x = r_xy * std::cos(alpha) + center.x;
			float y = r_xy * std::sin(alpha) + center.y;
			sphere->points.push_back({ x, y, z });
		}
	}
}
//==================================================================