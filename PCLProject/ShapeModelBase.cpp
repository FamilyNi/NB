#include "ShapeModelBase.h"
#include "ContourOpr.h"
#include <map>

//计算非极大值抑制的长宽=======================================================
void  ComputeNMSRange(vector<Point2f>& contour, int& min_x, int& min_y)
{
	Rect rect = boundingRect(contour);
	min_x = rect.width / 2.0f + 0.5f;
	min_y = rect.height / 2.0f + 0.5f;
}
//============================================================================

//构建图像金字塔======================================================================================
void get_pyr_image(Mat &srcImg, vector<Mat> &pyrImg, int pyrNumber)
{
	if (pyrImg.size() != 0) {
		pyrImg.clear();
	}
	if (pyrNumber < 1) {
		pyrImg.push_back(srcImg);
	}
	else {
		pyrImg.push_back(srcImg);
		Mat e_PyrImg;
		pyrDown(srcImg, e_PyrImg);
		pyrImg.push_back(e_PyrImg);
		for (int i = 2; i < pyrNumber; i++)
		{
			pyrDown(e_PyrImg, e_PyrImg);
			pyrImg.push_back(e_PyrImg);
		}
	}
}
//====================================================================================================

//获得模板信息============================================================================
void ExtractModelContour(Mat &srcImg, SPAPLEMODELINFO &shapeModelInfo, vector<Point> &contour)
{
	vector<vector<Point>> contours;
	ExtractContour(srcImg, contours, shapeModelInfo.lowVal, shapeModelInfo.highVal, shapeModelInfo.extContouMode);
	//vector<vector<Point>> selContours(0);
	//SelContourLen(contours, selContours, shapeModelInfo.minContourLen, shapeModelInfo.maxContourLen);
	MergeContour(contours, contour);
}
//=========================================================================================

//获取模板梯度============================================================================
void ExtractModelInfo(Mat &srcImg, vector<Point> &contour, vector<Point2f> &v_Coord, vector<Point2f> &v_Grad, vector<float> &v_Amplitude)
{
	Mat sobel_x, sobel_y;
	Sobel(srcImg, sobel_x, CV_32FC1, 1, 0, 3);
	Sobel(srcImg, sobel_y, CV_32FC1, 0, 1, 3);
	v_Coord.reserve(contour.size());
	v_Grad.reserve(contour.size());
	v_Amplitude.reserve(contour.size());
	for (size_t i = 0; i < contour.size(); ++i)
	{
		float grad_x = sobel_x.at<float>(contour[i]);
		float grad_y = sobel_y.at<float>(contour[i]);
		if (abs(grad_x) > 0 || abs(grad_y) > 0)
		{

			v_Coord.push_back((Point2f)contour[i]);
			float norm = sqrt(grad_x * grad_x + grad_y * grad_y);
			v_Amplitude.push_back(norm);
			v_Grad.push_back(Point2f(grad_x / norm, grad_y / norm));
		}
	}
}
//========================================================================================

//归一化梯度==========================================================================================
void NormalGrad(int grad_x, int grad_y, float &grad_xn, float &grad_yn)
{
	float norm = 1.0f / sqrtf((float)(grad_x * grad_x + grad_y * grad_y));
	grad_xn = (float)grad_x * norm;
	grad_yn = (float)grad_y * norm;
}
//====================================================================================================

//比较分数大小=======================================================================================
bool CompareSore(MatchRes& a, MatchRes& b)
{
	return (a.score > b.score);
}
//===================================================================================================

//非极大值抑制========================================================================================
void NMS(vector<MatchRes> &MatchReses, vector<MatchRes> &nmsRes, int x_min, int y_min)
{
	std::sort(MatchReses.begin(), MatchReses.end(), CompareSore);
	size_t res_num = MatchReses.size();
	vector<bool>  isLabel(res_num, false);
	for (size_t i = 0; i < res_num; ++i)
	{
		if (isLabel[i])
			continue;
		MatchRes& ref_res = MatchReses[i];
		nmsRes.push_back(ref_res);
		isLabel[i] = true;
		for (size_t j = 0; j < res_num; ++j)
		{
			MatchRes& res_ = MatchReses[j];
			if (abs(ref_res.c_x - res_.c_x) < x_min && abs(ref_res.c_y - res_.c_y) < y_min)
				isLabel[j] = true;
		}
	}
}
//====================================================================================================

//计算梯度============================================================================================
void ComputeGrad(const Mat &srcImg, int idx_x, int idx_y, int& grad_x, int& grad_y)
{
	//注意这里默认是srcImg在内存中的存储是连续的
	const uchar* pUpImg = srcImg.ptr<uchar>(idx_y - 1);
	const uchar* pImg = srcImg.ptr<uchar>(idx_y);
	const uchar* pDownImg = srcImg.ptr<uchar>(idx_y + 1);
	int idx_xr = idx_x - 1;
	int idx_xl = idx_x + 1;

	grad_x = (int)pDownImg[idx_xl] + 2 * (int)pImg[idx_xl] + (int)pUpImg[idx_xl] -
		((int)pDownImg[idx_xr] + 2 * (int)pImg[idx_xr] + (int)pUpImg[idx_xr]);

	grad_y = (int)pDownImg[idx_xr] + 2 * (int)pDownImg[idx_x] + (int)pDownImg[idx_xl] -
		((int)pUpImg[idx_xr] + 2 * (int)pUpImg[idx_x] + (int)pUpImg[idx_xl]);
}
//====================================================================================================

//梯度以及坐标旋转====================================================================================
void RotateCoordGrad(const Mat &x_, const Mat &y_, Mat &r_x, Mat &r_y, float rotAng)
{
	float rotRad = rotAng / 180 * CV_PI;
	float sinVal = sin(rotRad);
	float cosVal = cos(rotRad);
	r_x = x_ * cosVal - y_ * sinVal;
	r_y = y_ * cosVal + x_ * sinVal;
}
void RotateCoordGrad(const vector<Point2f> &coord, const vector<Point2f> &grad, 
	vector<Point2f> &r_coord, vector<Point2f> &r_grad, float rotAng)
{
	float rotRad = rotAng / 180 * CV_PI;
	float sinVal = sin(rotRad);
	float cosVal = cos(rotRad);
	r_coord.resize(coord.size());
	r_grad.resize(grad.size());
	for (int i = 0; i < coord.size(); ++i)
	{
		r_coord[i].x = coord[i].x * cosVal - coord[i].y * sinVal;
		r_grad[i].x = grad[i].x * cosVal - grad[i].y * sinVal;

		r_coord[i].y = coord[i].y * cosVal + coord[i].x * sinVal;
		r_grad[i].y = grad[i].y * cosVal + grad[i].x * sinVal;
	}
}
//====================================================================================================

//绘制轮廓============================================================================================
void draw_contours(Mat &srcImg, vector<Point2f> &contours, Point2f offset)
{
	if (srcImg.empty() || contours.size() == 0) return;
	for (int i = 0; i < contours.size(); i++)
	{
		Point2f drawPoint = contours[i] + offset + Point2f(0.5, 0.5);
		line(srcImg, drawPoint, drawPoint, Scalar(0, 0, 255), 1);
	}
}
void draw_contours(Mat &srcImg, float *pCoord_x, float *pCoord_y, Point offset, int length)
{
	if (srcImg.empty() || pCoord_x == NULL || pCoord_y == NULL)
		return;
	for (int i = 0; i < length; i++)
	{
		Point drawPoint = Point(pCoord_x[i], pCoord_y[i]) + offset;
		line(srcImg, drawPoint, drawPoint, Scalar(0, 0, 255));
	}
}
void draw_contours(Mat &srcImg, vector<Point2f> &contours, vector<uint> &index, Point2f offset)
{
	if (srcImg.empty() || index.size() == 0) return;
	for (int i = 0; i < index.size(); i++)
	{
		Point2f drawPoint = contours[index[i]] + offset + Point2f(0.5, 0.5);
		line(srcImg, drawPoint, drawPoint, Scalar(0, 0, 255), 3);
	}
}
//====================================================================================================

//减少匹配点个数==========================================================================
void ReduceMatchPoint(vector<Point2f> &v_Coord, vector<Point2f> &v_Grad, vector<float> &v_Amplitude,
	vector<Point2f> &v_RedCoord, vector<Point2f> &v_RedGrad, int step)
{
	if (v_Coord.size() < 10 * step)
	{
		v_RedCoord = v_Coord;
		v_RedGrad = v_Grad;
		return;
	}
	int step_ = step;
	if (v_Coord.size() > 20000 * step)
	{
		step_ = v_Coord.size() / 20000;
	}
	for (int i = 0; i < v_Coord.size(); i += step_)
	{
		std::map<float, int> c_g, dst_cg;
		for (int j = i; j < i + step_; ++j)
		{
			if (j < v_Coord.size())
			{
				c_g.insert(pair<float, int>(v_Amplitude[j], j));
			}
		}
		v_RedCoord.push_back(v_Coord[c_g.rbegin()->second]);
		v_RedGrad.push_back(v_Grad[c_g.rbegin()->second]);
	}
}
//========================================================================================
