#pragma once
#include "utils.h"

//点到平面的距离
double PC_PtToPlaneDist(P_XYZ& pt, cv::Vec4d& plane);

//向量归一化
template <typename T>
void PC_VecNormal(T& p);

//点到平面的投影点
void PC_PtProjPlanePt(P_XYZ& pt, cv::Vec4d& plane, P_XYZ& projPt);

//空间点到空间直线的距离
double PC_3DPtTo3DLineDist(P_XYZ& pt, cv::Vec6d& line);

//空间点到空间直线的投影
void PC_3DPtProjLinePt(P_XYZ& pt, cv::Vec6d& line, P_XYZ& projPt);

//十进制转二进制
void DecToBin(const int dec_num, vector<bool>& bin);

//二进制转十进制
void BinToDec(const vector<bool>& bin, int& dec_num);

//二进制转格雷码
void BinToGrayCode(const vector<bool>& bin, vector<bool>& grayCode);

//格雷码转二进制
void GrayCodeToBin(const vector<bool>& grayCode, vector<bool>& bin);

/*三维向量叉乘:
	vec1、vec2：[in]表示向量1、2
	vec：[out]叉乘后的结果
*/
void VecCross_PC(P_XYZ& vec1, P_XYZ& vec2, P_XYZ& vec);


/*空间两线距离最近的点:
	line1：[in]直线1---前三个数据为方向向量，后三个为空间点
	line2：[in]直线2---前三个数据为方向向量，后三个为空间点
	pt1：[out]直线1上的点
	pt2：[out]直线2上的点
	返回为pt1、pt2距离的平方
*/
float SpaceLineNearestPt(Vec6f& line1, Vec6f& line2, P_XYZ& pt1, P_XYZ& pt2);

/*三面求空间点:
	plane1、plane2、plane3：[in]表示三个平面
	point：[out]三面相交的点
*/
void ComputePtBasePlanes(Plane3D plane1, Plane3D plane2, Plane3D plane3, P_XYZ& point);


/*计算两点之间的距离--二维*/
template <typename T1, typename T2>
double Img_ComputePPDist(T1& pt1, T2& pt2);