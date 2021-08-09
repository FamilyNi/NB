#pragma once
#include "utils.h"

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

/*向量归一化：
	p：[in--out]待归一化向量
*/
void Normal_PC(P_XYZ& p);

/*空间两线距离最近的点:
	line1：[in]直线1---前三个数据为方向向量，后三个为空间点
	line2：[in]直线2---前三个数据为方向向量，后三个为空间点
	pt1：[out]直线1上的点
	pt2：[out]直线2上的点
	返回为pt1、pt2距离的平方
*/
float SpaceLineNearestPt(Vec6f& line1, Vec6f& line2, P_XYZ& pt1, P_XYZ& pt2);

//四点计算球
void ComputeSphere(vector<P_XYZ>& pts, Sphere& pSphere);

/*三面求空间点:
	plane1、plane2、plane3：[in]表示三个平面
	point：[out]三面相交的点
*/
void ComputePtBasePlanes(Plane3D plane1, Plane3D plane2, Plane3D plane3, P_XYZ& point);

/*三点求面:
	pts1、pts2、pts3：[in]输入三个点
	plane：[out]计算得到的平面
	注：三个平面的法向量a、b、c必须是归一化的
*/
void ComputePlane(P_XYZ& pt1, P_XYZ& pt2, P_XYZ& pt3, Plane3D& plane);