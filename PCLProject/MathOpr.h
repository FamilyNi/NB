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

/*空间两线距离最近的点:
	line1：[in]直线1---前三个数据为方向向量，后三个为空间点
	line2：[in]直线2---前三个数据为方向向量，后三个为空间点
	pt1：[out]直线1上的点
	pt2：[out]直线2上的点
	返回为pt1、pt2距离的平方
*/
float SpaceLineNearestPt(Vec6f& line1, Vec6f& line2, P_XYZ& pt1, P_XYZ& pt2);

//四点计算球
void ComputeSphere(vector<P_XYZ>& pts, double* pSphere);;