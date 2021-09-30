#pragma once
#include "utils.h"

//点到平面的距离
void PC_PtToPlaneDist(P_XYZ& pt, cv::Vec4d& plane, double& dist);

//向量归一化
template <typename T>
void PC_VecNormal(T& p);

//点到平面的投影点
void PC_PtProjPlanePt(P_XYZ& pt, cv::Vec4d& plane, P_XYZ& projPt);

//空间点到空间直线的距离
void PC_PtToLineDist(P_XYZ& pt, cv::Vec6d& line, double& dist);

//空间点到空间直线的投影
void PC_PtProjLinePt(P_XYZ& pt, cv::Vec6d& line, P_XYZ& projPt);

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
template <typename T>
void VecCross_PC(T& vec1, T& vec2, T& vec);

/*计算两点之间的距离--二维*/
template <typename T>
void Img_ComputePPDist(T& pt1, T& pt2, double& dist);