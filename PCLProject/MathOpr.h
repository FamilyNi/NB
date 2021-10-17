#pragma once
#include "utils.h"

/*点到平面的距离：
	pt：[in]输入点
	plane：[in]平面方程
	dist：[out]输出距离
*/
template <typename T1, typename T2>
void PC_PtToPlaneDist(T1& pt, T2& plane, double& dist);

/*向量归一化*/
template <typename T>
void PC_VecNormal(T& p);

/*点到平面的投影点
	pt：[in]输入点
	plane：[in]平面方程
	projPt：[out]投影点
*/
template <typename T1, typename T2, typename T3>
void PC_PtProjPlanePt(T1& pt, T2& plane, T3& projPt);

/*空间点到空间直线的距离
	pt：[in]输入点
	line：[in]直线方程
	dist：[out]输出距离
*/
template <typename T1, typename T2>
void PC_PtToLineDist(T1& pt, T2& line, double& dist);

/*空间点到空间直线的投影
	pt：[in]输入点
	line：[in]直线方程
	projPt：[out]投影点
*/
template <typename T1, typename T2, typename T3>
void PC_PtProjLinePt(T1& pt, T2& line, T3& projPt);

/*三维向量叉乘
	vec1、vec2：[in]表示向量1、2
	vec：[out]叉乘后的结果
*/
template <typename T1, typename T2, typename T3>
void PC_VecCross(T1& vec1, T2& vec2, T3& vec, bool isNormal);

/*计算两点之间的距离--二维*/
template <typename T1, typename T2>
void Img_ComputePPDist(T1& pt1, T2& pt2, double& dist);

/*平面上点到直线的投影
	pt：[in]输入点
	line：[in]直线方程
	projPt：[out]投影点
*/
template <typename T1, typename T2, typename T3>
void Img_PtProjLinePt(T1& pt, T2& line, T3& projPt);

/*罗格里格斯公式：
	rotAxis：[in]旋转轴
	rotAng：[in]两向量之间的夹角
	rotMat：[out]旋转矩阵
*/
template <typename T>
void RodriguesFormula(T& rotAxis, double rotAng, cv::Mat& rotMat);

