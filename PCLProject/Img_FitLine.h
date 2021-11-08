#pragma once
#include "utils.h"
#include "OpenCV_Utils.h"

/*说明：
	直线方程： a * x + b * y + c = 0
	pts：[in]平面上的点簇
*/

/*随机一致采样算法计算直线
	pts：[in]平面上的点簇
	line：[out]输出的圆---[0]：x方向上法向量、
				[1]：y方向的法向量、[2]：c值
	inliners：[in]局内点
	thres：[in]阈值
*/
void Img_RANSACFitLine(NB_Array2D pts, Line2D& line, vector<int>& inliners, double thres);

/*最小二乘法拟合直线：
	weights：[in]权重
	line：[out]
*/
void Img_OLSFitLine(NB_Array2D pts, vector<double>& weights, Line2D& line);

/*Huber计算权重：
	line：[in]
	weights：[out]权重
*/
void Img_HuberLineWeights(NB_Array2D pts, Line2D& line, vector<double>& weights);

/*Tukey计算权重：
	line：[in]
	weights：[out]权重
*/
void Img_TukeyLineWeights(NB_Array2D pts, Line2D& line, vector<double>& weights);

/*直线拟合：
	line：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、turkey
*/
void Img_FitLine(NB_Array2D pts, Line2D& line, int k, NB_MODEL_FIT_METHOD method);

//二维直线拟合测试
void Img_FitLineTest();

