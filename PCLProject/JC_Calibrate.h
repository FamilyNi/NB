#pragma once
#include "utils.h"

//读取世界坐标
void ReadWorldCoord(const string& filename, vector<vector<cv::Point3f>>& worldPts);

//读取相机坐标
void ReadCamCoord(const string& filename, vector<cv::Point3f>& camPts);

//标定相机1
float CablibCam_1(vector<vector<cv::Point3f>>& worldPts, vector<cv::Point3f>& camPts, vector<double> &transMat);

//四点求仿射变换矩阵
void GetTransMat(vector<cv::Point3f> &worldPts, vector<cv::Point3f> &camPts, cv::Mat &transMat);

//去中心化
void PtDecentration(vector<cv::Point3f>& srcPts, vector<cv::Point3f>& dstPts);

//计算标定误差
float CalError(vector<cv::Point3f>& worldPts, vector<cv::Point3f>& camPts, vector<double>& calibMat);

//最小二乘法求解变换矩阵：AX = B
void LSMCalTransMat(vector<cv::Point3f> &worldPts, vector<cv::Point3f> &camPts, vector<double> &transMat);

//拉格朗日方法求解总体最小二乘法
void LagrangeSolveTLS(vector<cv::Point3f> &worldPts, vector<cv::Point3f> &camPts, vector<double> &transMat, float thresVal);

//标定测试程序
void CalibTest();