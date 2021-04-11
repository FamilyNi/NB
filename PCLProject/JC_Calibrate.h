#pragma once
#include "utils.h"

const float CamData1[72] = { 22.34511, 180.393776, 4.440888, 46.480332, 180.433246, 4.967454, 70.447672, 180.451374, 5.461712, 94.45198, 180.513628, 5.947162, 22.95738,
						216.639888, -2.187154, 47.086124, 216.631518, -1.672338, 71.06871, 216.687402, -1.195288, 95.044326, 216.754128, -0.718002, 23.573504, 252.846312,
						-8.782996, 47.704204, 252.878102, -8.256038, 71.70445, 252.9142, -7.748848, 95.641044, 253.00124, -7.350074, 24.453098, 332.519856, -9.646054,
						48.57752, 332.568896, -9.086526, 72.548276, 332.615704, -8.603384, 96.533024, 332.67457, -8.184292, 24.929798, 375.801052, -3.96811, 49.009306,
						375.843238, -3.408676, 72.994908, 375.909952, -2.8712, 97.00792, 375.942342, -2.476708, 25.403946, 419.10606, 1.8466, 49.507878, 419.136466, 2.355808,
						73.48378, 419.165554, 2.85379, 97.475062, 419.23875, 3.288734 };

const float CamData2[72] = { 33.806892, 111.640434, 9.244512, 57.850734, 112.118338, 8.99824, 81.879004, 112.608434, 8.86546, 106.158692, 113.204482, 9.244604, 33.391634,
						155.312314, 2.550272, 57.466162, 155.79407, 2.269378, 81.489258, 156.331458, 2.126852, 105.780032, 156.921638, 2.472972, 32.846484, 199.134068,
						-4.131978, 56.924234, 199.588364, -4.339732, 80.943196, 200.109404, -4.497912, 105.206442, 200.694322, -4.189868, 31.733968, 279.91098, -5.088496,
						55.804528, 280.35097, -5.308496, 79.852584, 280.799656, -5.353412, 104.115508, 281.43089, -5.114552, 31.420864, 316.666326, 0.736032, 55.510146,
						317.140756, 0.505846, 79.518828, 317.652984, 0.29607, 103.759574, 318.242582, 0.581392, 31.019432, 353.389092, 6.467646, 55.105012, 353.87425,
						6.251622, 79.12896, 354.34982, 6.073114, 103.359984, 354.94246, 6.36874 };

const float CamData3[36] = { 59.352112, 165.142198, 25.785308, 63.726654, 205.123358, 22.351684, 68.003584, 245.15806, 19.02532, 69.149506, 325.368664, 19.183842, 66.112948,
						365.372138, 22.747672, 63.04921, 405.326554, 26.37878, 76.094878, 165.06425, 42.939236, 80.293114, 205.075846, 39.490738, 84.635098, 245.145492,
						36.211624, 85.812452, 325.379796, 36.36158, 82.756326, 365.343704, 39.931812, 79.697738, 405.325022, 43.538084 };

const float CamData4[36] = { 68.30064, 126.911916, 34.776996, 71.867962, 166.774476, 30.62107, 75.303536, 206.74817, 26.413906, 75.078724, 287.02271, 25.15503, 71.509482,
						327.132378, 28.166864, 67.848866, 367.105798, 31.186036, 85.34766, 126.78941, 51.754194, 88.949002, 166.648676, 47.659836, 92.402042, 206.680432,
						43.477744, 92.178438, 286.91597, 42.216804, 88.591492, 326.993808, 45.16815, 84.901358, 367.020916, 48.178452 };

const float CamData1_T[72] = { 96, 45, 10, 72, 45, 10, 48, 45, 10, 24, 45, 10, 96, 85, 15, 72, 85, 15, 48, 85, 15, 24, 85, 15, 96, 125, 20, 72, 125, 20, 48, 125, 20, 24, 125, 20,
						96, 205, 20, 72, 205, 20, 48, 205, 20, 24, 205, 20, 96, 245, 15, 72, 245, 15, 48, 245, 15, 24, 245, 15, 96, 285, 10, 72, 285, 10, 48, 285, 10, 24, 285, 10 };

const float CamData2_T[72] = { 24, 35, 10, 48, 35, 10, 72, 35, 10, 96, 35, 10, 24, 75, 15, 48, 75, 15, 72, 75, 15, 96, 75, 15, 24, 115, 20, 48, 115, 20, 72, 115, 20, 96, 115, 20,
						24, 195, 20, 48, 195, 20, 72, 195, 20, 96, 195, 20, 24, 235, 15, 48, 235, 15, 72, 235, 15, 96, 235, 15, 24, 275, 10, 48, 275, 10, 72, 275, 10, 96, 275, 10 };

const float CamData3_T[36] = { 101, 40, 10, 101, 80, 15, 101, 120, 20, 101, 200, 20, 101, 240, 15, 101, 280, 10,
						77, 40, 10, 77, 80, 15, 77, 120, 20, 77, 200, 20, 77, 240, 15, 77, 280, 10 };

const float CamData4_T[36] = { 19, 40, 10, 19, 80, 15, 19, 120, 20, 19, 200, 20, 19, 240, 15, 19, 280, 10,
						43, 40, 10, 43, 80, 15, 43, 120, 20, 43, 200, 20, 43, 240, 15, 43, 280, 10 };

//数组转向量
void ArrayToVector(vector<cv::Point3f> &truthPoint, vector<cv::Point3f> &calibPoint, int mode);

//四点求仿射变换矩阵
void GetTransMat(vector<cv::Point3f> &truthPoints, vector<cv::Point3f> &calibPoints, cv::Mat &transMat);

//计算点之间的误差
float CalError(cv::Mat &transMat, cv::Point3f &TPoint);

//随机采样一致算法
void RANSAC(vector<cv::Point3f> &truthPoint, vector<cv::Point3f> &calibPoint, cv::Mat &transMat, vector<int> &index, float thres = 0.1f);

//最小二乘法求解变换矩阵
void LSMCalTransMat_V1(vector<cv::Point3f> &truthPoint, vector<cv::Point3f> &calibPoint, vector<double> &transMat);

void LSMCalTransMat_V2(vector<cv::Point3f> &truthPoint, vector<cv::Point3f> &calibPoint, vector<double> &transMat);

//标定测试程序
void CalibTest();