#include "JC_Calibrate.h"
#include <pcl/segmentation/region_growing.h>
#include "include/BaseOprFile/FileOpr.h"
#include "include/PointCloudFile/PC_Filter.h"
#include "include/PointCloudFile/PC_Seg.h"

//读取世界坐标========================================================================
void ReadWorldCoord(const string& filename, vector<vector<cv::Point3f>>& worldPts)
{
	vector<vector<string>> strArray(0);
	if (ReadCSVFile(filename, strArray) != 0)
		return;
	worldPts.resize(0);
	if (strArray.size() == 0)
		return;
	//第一行为标签
	size_t num = strArray.size() - 1;
	worldPts.resize(num);
	for (size_t i = 0; i < num; ++i)
	{
		//第一列为点序号
		size_t pt_num = (strArray[i].size()-1) / 3;
		worldPts[i].resize(pt_num);
		for (size_t j = 0; j < pt_num; ++j)
		{
			//strArray从第二行开始与第二列开始
			worldPts[i][j].x = atof(strArray[i+1][3 * j + 1].c_str());
			worldPts[i][j].y = atof(strArray[i+1][3 * j + 2].c_str());
			worldPts[i][j].z = atof(strArray[i+1][3 * j + 3].c_str());
		}
	}
}
//====================================================================================

//读取相机坐标========================================================================
void ReadCamCoord(const string& filename, vector<cv::Point3f>& camPts)
{
	vector<vector<string>> strArray(0);
	if (ReadCSVFile(filename, strArray) != 0)
		return;
	camPts.resize(0);
	if (strArray.size() < 2)
		return;
	//第一行为标签,第一列为日期
	size_t num = (strArray[1].size() - 1) / 3;
	camPts.resize(num);
	for (size_t i = 0; i < num; ++i)
	{
		//strArray从第二行开始与第二列开始
		camPts[i].x = atof(strArray[1][3 * i + 1].c_str());
		camPts[i].y = atof(strArray[1][3 * i + 2].c_str());
		camPts[i].z = atof(strArray[1][3 * i + 3].c_str());
	}
}
//====================================================================================

//标定相机1===========================================================================
float CablibCam_1(vector<vector<cv::Point3f>>& worldPts, vector<cv::Point3f>& camPts, vector<double> &transMat)
{
	//获取相机1坐标对应的世界坐标
	size_t pt_num = camPts.size();
	if (worldPts.size() != pt_num)
		return -1;
	if (transMat.size() != 12)
		transMat.resize(12);
	vector<cv::Point3f> cam1_worldPts(pt_num);
	for (size_t i = 0; i < pt_num; ++i)
	{
		cam1_worldPts[i] = worldPts[i][1];
	}
	LagrangeSolveTLS(cam1_worldPts, camPts, transMat, 0.001);

	//计算误差
	return CalError(cam1_worldPts, camPts, transMat);
}
//====================================================================================

//四点求仿射变换矩阵=================================================================
void GetTransMat(vector<cv::Point3f> &worldPts, vector<cv::Point3f> &camPts, cv::Mat &transMat)
{
	if (worldPts.size() != 4 || worldPts.size() != camPts.size());
		return;
	cv::Mat MatA = cv::Mat(cv::Size(4, 4), CV_32FC1, cv::Scalar(1.0f));
	cv::Mat MatB = cv::Mat(cv::Size(4, 3), CV_32FC1, cv::Scalar(1.0f));
	float* pMatA = MatA.ptr<float>();
	float* pMatB = MatB.ptr<float>();
	for (int i = 0; i < worldPts.size(); ++i)
	{
		pMatA[i] = camPts[i].x;
		pMatA[i + 4] = camPts[i].y;
		pMatA[i + 8] = camPts[i].z;

		pMatB[i] = worldPts[i].x;
		pMatB[i + 4] = worldPts[i].y;
		pMatB[i + 8] = worldPts[i].z;
	}
	transMat = MatB.inv() * MatA;
}
//==================================================================================

//去中心化==========================================================================
void PtDecentration(vector<cv::Point3f>& srcPts, vector<cv::Point3f>& dstPts)
{
	float sum_x = 0.0f;
	float sum_y = 0.0f;
	float sum_z = 0.0f;
	for (size_t i = 0; i < srcPts.size(); ++i)
	{
		sum_x += srcPts[i].x;
		sum_y += srcPts[i].y;
		sum_z += srcPts[i].z;
	}
	float mean_x = sum_x / (float)srcPts.size();
	float mean_y = sum_y / (float)srcPts.size();
	float mean_z = sum_z / (float)srcPts.size();
	dstPts.resize(srcPts.size());
	for (size_t i = 0; i < srcPts.size(); ++i)
	{
		dstPts[i].x = srcPts[i].x - mean_x;
		dstPts[i].y = srcPts[i].y - mean_y;
		dstPts[i].z = srcPts[i].z - mean_z;
	}
}
//==================================================================================

//计算标定误差======================================================================
float CalError(vector<cv::Point3f>& worldPts, vector<cv::Point3f>& camPts, vector<double>& calibMat)
{
	if (camPts.size() != worldPts.size() || calibMat.size() != 12)
		return -1;
	float error = 0.0f;
	for (size_t i = 0; i < camPts.size(); ++i)
	{
		cv::Point3f p_;
		cv::Point3f& camPt = camPts[i];
		p_.x = camPt.x * calibMat[0] + camPt.y * calibMat[1] + camPt.z * calibMat[2] + calibMat[3];
		p_.y = camPt.x * calibMat[4] + camPt.y * calibMat[5] + camPt.z * calibMat[6] + calibMat[7];
		p_.z = camPt.x * calibMat[8] + camPt.y * calibMat[9] + camPt.z * calibMat[10] + calibMat[11];
		cv::Point3f& worldPt = worldPts[i];
		error += std::pow(worldPt.x - p_.x, 2) + std::pow(worldPt.y - p_.y, 2) + std::pow(worldPt.z - p_.z, 2);
	}
	error /= (float)camPts.size();
	return error;
}
//==================================================================================

//最小二乘法求解变换矩阵============================================================
void LSMCalTransMat(vector<cv::Point3f> &worldPts, vector<cv::Point3f> &camPts, vector<double> &transMat)
{
	if (worldPts.empty() || worldPts.size() != camPts.size())
		return;
	int point_num = worldPts.size();
	cv::Point3f sum(0.0f, 0.0f, 0.0f), sum_t(0.0f, 0.0f, 0.0f);
	cv::Point3f mean(0.0f, 0.0f, 0.0f), mean_t(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < point_num; ++i)
	{
		sum += camPts[i]; sum_t += worldPts[i];
	}
	mean = sum / point_num; mean_t = sum_t / point_num;
	float xx = 0.0f, yy = 0.0f, zz = 0.0f;
	float xy = 0.0f, xz = 0.0f, yz = 0.0f;
	float x_tx = 0.0f, y_tx = 0.0f, z_tx = 0.0f;
	float x_ty = 0.0f, y_ty = 0.0f, z_ty = 0.0f;
	float x_tz = 0.0f, y_tz = 0.0f, z_tz = 0.0f;
	for (int i = 0; i < point_num; ++i)
	{
		float x_ = camPts[i].x - mean.x;
		float y_ = camPts[i].y - mean.y;
		float z_ = camPts[i].z - mean.z;

		float tx_ = worldPts[i].x - mean_t.x;
		float ty_ = worldPts[i].y - mean_t.y;
		float tz_ = worldPts[i].z - mean_t.z;

		xx += x_ * x_; yy += y_ * y_; zz += z_ * z_;
		xy += x_ * y_; xz += x_ * z_; yz += y_ * z_;

		x_tx += x_ * tx_; y_tx += y_ * tx_; z_tx += z_ * tx_;
		x_ty += x_ * ty_; y_ty += y_ * ty_; z_ty += z_ * ty_;
		x_tz += x_ * tz_; y_tz += y_ * tz_; z_tz += z_ * tz_;
	}

	//求解x
	cv::Mat A = cv::Mat(cv::Size(3, 3), CV_32FC1, cv::Scalar(point_num));
	cv::Mat B = cv::Mat(cv::Size(1, 3), CV_32FC1, cv::Scalar(0));
	float *pA = A.ptr<float>(0);
	float *pB = B.ptr<float>(0);
	pA[0] = xx; pA[1] = xy; pA[2] = xz;
	pA[3] = xy; pA[4] = yy; pA[5] = yz;
	pA[6] = xz; pA[7] = yz; pA[8] = zz;
	pB[0] = x_tx; pB[1] = y_tx; pB[2] = z_tx;
	cv::Mat transX = A.inv() * B;
	float* pTranX = transX.ptr<float>(0);
	transMat[0] = pTranX[0]; transMat[1] = pTranX[1]; transMat[2] = pTranX[2]; 
	transMat[3] = mean_t.x - pTranX[0] * mean.x - pTranX[1] * mean.y - pTranX[2] * mean.z;

	//求解y
	pB[0] = x_ty; pB[1] = y_ty; pB[2] = z_ty;
	cv::Mat transY = A.inv() * B;
	float* pTranY = transY.ptr<float>(0);
	transMat[4] = pTranY[0]; transMat[5] = pTranY[1]; transMat[6] = pTranY[2];
	transMat[7] = mean_t.y - pTranY[0] * mean.x - pTranY[1] * mean.y - pTranY[2] * mean.z;

	//求解z
	pB[0] = x_tz; pB[1] = y_tz; pB[2] = z_tz;
	cv::Mat transZ = A.inv() * B;
	float* pTranZ = transZ.ptr<float>(0);
	transMat[8] = pTranZ[0]; transMat[9] = pTranZ[1]; transMat[10] = pTranZ[2];
	transMat[11] = mean_t.z - pTranZ[0] * mean.x - pTranZ[1] * mean.y - pTranZ[2] * mean.z;
}
//==================================================================================

//拉格朗日方法求解总体最小二乘法====================================================
void LagrangeSolveTLS(vector<cv::Point3f> &worldPts, vector<cv::Point3f> &camPts, vector<double> &transMat, float thresVal)
{
	//计算初始变换矩阵
	vector<cv::Point3f> worldPts_d, camPts_d;
	PtDecentration(worldPts, worldPts_d);
	PtDecentration(camPts, camPts_d);
	
	//向量转Mat
	cv::Mat worldMatPt(cv::Size(3, worldPts.size()), CV_32FC1);
	cv::Mat camdMatPt(cv::Size(3, worldPts.size()), CV_32FC1);
	for (int i = 0; i < worldPts.size(); ++i)
	{
		worldMatPt.at<float>(i, 0) = worldPts_d[i].x;
		worldMatPt.at<float>(i, 1) = worldPts_d[i].y;
		worldMatPt.at<float>(i, 2) = worldPts_d[i].z;

		camdMatPt.at<float>(i, 0) = camPts_d[i].x;
		camdMatPt.at<float>(i, 1) = camPts_d[i].y;
		camdMatPt.at<float>(i, 2) = camPts_d[i].z;
	}

	cv::Mat A_T_A_INV = ((camdMatPt.t()) * camdMatPt).inv();
	cv::Mat calibMat = A_T_A_INV * (camdMatPt.t() * worldMatPt);
	cv::Mat calibMat_ = calibMat.clone();

	cv::Mat I_Mat = cv::Mat::eye(3, 3, CV_32FC1);
	for (int iter = 0; iter < 10; ++iter)
	{
		cv::Mat EE = (calibMat_.t()) * calibMat_;
		cv::Mat Y_AE = worldMatPt - camdMatPt * calibMat_;
		cv::Mat N = ((I_Mat + EE).inv()) * (Y_AE.t()) * Y_AE;

		cv::Mat AYEN = (camdMatPt.t()) * worldMatPt + calibMat_ * N;
		calibMat = A_T_A_INV * AYEN;

		cv::Mat diff_calib = calibMat - calibMat_;
		float* pDiff = diff_calib.ptr<float>();
		float sum_diff = 0.0f;
		for (int i = 0; i < 9; ++i)
			sum_diff += (pDiff[i] * pDiff[i]);
		sum_diff = std::sqrt(sum_diff);
		if (sum_diff < 1e-8)
			break;
		calibMat_ = calibMat.clone();
	}
	calibMat = calibMat.t();
	transMat.resize(12);
	for (int y = 0; y < 3; ++y)
	{
		float* pCalibMat = calibMat.ptr<float>(y);
		int index = 4 * y;
		for (int x = 0; x < 3; ++x)
		{
			transMat[index + x] = pCalibMat[x];
		}
	}
}
//==================================================================================

//标定测试程序
void CalibTest()
{
	const string calibFile = "D:/JC_Config/相机数据/C5相机/20210703标定/CaliParaSensor1.clb";
	vector<double> data(12);
	readBinFile(calibFile, data);

	//读取世界坐标
	string w_filename = "G:/调试软件20210609/User/CaliStdBlockData.csv";
	vector<vector<cv::Point3f>> worldPts;
	ReadWorldCoord(w_filename, worldPts);

	//读取相机坐标
	string c_filename = "D:/JC_Config/相机数据/C5相机/20210703标定/1-DataLog.csv";
	vector<cv::Point3f> camPts;
	ReadCamCoord(c_filename, camPts);

	//标定相机1
	vector<double> transMat(12);
	float error = CablibCam_1(worldPts, camPts, transMat);
}