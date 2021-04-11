#include "JC_Calibrate.h"
#include "PC_Filter.h"
#include "FitModel.h"
#include "PC_Seg.h"
#include <pcl/segmentation/region_growing.h>

//数组转向量==========================================================================
void ArrayToVector(vector<cv::Point3f> &truthPoint, vector<cv::Point3f> &calibPoint, int mode)
{
	if (mode == 1)
	{
		truthPoint.resize(24);
		calibPoint.resize(24);
		for (int i = 0; i < 24; ++i)
		{
			truthPoint[i].x = CamData1_T[3 * i];
			truthPoint[i].y = CamData1_T[3 * i + 1];
			truthPoint[i].z = CamData1_T[3 * i + 2];

			calibPoint[i].x = CamData1[3 * i];
			calibPoint[i].y = CamData1[3 * i + 1];
			calibPoint[i].z = CamData1[3 * i + 2];
		}
	}
	else if (mode == 2)
	{
		truthPoint.resize(24);
		calibPoint.resize(24);
		for (int i = 0; i < 24; ++i)
		{
			truthPoint[i].x = CamData2_T[3 * i];
			truthPoint[i].y = CamData2_T[3 * i + 1];
			truthPoint[i].z = CamData2_T[3 * i + 2];

			calibPoint[i].x = CamData2[3 * i];
			calibPoint[i].y = CamData2[3 * i + 1];
			calibPoint[i].z = CamData2[3 * i + 2];
		}
	}
	else if (mode == 3)
	{
		truthPoint.resize(12);
		calibPoint.resize(12);
		for (int i = 0; i < 12; ++i)
		{
			truthPoint[i].x = CamData3_T[3 * i];
			truthPoint[i].y = CamData3_T[3 * i + 1];
			truthPoint[i].z = CamData3_T[3 * i + 2];

			calibPoint[i].x = CamData3[3 * i];
			calibPoint[i].y = CamData3[3 * i + 1];
			calibPoint[i].z = CamData3[3 * i + 2];
		}
	}
	else if (mode == 4)
	{
		truthPoint.resize(12);
		calibPoint.resize(12);
		for (int i = 0; i < 12; ++i)
		{
			truthPoint[i].x = CamData4_T[3 * i];
			truthPoint[i].y = CamData4_T[3 * i + 1];
			truthPoint[i].z = CamData4_T[3 * i + 2];

			calibPoint[i].x = CamData4[3 * i];
			calibPoint[i].y = CamData4[3 * i + 1];
			calibPoint[i].z = CamData4[3 * i + 2];
		}
	}
}
//===================================================================================

//四点求仿射变换矩阵=================================================================
void GetTransMat(vector<cv::Point3f> &truthPoints, vector<cv::Point3f> &calibPoints, cv::Mat &transMat)
{
	if (truthPoints.size() != 4 || truthPoints.size() != calibPoints.size());
		return;
	cv::Mat MatA = cv::Mat(cv::Size(4, 4), CV_32FC1, cv::Scalar(1.0f));
	cv::Mat MatB = cv::Mat(cv::Size(4, 3), CV_32FC1, cv::Scalar(1.0f));
	float* pMatA = MatA.ptr<float>();
	float* pMatB = MatB.ptr<float>();
	for (int i = 0; i < truthPoints.size(); ++i)
	{
		pMatA[i] = calibPoints[i].x;
		pMatA[i + 4] = calibPoints[i].y;
		pMatA[i + 8] = calibPoints[i].z;

		pMatB[i] = truthPoints[i].x;
		pMatB[i + 4] = truthPoints[i].y;
		pMatB[i + 8] = truthPoints[i].z;
	}
	transMat = MatB.inv() * MatA;
}
//==================================================================================

//计算点之间的误差==================================================================
float CalError(cv::Mat &transMat, cv::Point3f &TPoint)
{
	float* pTransMat = transMat.ptr<float>();
	float error = 0.0f;
	error += abs(pTransMat[0] - TPoint.x);
	error += abs(pTransMat[1] - TPoint.y);
	error += abs(pTransMat[2] - TPoint.z);
	return error;
}
//==================================================================================

//随机采样一致算法===================================================================
void RANSAC(vector<cv::Point3f> &truthPoint, vector<cv::Point3f> &calibPoint, cv::Mat &transMat, vector<int> &index, float thres)
{
	if (truthPoint.size() < 3 || truthPoint.size() != calibPoint.size())
	{
		return;
	}
	vector<cv::Point3f> TPoints(4), CPoints(4);
	cv::Mat tranPointMat = cv::Mat(cv::Size(1, 3), CV_32FC1, cv::Scalar(0.0f));
	cv::Mat calibPointMat = cv::Mat(cv::Size(1, 4), CV_32FC1, cv::Scalar(1.0f));
	vector<cv::Point3f> bestModel;
	float P = 0.995;
	float log_P = log(1 - P);
	int size = truthPoint.size();
	int maxEpo = 100000;
	index.clear();
	for (int i = 0; i < maxEpo; ++i)
	{
		vector<int> model_index;
		int effetPoints = 0;
		int index_0 = rand() % size;
		int index_1 = rand() % size;
		int index_2 = rand() % size;
		int index_3 = rand() % size;

		model_index.push_back(index_0);
		model_index.push_back(index_1);
		model_index.push_back(index_2);
		model_index.push_back(index_3);

		cv::Mat transMat;
		TPoints[0] = truthPoint[index_0];
		TPoints[1] = truthPoint[index_1];
		TPoints[2] = truthPoint[index_2];
		TPoints[3] = truthPoint[index_3];

		CPoints[0] = calibPoint[index_0];
		CPoints[1] = calibPoint[index_1];
		CPoints[2] = calibPoint[index_2];
		CPoints[3] = calibPoint[index_3];
		GetTransMat(TPoints, CPoints, transMat);
		for (int i = 0; i < size; ++i)
		{
			if (i == index_0 || i == index_1 || i == index_2 || i == index_3)
				continue;
			float* pCalibPointMat = calibPointMat.ptr<float>();
			pCalibPointMat[0] = calibPoint[i].x;
			pCalibPointMat[1] = calibPoint[i].y;
			pCalibPointMat[2] = calibPoint[i].z;
			tranPointMat = transMat * calibPointMat;
			if (CalError(tranPointMat, truthPoint[i]) > thres)
			{
				model_index.push_back(i);
			}
		}
		if (model_index.size() > index.size())
		{
			index = model_index;
			float t_P = index.size() / size;
			maxEpo = log_P / log(1 - t_P * t_P);
		}
		if (index.size() > size / 2)
		{
			break;
		}
	}
}
//==================================================================================

//去中心化==========================================================================
void Decentration(vector<cv::Point3f> &points)
{
	float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
	for (int i = 0; i < points.size(); ++i)
	{
		sum_x += points[i].x;
		sum_y += points[i].y;
	}
	float mean_x = sum_x / points.size();
	float mean_y = sum_y / points.size();
	for (int i = 0; i < points.size(); ++i)
	{
		points[i].x -= mean_x;
		points[i].y -= mean_y;
	}
}
//==================================================================================

//==================================================================================
void LSMCalTransMat_V1(vector<cv::Point3f> &truthPoint, vector<cv::Point3f> &calibPoint, vector<double> &transMat)
{
	if (truthPoint.empty() || truthPoint.size() != calibPoint.size())
		return;
	int point_num = truthPoint.size();
	cv::Point3f sum(0.0f, 0.0f, 0.0f), sum_t(0.0f, 0.0f, 0.0f);
	cv::Point3f mean(0.0f, 0.0f, 0.0f), mean_t(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < point_num; ++i)
	{
		sum += calibPoint[i]; sum_t += truthPoint[i];
	}
	mean = sum / point_num; mean_t = sum_t / point_num;
	float xx = 0.0f, yy = 0.0f, zz = 0.0f;
	float xy = 0.0f, xz = 0.0f, yz = 0.0f;
	float x_tx = 0.0f, y_tx = 0.0f, z_tx = 0.0f;
	float x_ty = 0.0f, y_ty = 0.0f, z_ty = 0.0f;
	float x_tz = 0.0f, y_tz = 0.0f, z_tz = 0.0f;
	for (int i = 0; i < point_num; ++i)
	{
		float x_ = calibPoint[i].x - mean.x;
		float y_ = calibPoint[i].y - mean.y;
		float z_ = calibPoint[i].z - mean.z;

		float tx_ = truthPoint[i].x - mean_t.x;
		float ty_ = truthPoint[i].y - mean_t.y;
		float tz_ = truthPoint[i].z - mean_t.z;

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

//==================================================================================
void LSMCalTransMat_V2(vector<cv::Point3f> &truthPoint, vector<cv::Point3f> &calibPoint, vector<double> &transMat)
{
	cv::Mat A = cv::Mat(cv::Size(4, calibPoint.size()), CV_32FC1, cv::Scalar::all(1));
	cv::Mat b = cv::Mat(cv::Size(1, truthPoint.size()), CV_32FC1, cv::Scalar::all(1));
	for (int i = 0; i < calibPoint.size(); ++i)
	{
		float* pA = A.ptr<float>(i);
		pA[0] = calibPoint[i].x;
		pA[1] = calibPoint[i].y;
		pA[2] = calibPoint[i].z;
	}
	cv::Mat a_t = (A.t() * A).inv() * (A.t());
	//求解x
	for (int i = 0; i < calibPoint.size(); ++i)
	{
		float* pB = b.ptr<float>(i);
		pB[0] = truthPoint[i].x;
	}
	cv::Mat x = a_t * b;
	float* pX = x.ptr<float>(0);
	transMat[0] = pX[0]; transMat[1] = pX[1]; transMat[2] = pX[2]; transMat[3] = pX[3];

	//求解y
	for (int i = 0; i < calibPoint.size(); ++i)
	{
		float* pB = b.ptr<float>(i);
		pB[0] = truthPoint[i].y;
	}
	cv::Mat y = a_t * b;
	float* pY = y.ptr<float>(0);
	transMat[4] = pY[0]; transMat[5] = pY[1]; transMat[6] = pY[2]; transMat[7] = pY[3];

	//求解z
	for (int i = 0; i < calibPoint.size(); ++i)
	{
		float* pB = b.ptr<float>(i);
		pB[0] = truthPoint[i].z;
	}
	cv::Mat z = a_t * b;
	float* pZ = z.ptr<float>(0);
	transMat[8] = pZ[0]; transMat[9] = pZ[1]; transMat[10] = pZ[2]; transMat[11] = pZ[3];
}
//==================================================================================

//标定测试程序
void CalibTest()
{
	const std::string cabRotPath = "D:/JC_Config";
	//相机1
	vector<cv::Point3f> truthPoint1, calibPoint1;
	ArrayToVector(truthPoint1, calibPoint1, 1);
	vector<double> transMat1(12);
	LSMCalTransMat_V2(truthPoint1, calibPoint1, transMat1);
	PC_XYZ::Ptr pc1(new PC_XYZ());
	string path1 = "D:/JC_Config/整体点云/PC_1.ply";
	ReadPointCloud(path1, pc1);
	PC_XYZ::Ptr pc1_t(new PC_XYZ());
	//PC_Transform(pc1, pc1_t, transMat1.data());

	//相机2
	vector<cv::Point3f> truthPoint2, calibPoint2;
	ArrayToVector(truthPoint2, calibPoint2, 2);
	vector<double> transMat2(12);
	LSMCalTransMat_V2(truthPoint2, calibPoint2, transMat2);
	PC_XYZ::Ptr pc2(new PC_XYZ());
	string path2 = "D:/JC_Config/整体点云/PC_2.ply";
	ReadPointCloud(path2, pc2);
	PC_XYZ::Ptr pc2_t(new PC_XYZ());
	//PC_Transform(pc2, pc2_t, transMat2.data());

	//相机3
	vector<cv::Point3f> truthPoint3, calibPoint3;
	ArrayToVector(truthPoint3, calibPoint3, 3);
	vector<double> transMat3(12);
	LSMCalTransMat_V2(truthPoint3, calibPoint3, transMat3);
	PC_XYZ::Ptr pc3(new PC_XYZ());
	string path3 = "D:/JC_Config/整体点云/PC_3.ply";
	ReadPointCloud(path3, pc3);
	PC_XYZ::Ptr pc3_t(new PC_XYZ());
	//PC_Transform(pc3, pc3_t, transMat3.data());

	//相机4
	vector<cv::Point3f> truthPoint4, calibPoint4;
	ArrayToVector(truthPoint4, calibPoint4, 4);
	vector<double> transMat4(12);
	LSMCalTransMat_V2(truthPoint4, calibPoint4, transMat4);
	PC_XYZ::Ptr pc4(new PC_XYZ());
	string path4 = "D:/JC_Config/整体点云/PC_4.ply";
	ReadPointCloud(path4, pc4);
	PC_XYZ::Ptr pc4_t(new PC_XYZ());
	//PC_Transform(pc4, pc4_t, transMat4.data());

	pcl::visualization::PCLVisualizer viewer;
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> white(pc1_t, 255, 255, 255); //设置点云颜色
	viewer.addPointCloud(pc1_t, white, "pc1_t");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "pc1_t");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(pc2_t, 255, 0, 0);
	viewer.addPointCloud(pc2_t, red, "pc2_t");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "pc2_t");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(pc3_t, 0, 255, 0);
	viewer.addPointCloud(pc3_t, green, "pc3_t");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "pc3_t");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> blue(pc4_t, 0, 0, 255);
	viewer.addPointCloud(pc4_t, blue, "pc4_t");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "pc4_t");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}