#include "FitModel.h"
#include "PointCloudOpr.h"

//RANSAC拟合平面=================================================================
void PC_RandomFitPlane(PC_XYZ::Ptr &srcPC, vector<int> &inliers, double thresValue)
{
	if (srcPC->points.size() == 0)
		return;
	if (!inliers.empty())
		inliers.clear();
	SampleConsensusModelPlane<PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointXYZ>(srcPC));
	RandomSampleConsensus<PointXYZ> ransac(model_p);
	ransac.setDistanceThreshold(thresValue);
	ransac.computeModel();
	ransac.getInliers(inliers);
}
//===============================================================================

//最小二乘法拟合平面=============================================================
void PC_OLSFitPlane(PC_XYZ::Ptr& srcPC, Plane3D &plane)
{
	cv::Mat covMat = cv::Mat(cv::Size(3, 3), CV_32FC1, cv::Scalar(0));
	P_XYZ gravity{ 0.0f, 0.0f, 0.0f };
	PC_ComputeCovMat(srcPC, covMat, gravity);

	cv::Mat eigenVal, eigenVec;
	cv::eigen(covMat, eigenVal, eigenVec);
	double minAbsEigenVal = abs(eigenVal.ptr<double>(0)[0]);
	int index = 0;
	for (int i = 1; i < eigenVal.rows; ++i)
	{
		if (minAbsEigenVal > abs(eigenVal.ptr<double>(i)[0]))
		{
			minAbsEigenVal = abs(eigenVal.ptr<double>(i)[0]);
			index = i;
		}
	}
	plane.a = eigenVec.ptr<double>(index)[0];
	plane.b = eigenVec.ptr<double>(index)[1];
	plane.c = eigenVec.ptr<double>(index)[2];
	plane.d = plane.a * gravity.x + plane.b * gravity.y + plane.c * gravity.z;
}
//===============================================================================

//基于权重的平面拟合=============================================================
void FitPlaneBaseOnWeight(PC_XYZ::Ptr &srcPC, P_N &normal, uint iter_k)
{
	size_t length = srcPC->points.size();
	PC_XYZ::Ptr w_srcPC(new PC_XYZ);
	w_srcPC->points.resize(length);
	for (size_t i = 0; i < length; ++i)
	{
		w_srcPC->points[i] = srcPC->points[i];
	}
	cv::Mat covMat(cv::Size(3, 3), CV_32FC1, cv::Scalar(0));
	cv::Mat eigenVal, eigenVec;
	P_XYZ gravity{ 0.0f, 0.0f, 0.0f };
	for (uint i = 0; i < iter_k; ++i)
	{
		PC_ComputeCovMat(w_srcPC, covMat, gravity);
		cv::eigen(covMat, eigenVal, eigenVec);
		float* pEigenVal = eigenVec.ptr<float>(2);
		float d = -(pEigenVal[0] * gravity.x + pEigenVal[1] * gravity.y + pEigenVal[2] * gravity.z);
		for (size_t j = 0; j < length; ++j)
		{
			P_XYZ& w_p = w_srcPC->points[j];
			P_XYZ& p_ = srcPC->points[j];
			float w_ = std::expf(-fabs(pEigenVal[0] * w_p.x + pEigenVal[1] * w_p.y + pEigenVal[2] * w_p.z + d));
			w_p.x = w_ * p_.x; w_p.y = w_ * p_.y; w_p.z = w_ * p_.z;
		}
	}
	normal.normal_x = eigenVec.ptr<float>(2)[0];
	normal.normal_y = eigenVec.ptr<float>(2)[1];
	normal.normal_z = eigenVec.ptr<float>(2)[2];
}
//===============================================================================

/*测试程序*/
void PC_FitPlaneTest()
{
	int ttt = 0;
}