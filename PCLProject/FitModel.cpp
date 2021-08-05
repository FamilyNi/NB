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

//四点计算球=====================================================================
void ComputeSphere(vector<P_XYZ>& pts, double* pSphere)
{
	if (pts.size() < 4)
		return;
	cv::Mat XYZ(cv::Size(3, 3), CV_64FC1, cv::Scalar(0));
	double* pXYZ = XYZ.ptr<double>();
	cv::Mat m(cv::Size(1, 3), CV_64FC1, cv::Scalar(0));
	double* pM = m.ptr<double>();
	for (int i = 0; i < pts.size() - 1; ++i)
	{
		int idx = 3 * i;
		pXYZ[idx] = pts[i].x - pts[i+1].x;
		pXYZ[idx + 1] = pts[i].y - pts[i + 1].y;
		pXYZ[idx + 2] = pts[i].z - pts[i + 1].z;

		double pt0_d = pts[i].x * pts[i].x + pts[i].y * pts[i].y + pts[i].z * pts[i].z;
		double pt1_d = pts[i+1].x * pts[i+1].x + pts[i+1].y * pts[i+1].y + pts[i+1].z * pts[i+1].z;
		pM[i] = (pt0_d - pt1_d) / 2.0;
	}

	cv::Mat center = (XYZ.inv()) * m;
	pSphere[0] = center.ptr<double>(0)[0];
	pSphere[1] = center.ptr<double>(0)[1];
	pSphere[2] = center.ptr<double>(0)[2];
	double diff_x = pts[0].x - pSphere[0];
	double diff_y = pts[0].y - pSphere[1];
	double diff_z = pts[0].z - pSphere[2];
	pSphere[3] = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
	return;
}
//===============================================================================

//RANSAC拟合球===================================================================
void PC_RandomFitSphere(PC_XYZ::Ptr &srcPC, double thresValue)
{
	if (srcPC->empty())
		return;
	SampleConsensusModelSphere<PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelSphere<PointXYZ>(srcPC));
	RandomSampleConsensus<PointXYZ> ransac(model_p);
	ransac.setDistanceThreshold(thresValue);
	ransac.computeModel();
	Eigen::VectorXf model_coefficients;
	ransac.getModelCoefficients(model_coefficients);
}
//===============================================================================

//最小二乘法拟合球===============================================================
void PC_OLSFitSphere_(PC_XYZ::Ptr& srcPC, Sphere& sphere)
{
	if (srcPC->empty())
		return;
	double sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f, sum_xyz = 0.0f;
	int num_pts = srcPC->points.size();
	for (int i = 0; i < num_pts; ++i)
	{
		P_XYZ& p_ = srcPC->points[i];
		sum_x += p_.x; sum_y += p_.y; sum_z += p_.z;
		sum_xyz += p_.x*p_.x + p_.y*p_.y + p_.z*p_.z;
	}
	P_XYZ mean_p(sum_x / num_pts, sum_y / num_pts, sum_z / num_pts);
	float mean_xyz = sum_xyz / num_pts;
	Mat A(cv::Size(3, 3), CV_64FC1, cv::Scalar(0));
	Mat B(cv::Size(1, 3), CV_64FC1, cv::Scalar(0));
	double* pA = A.ptr<double>(0);
	double* pB = B.ptr<double>(0);
	for (int i = 0; i < num_pts; ++i)
	{
		P_XYZ& p_ = srcPC->points[i];
		double x_ = p_.x - mean_p.x;
		double y_ = p_.y - mean_p.y;
		double z_ = p_.z - mean_p.z;
		double xyz_ = p_.x*p_.x + p_.y*p_.y + p_.z*p_.z - mean_xyz;
		pB[0] -= x_ * xyz_; pB[1] -= y_ * xyz_; pB[2] -= z_ * xyz_;
		pA[0] += x_ * x_; pA[1] += x_ * y_; pA[2] += x_ * z_;
		pA[4] += y_ * y_; pA[5] += y_ * z_; pA[8] += z_ * z_;
	}
	pA[3] = pA[1]; pA[6] = pA[2]; pA[7] = pA[5];

	Mat res = (A.inv()) * B;
	double* pRes = res.ptr<double>();
	sphere.c_x = -pRes[0] / 2.0;
	sphere.c_y = -pRes[1] / 2.0;
	sphere.c_z = -pRes[2] / 2.0;
	double d = -(pRes[0] * mean_p.x + pRes[1] * mean_p.y + pRes[2] * mean_p.z + mean_xyz);
	sphere.r = std::sqrt(sphere.c_x * sphere.c_x + sphere.c_y * sphere.c_y + sphere.c_z * sphere.c_z - d);
}
//===============================================================================

/*测试程序*/
void PC_FitPlaneTest()
{
	int ttt = 0;
}