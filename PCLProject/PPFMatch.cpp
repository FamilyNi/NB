#include "PPFMatch.h"
#include "PC_Filter.h"
#include "utils.h"
#include "PointCloudOpr.h"

//罗格里德斯公式====================================================================
void RodriguesFormula(P_N& rotAxis, float rotAng, cv::Mat& rotMat)
{
	float cosVal = std::cos(rotAng);
	float conVal_ = 1 - cosVal;
	float sinVal = std::sin(rotAng);
	float* pRotMat = rotMat.ptr<float>();

	pRotMat[0] = cosVal + rotAxis.normal_x * rotAxis.normal_x * conVal_;
	pRotMat[1] = rotAxis.normal_x * rotAxis.normal_y * conVal_ - rotAxis.normal_z * sinVal;
	pRotMat[2] = rotAxis.normal_x * rotAxis.normal_z * conVal_ + rotAxis.normal_y * sinVal;

	pRotMat[3] = rotAxis.normal_y * rotAxis.normal_x * conVal_ + rotAxis.normal_z * sinVal;
	pRotMat[4] = cosVal + rotAxis.normal_y * rotAxis.normal_y * conVal_;
	pRotMat[5] = rotAxis.normal_y * rotAxis.normal_z * conVal_ - rotAxis.normal_x * sinVal;

	pRotMat[6] = rotAxis.normal_z * rotAxis.normal_x * conVal_ - rotAxis.normal_y * sinVal;
	pRotMat[7] = rotAxis.normal_z * rotAxis.normal_y * conVal_ + rotAxis.normal_x * sinVal;
	pRotMat[8] = cosVal + rotAxis.normal_z * rotAxis.normal_z * conVal_;
}
//==================================================================================

//计算PPF特征=======================================================================
void ComputePPFFEATRUE(P_XYZ& ref_p, P_XYZ& p_, P_N& ref_pn, P_N& p_n, PPFFEATRUE& ppfFEATRUE)
{
	P_XYZ p_v(ref_p.x - p_.x, ref_p.y - p_.y, ref_p.z - p_.z);
	ppfFEATRUE.dist = std::sqrt(p_v.x * p_v.x + p_v.y * p_v.y + p_v.z * p_v.z);
	float normal_ = 1.0f / std::max(ppfFEATRUE.dist, EPS);
	p_v.x *= normal_; p_v.y *= normal_; p_v.z *= normal_;

	ppfFEATRUE.ang_N1N2 = acosf(ref_pn.normal_x * p_n.normal_x + ref_pn.normal_y * p_n.normal_y + ref_pn.normal_z * p_n.normal_z);
	ppfFEATRUE.ang_N1D = acosf(ref_pn.normal_x * p_v.x + ref_pn.normal_y * p_v.y + ref_pn.normal_z * p_v.z);
	ppfFEATRUE.ang_N2D = acosf(p_n.normal_x * p_v.x + p_n.normal_y * p_v.y + p_n.normal_z * p_v.z);
}
//==================================================================================

//将点对以PPF特征推送到hash表中=====================================================
void PushPPFToHashMap(hash_map<string, vector<PPFCELL>>& hashMap, PPFFEATRUE& ppfFEATRUE,
	float distStep, float stepAng, int ref_i, float alpha)
{
	int dist = ppfFEATRUE.dist / distStep;
	int ang_N1D = ppfFEATRUE.ang_N1D / stepAng;
	int ang_N2D = ppfFEATRUE.ang_N2D / stepAng;
	int ang_N1N2 = ppfFEATRUE.ang_N1N2 / stepAng;
	string hashKey = std::to_string(dist) + std::to_string(ang_N1D)	+ std::to_string(ang_N2D) + std::to_string(ang_N1N2);
	PPFCELL ppfCell;
	ppfCell.ref_alpha = alpha;
	ppfCell.ref_i = ref_i;
	hashMap[hashKey].push_back(ppfCell);
}
//==================================================================================

//提取PPF的法向量===================================================================
void ExtractPPFNormals(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& downSamplepC, PC_N::Ptr& normals, float radius)
{
	PC_N::Ptr model_pcn(new PC_N);
	PC_ComputePCNormal(srcPC, model_pcn, radius);
	size_t length = downSamplepC->size();
	normals->points.resize(length);
	pcl::KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC);
	for (size_t i = 0; i < length; ++i)
	{
		vector<int> PIdx(0);
		vector<float> DistIdx(0);
		kdtree.radiusSearch(downSamplepC->points[i], radius, PIdx, DistIdx);
		size_t p_num = PIdx.size();
		P_N& normal_ = normals->points[i];
		float sum_nx = 0.0f, sum_ny = 0.0f, sum_nz = 0.0f;
		for (size_t i = 0; i < p_num; ++i)
		{
			P_N& normal = model_pcn->points[PIdx[i]];
			sum_nx += normal.normal_x; sum_ny += normal.normal_y; sum_nz += normal.normal_z;
		}
		normal_.normal_x = sum_nx / p_num; normal_.normal_y = sum_ny / p_num; normal_.normal_z = sum_nz / p_num;
		float norm = std::sqrt(normal_.normal_x * normal_.normal_x + normal_.normal_y 
			* normal_.normal_y + normal_.normal_z*normal_.normal_z);
		if (norm > EPS)
		{
			normal_.normal_x /= norm; normal_.normal_y /= norm; normal_.normal_z /= norm;
		}
		else
		{
			normal_.normal_x = 0.0f; normal_.normal_y = 0.0f; normal_.normal_z = 0.0f;
		}
	}
}
//==================================================================================

//计算局部转换矩阵==================================================================
void ComputeLocTransMat(P_XYZ& ref_p, P_N& ref_pn, cv::Mat& transMat)
{
	if (!transMat.empty())
		transMat.release();
	transMat = cv::Mat(cv::Size(4, 4), CV_32FC1, cv::Scalar(0));
	float rotAng = std::acosf(ref_pn.normal_x);
	P_N rotAxis(0.0f, ref_pn.normal_z, -ref_pn.normal_y); //旋转轴垂直于x轴与参考点法向量
	if (abs(rotAxis.normal_y) < EPS && abs(ref_pn.normal_z) < EPS)
	{
		rotAxis.normal_y = 1.0f; 
		rotAxis.normal_z = 0.0f;
	}
	else
	{
		float norm = 1.0f / std::sqrt(rotAxis.normal_y * rotAxis.normal_y + rotAxis.normal_z * rotAxis.normal_z);
		rotAxis.normal_y *= norm; rotAxis.normal_z *= norm;
	}
	cv::Mat rotMat(cv::Size(3, 3), CV_32FC1, cv::Scalar(0));
	RodriguesFormula(rotAxis, rotAng, rotMat);
	float* pTransMat = transMat.ptr<float>(0);
	rotMat.copyTo(transMat(cv::Rect(0, 0, 3, 3)));
	pTransMat[3] = -(pTransMat[0] * ref_p.x + pTransMat[1] * ref_p.y + pTransMat[2] * ref_p.z);
	pTransMat[7] = -(pTransMat[4] * ref_p.x + pTransMat[5] * ref_p.y + pTransMat[6] * ref_p.z);
	pTransMat[11] = -(pTransMat[8] * ref_p.x + pTransMat[9] * ref_p.y + pTransMat[10] * ref_p.z);
	pTransMat[15] = 1.0f;
}
//==================================================================================

//计算局部坐标系下的alpha===========================================================
float ComputeLocalAlpha(P_XYZ& ref_p, P_N& ref_pn, P_XYZ& p_, cv::Mat& refTransMat)
{
	float* pTransMat = refTransMat.ptr<float>();
	float y = pTransMat[4] * p_.x + pTransMat[5] * p_.y + pTransMat[6] * p_.z + pTransMat[7];
	float z = pTransMat[8] * p_.x + pTransMat[9] * p_.y + pTransMat[10] * p_.z + pTransMat[11];
	float alpha = std::atan2(-z, y);
	if (sin(alpha) * z < EPS)
	{
		alpha = -alpha;
	}
	return (-alpha);
}
//==================================================================================

//创建PPF模板=======================================================================
void CreatePPFModel(PC_XYZ::Ptr& modelPC, PPFMODEL& ppfModel, float distRatio)
{
	P_XYZ min_p, max_p;
	pcl::getMinMax3D(*modelPC, min_p, max_p);
	float diff_x = max_p.x - min_p.x;
	float diff_y = max_p.y - min_p.y;
	float diff_z = max_p.z - min_p.z;
	ppfModel.distStep = std::sqrtf(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) * distRatio;

	ppfModel.alphStep = (float)CV_2PI / ppfModel.numAng;
	ppfModel.angThres = (float)CV_2PI / ppfModel.alphStep;
	ppfModel.distThres = ppfModel.distStep;
	//ppfModel.numAng = (int)std::floor((float)CV_2PI / ppfModel.alphStep);

	PC_XYZ::Ptr downSampplePC(new PC_XYZ);
	PC_VoxelGrid(modelPC, downSampplePC, ppfModel.distStep);
	PC_N::Ptr normals(new PC_N);
	ExtractPPFNormals(modelPC, downSampplePC, normals, ppfModel.distStep);
	size_t p_number = downSampplePC->points.size();

	ppfModel.refTransMat.resize(p_number);
	for (size_t i = 0; i < p_number; ++i)
	{
		P_XYZ& ref_p = downSampplePC->points[i];
		P_N& ref_pn = normals->points[i];
		ComputeLocTransMat(ref_p, ref_pn, ppfModel.refTransMat[i]);
		for (size_t j = 0; j < p_number; ++j)
		{
			if (i != j)
			{
				P_XYZ& p_ = downSampplePC->points[j];
				P_N& p_n = normals->points[j];
				PPFFEATRUE ppfFEATRUE;
				ComputePPFFEATRUE(ref_p, p_, ref_pn, p_n, ppfFEATRUE);
				float alpha_ = ComputeLocalAlpha(ref_p, ref_pn, p_, ppfModel.refTransMat[i]);
				PushPPFToHashMap(ppfModel.hashMap, ppfFEATRUE, ppfModel.distStep, ppfModel.alphStep, i, alpha_);
			}
		}
	}
}
//==================================================================================

//重置投票器========================================================================
void ResetAccumulator(vector<vector<uint>>& accumulator)
{
	for (size_t i = 0; i < accumulator.size(); ++i)
	{
		for (size_t j = 0; j < accumulator[i].size(); ++j)
		{
			accumulator[i][j] = 0;
		}
	}
}
//==================================================================================

//计算变换矩阵======================================================================
void ComputeTransMat(cv::Mat& SToGMat, float alpha, const cv::Mat& RToGMat, cv::Mat& transMat)
{
	float sinVal = std::sin(alpha);
	float cosVal = std::cos(alpha);
	cv::Mat RAlphaMat(cv::Size(4, 4), CV_32FC1, cv::Scalar(0));
	float* pRAlphaMat = RAlphaMat.ptr<float>();
	pRAlphaMat[5] = cosVal;	pRAlphaMat[6] = -sinVal;
	pRAlphaMat[9] = sinVal;	pRAlphaMat[10] = cosVal;
	pRAlphaMat[0] = 1.0f; pRAlphaMat[15] = 1.0f;

	cv::Mat SToGMatInv(cv::Size(4, 4), CV_32FC1, cv::Scalar(0));
	SToGMatInv = SToGMat.inv();
	//SToGMatInv(cv::Rect(0, 0, 3, 3)) = (SToGMat(cv::Rect(0, 0, 3, 3))).inv();
	//SToGMatInv(cv::Rect(3, 0, 1, 4)) = -SToGMat(cv::Rect(3, 0, 1, 4));
	//SToGMatInv.at<float>(3, 3) = 1.0f;
	transMat = SToGMatInv * RAlphaMat * RToGMat;
}
//==================================================================================

//排序==============================================================================
bool ComparePose(PPFPose& a, PPFPose& b)
{
	return a.votes > b.votes;
}
//==================================================================================

//求旋转矩阵的旋转角================================================================
float ComputeRotMatAng(cv::Mat& transMat)
{
	float* pTransMat = transMat.ptr<float>();
	float trace = pTransMat[0] + pTransMat[5] + pTransMat[10];
	if (fabs(trace - 3) <= EPS)
		return 0.0f;
	else
	{
		if (fabs(trace + 1) <= EPS)
			return 3.1415926f;
		else
			return (acosf((trace - 1) / 2));
	}
}
//==================================================================================

//判定条件==========================================================================
bool DecisionCondition(PPFPose& a, PPFPose& b, float angThres, float distThres)
{
	float angle_a = ComputeRotMatAng(a.transMat);
	float angle_b = ComputeRotMatAng(b.transMat);

	float* pATransMat = a.transMat.ptr<float>();
	float* pBTransMat = a.transMat.ptr<float>();
	float diff_x = pATransMat[3] - pBTransMat[3];
	float diff_y = pATransMat[7] - pBTransMat[7];
	float diff_z = pATransMat[11] - pBTransMat[11];
	
	return (abs(angle_a - angle_b) < angThres && (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) < distThres);
}
//==================================================================================

//非极大值抑制======================================================================
void NonMaxSuppression(vector<PPFPose>& ppfPoses, vector<PPFPose>& resPoses, float angThres, float distThres)
{
	std::sort(ppfPoses.begin(), ppfPoses.end(), ComparePose);
	size_t pose_num = ppfPoses.size();
	vector<bool> isLabel(pose_num, false);
	for (size_t i = 0; i < pose_num; ++i)
	{
		if (isLabel[i])
			continue;
		PPFPose& ref_pose = ppfPoses[i];
		resPoses.push_back(ref_pose);
		isLabel[i] = true;
		for (size_t j = i; j < pose_num; ++j)
		{
			if (!isLabel[j])
			{
				PPFPose& pose_ = ppfPoses[i];
				if (DecisionCondition(ref_pose, pose_, angThres, distThres))
				{
					isLabel[j] = true;
				}
			}
		}
	}
}
//==================================================================================

//查找模板==========================================================================
void MatchPose(PC_XYZ::Ptr& srcPC, PPFMODEL& ppfModel, vector<PPFPose>& resPoses)
{
	if (resPoses.size() != 0)
		resPoses.resize(0);
	PC_XYZ::Ptr downSampplePC(new PC_XYZ);
	PC_VoxelGrid(srcPC, downSampplePC, ppfModel.distStep);
	PC_N::Ptr normals(new PC_N);
	ExtractPPFNormals(srcPC, downSampplePC, normals, ppfModel.distStep);

	uint numAngles = ppfModel.numAng;
	size_t ref_p_num = ppfModel.refTransMat.size();
	size_t p_number = downSampplePC->points.size();

	vector<vector<uint>> accumulator(ref_p_num, vector<uint>(numAngles));
	vector<PPFPose> v_ppfPose(0);
	for (size_t i = 0; i < p_number; ++i)
	{
		ResetAccumulator(accumulator);
		P_XYZ& ref_p = downSampplePC->points[i];
		P_N& ref_pn = normals->points[i];
		cv::Mat SToGMat;
		ComputeLocTransMat(ref_p, ref_pn, SToGMat);
		for (size_t j = 0; j < p_number; ++j)
		{
			if (i != j)
			{
				P_XYZ& p_ = downSampplePC->points[j];
				P_N& p_n = normals->points[j];
				PPFFEATRUE ppfFEATRUE;
				ComputePPFFEATRUE(ref_p, p_, ref_pn, p_n, ppfFEATRUE);

				int dist = ppfFEATRUE.dist / ppfModel.distStep;
				int ang_N1D = ppfFEATRUE.ang_N1D / ppfModel.alphStep;
				int ang_N2D = ppfFEATRUE.ang_N2D / ppfModel.alphStep;
				int ang_N1N2 = ppfFEATRUE.ang_N1N2 / ppfModel.alphStep;

				string hashKey = std::to_string(dist) + std::to_string(ang_N1D) + std::to_string(ang_N2D) + std::to_string(ang_N1N2);
				vector<PPFCELL>& ppfCell_v = ppfModel.hashMap[hashKey];
				float alpha_ = ComputeLocalAlpha(ref_p, ref_pn, p_, SToGMat);
				for (size_t k = 0; k < ppfCell_v.size(); ++k)
				{
					//这样处理的原因防止出现负的索引，只是一个简单的转换而已没啥
					int alpha_index = (int)(numAngles * (ppfCell_v[k].ref_alpha - alpha_ + CV_2PI) / (4 * M_PI));
					accumulator[ppfCell_v[k].ref_i][alpha_index]++;
				}
			}
		}
		PPFPose pose;
		pose.votes = accumulator[0][0];
		for (size_t j = 0; j < ref_p_num; ++j)
		{
			for (size_t k = 0; k < numAngles; ++k)
			{
				if (pose.votes < accumulator[j][k])
				{
					pose.votes = accumulator[j][k];
					pose.ref_i = j; pose.i_ = k;
				}
			}
		}
		float alpha = (pose.i_*(4 * M_PI)) / numAngles - CV_2PI;
		ComputeTransMat(SToGMat, alpha, ppfModel.refTransMat[pose.ref_i], pose.transMat);
		if (pose.votes != 0)
			v_ppfPose.push_back(pose);
	}
	NonMaxSuppression(v_ppfPose, resPoses, ppfModel.angThres, ppfModel.distThres);
}
//==================================================================================

//点云转换==========================================================================
void TransPointCloud(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& dstPC, const cv::Mat& transMat)
{
	size_t length = srcPC->points.size();
	dstPC->points.resize(length);
	const float* pTransMat = transMat.ptr<float>();
	for (size_t i = 0; i < length; ++i)
	{
		P_XYZ& src_p = srcPC->points[i];
		P_XYZ& dst_p = dstPC->points[i];
		dst_p.x = src_p.x * pTransMat[0] + src_p.y * pTransMat[1]
			+ src_p.z * pTransMat[2] + pTransMat[3];
		dst_p.y = src_p.x * pTransMat[4] + src_p.y * pTransMat[5]
			+ src_p.z * pTransMat[6] + pTransMat[7];
		dst_p.z = src_p.x * pTransMat[8] + src_p.y * pTransMat[9]
			+ src_p.z * pTransMat[10] + pTransMat[11];
	}
}
//==================================================================================

//测试程序==========================================================================
void TestProgram()
{
	PC_XYZ::Ptr modelPC(new PC_XYZ());
	string path = "H:/Point-Cloud-Processing-example-master/第十一章/6 template_alignment/source/data/object_template_1.pcd";
	pcl::io::loadPCDFile(path,*modelPC);
	
	PPFMODEL ppfModel;
	float distRatio = 0.1;
	ppfModel.numAng = 30;
	CreatePPFModel(modelPC, ppfModel, distRatio);

	PC_XYZ::Ptr testPC(new PC_XYZ());
	string path1 = "H:/Point-Cloud-Processing-example-master/第十一章/6 template_alignment/source/data/object_template_0.pcd";
	pcl::io::loadPCDFile(path1, *testPC);
	vector<PPFPose> resPoses;
	MatchPose(testPC, ppfModel, resPoses);

	PC_XYZ::Ptr dstPC(new PC_XYZ);
	cv::Mat transMat = resPoses[0].transMat;
	TransPointCloud(modelPC, dstPC, transMat);
	
	MatchPose(dstPC, ppfModel, resPoses);
	PC_XYZ::Ptr dstPC_(new PC_XYZ);
	cv::Mat transMatInv = resPoses[0].transMat;
	cv::Mat I = transMatInv * transMat;
	TransPointCloud(dstPC, dstPC_, transMatInv);

	pcl::visualization::PCLVisualizer viewer;
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> white(modelPC, 255, 255, 255);
	viewer.addPointCloud(modelPC, white, "modelPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "modelPC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(dstPC, 255, 0, 0);
	viewer.addPointCloud(dstPC, red, "testPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "testPC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(dstPC_, 0, 255, 0);
	viewer.addPointCloud(dstPC_, green, "dstPC_");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dstPC_");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
//==================================================================================