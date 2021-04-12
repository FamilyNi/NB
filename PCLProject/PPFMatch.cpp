#include "PPFMatch.h"
#include "PC_Filter.h"
#include "PC_UTILS.h"

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
	float normal_ = std::max(ppfFEATRUE.dist, EPS);
	p_v.x /= normal_; p_v.y /= normal_; p_v.z /= normal_;

	ppfFEATRUE.ang_N1N2 = ref_pn.normal_x * p_n.normal_x + ref_pn.normal_y * p_n.normal_y + ref_pn.normal_z * p_n.normal_z;
	ppfFEATRUE.ang_N1D = ref_pn.normal_x * p_v.x + ref_pn.normal_y * p_v.y + ref_pn.normal_z * p_v.z;
	ppfFEATRUE.ang_N2D = p_n.normal_x * p_v.x + p_n.normal_y * p_v.y + p_n.normal_z * p_v.z;
}
//==================================================================================

//将点对以PPF特征推送到hash表中=====================================================
void PushPPFToHashMap(hash_map<string, vector<PPFCELL>>& hashMap, PPFFEATRUE& ppfFEATRUE, int ref_i, float alpha)
{
	string hashKey = std::to_string(ppfFEATRUE.dist) + std::to_string(ppfFEATRUE.ang_N1D)
		+ std::to_string(ppfFEATRUE.ang_N2D) + std::to_string(ppfFEATRUE.ang_N1N2);
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
	ComputePCNormal(srcPC, model_pcn, radius);
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

//算转换矩阵========================================================================
void ComputeLocTransMat(P_XYZ& ref_p, P_N& ref_pn, cv::Mat& transMat)
{
	if (!transMat.empty())
		transMat.release();
	transMat = cv::Mat(cv::Size(4, 4), CV_32FC1, cv::Scalar(0));
	float rotAng = std::acos(ref_pn.normal_x);
	P_N rotAxis(0.0f, ref_pn.normal_z, -ref_pn.normal_y); //旋转轴垂直于x轴与参考点法向量
	if (rotAxis.normal_y == 0 && ref_pn.normal_z == 0)
	{
		rotAxis.normal_y = 1.0f; rotAxis.normal_z = 0.0f;
	}
	else
	{
		float norm = 1.0 / std::sqrt(rotAxis.normal_y * rotAxis.normal_y + rotAxis.normal_z * rotAxis.normal_z);
		rotAxis.normal_y *= norm; rotAxis.normal_z *= norm;
	}
	cv::Mat rotMat(cv::Size(3, 3), CV_32FC1, cv::Scalar(0));
	RodriguesFormula(rotAxis, rotAng, rotMat);
	float* pRotMat = rotMat.ptr<float>();
	float* pTransMat = transMat.ptr<float>();
	pTransMat[0] = pRotMat[0]; pTransMat[1] = pRotMat[1]; pTransMat[2] = pRotMat[2];
	pTransMat[0] = -(pRotMat[0] * ref_p.x + pRotMat[1] * ref_p.y + pRotMat[2] * ref_p.z);

	pTransMat[4] = pRotMat[3]; pTransMat[5] = pRotMat[4]; pTransMat[6] = pRotMat[5];
	pTransMat[7] = -(pRotMat[3] * ref_p.x + pRotMat[4] * ref_p.y + pRotMat[5] * ref_p.z);

	pTransMat[8] = pRotMat[6]; pTransMat[9] = pRotMat[7]; pTransMat[10] = pRotMat[8];
	pTransMat[11] = -(pRotMat[6] * ref_p.x + pRotMat[7] * ref_p.y + pRotMat[8] * ref_p.z);
	pTransMat[15] = 1;
}
//==================================================================================

//计算局部坐标系下的alpha===========================================================
float ComputeLocalAlpha(P_XYZ& ref_p, P_N& ref_pn, P_XYZ& p_, cv::Mat& refTransMat)
{
	float* pTransMat = refTransMat.ptr<float>();
	float y = pTransMat[4] * p_.x + pTransMat[5] * p_.y + pTransMat[6] * p_.x + pTransMat[7];
	float z = pTransMat[8] * p_.x + pTransMat[9] * p_.y + pTransMat[10] * p_.x + pTransMat[11];
	float alpha = std::atan2(-z, y);
	if (sin(alpha) * y < 0.0)
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
	float dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
	ppfModel.distStep = distRatio * dist;

	PC_VoxelGrid(modelPC, ppfModel.modelPC, ppfModel.distStep);
	PC_N::Ptr normals(new PC_N);
	ExtractPPFNormals(modelPC, ppfModel.modelPC, normals, ppfModel.distStep);

	size_t p_number = ppfModel.modelPC->points.size();
	ppfModel.refTransMat.resize(p_number);
	for (size_t i = 0; i < p_number; ++i)
	{
		P_XYZ& ref_p = ppfModel.modelPC->points[i];
		P_N& ref_pn = normals->points[i];
		ComputeLocTransMat(ref_p, ref_pn, ppfModel.refTransMat[i]);
		for (size_t j = 0; j < p_number; ++j)
		{
			if (i != j)
			{
				P_XYZ& p_ = ppfModel.modelPC->points[j];
				P_N& p_n = normals->points[j];
				PPFFEATRUE ppfFEATRUE;
				ComputePPFFEATRUE(ref_p, p_, ref_pn, p_n, ppfFEATRUE);
				PushPPFToHashMap(ppfModel.hashMap, ppfFEATRUE, i, j);
			}
		}
	}
}
//==================================================================================

//计算变换矩阵======================================================================
void ComputeTransMat(cv::Mat& SToGMat, float alpha, cv::Mat& RToGMat, cv::Mat& transMat)
{
	float sinVal = std::sin(alpha);
	float cosVal = std::cos(alpha);
	cv::Mat RAlphaMat(cv::Size(4, 4), CV_32FC1, cv::Scalar(1));
	float* pRAlphaMat = RAlphaMat.ptr<float>();
	pRAlphaMat[3] = 0;
	pRAlphaMat[5] = cosVal;	pRAlphaMat[6] = -sinVal; pRAlphaMat[7] = 0;
	pRAlphaMat[9] = sinVal;	pRAlphaMat[10] = cosVal; pRAlphaMat[11] = 0;
	pRAlphaMat[12] = 0;	pRAlphaMat[13] = 0; pRAlphaMat[14] = 0;
	transMat = (SToGMat.inv()) * RAlphaMat * RToGMat;
}
//==================================================================================

//查找模板==========================================================================
void MatchPose(PC_XYZ::Ptr& srcPC, PPFMODEL& ppfModel)
{
	PC_XYZ::Ptr downSampplePC(new PC_XYZ);
	PC_VoxelGrid(srcPC, downSampplePC, ppfModel.distStep);
	PC_N::Ptr normals(new PC_N);
	ExtractPPFNormals(srcPC, downSampplePC, normals, ppfModel.distStep);

	size_t numAngles = (int)(floor(2 * M_PI / ppfModel.alphStep));
	size_t ref_p_num = ppfModel.modelPC->points.size();
	size_t p_number = downSampplePC->points.size();
	vector<PPFPose> v_ppfPose(p_number);
	for (size_t i = 0; i < p_number; ++i)
	{
		P_XYZ& ref_p = ppfModel.modelPC->points[i];
		P_N& ref_pn = normals->points[i];
		vector<vector<uint>> accumulator(ref_p_num, vector<uint>(numAngles));
		cv::Mat SToGMat(cv::Size(4, 4), CV_32FC1, cv::Scalar(0));
		ComputeLocTransMat(ref_p,ref_pn, SToGMat);
		for (size_t j = 0; j < p_number; ++j)
		{
			if (i != j)
			{
				P_XYZ& p_ = ppfModel.modelPC->points[j];
				P_N& p_n = normals->points[j];
				PPFFEATRUE ppfFEATRUE;
				ComputePPFFEATRUE(ref_p, p_, ref_pn, p_n, ppfFEATRUE);
				string hashKey = std::to_string(ppfFEATRUE.dist) + std::to_string(ppfFEATRUE.ang_N1D)
					+ std::to_string(ppfFEATRUE.ang_N2D) + std::to_string(ppfFEATRUE.ang_N1N2);
				vector<PPFCELL>& ppfCell_v = ppfModel.hashMap[hashKey];
				float alpha_ = ComputeLocalAlpha(ref_p, ref_pn, p_, SToGMat);
				for (size_t k = 0; k < ppfCell_v.size(); ++k)
				{
					float alpha_m = 0;
					int alpha_index = (int)(numAngles * (alpha_ - alpha_m + 2 * M_PI) / (4 * M_PI));
					accumulator[ppfCell_v[i].ref_i][alpha_index]++;
				}
			}
		}
		PPFPose pose;
		pose.votes = accumulator[0][0];
		for (size_t j = 0; j < p_number; ++j)
		{
			for (size_t k = 0; k < numAngles; ++i)
			{
				if (pose.votes < accumulator[j][k])
				{
					pose.votes = accumulator[j][k];
					pose.ref_i = j; pose.i_ = k;
				}
			}
		}
		float alpha = (pose.i_*(4 * M_PI)) / numAngles - 2 * M_PI;
		ComputeTransMat(SToGMat, alpha, ppfModel.refTransMat[pose.ref_i], pose.transMat);
		v_ppfPose.push_back(pose);
	}
}
//==================================================================================