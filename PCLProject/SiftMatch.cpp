#include "SiftMatch.h"
#include<opencv2/opencv.hpp>

//图像转灰度图===================================================
void ImageToGray(Mat& srcImg, Mat& grayImg)
{
	if (srcImg.channels() > 1)
		cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
	else
		grayImg = srcImg.clone();
}
//===============================================================

//提取SIFT角点===================================================
void ExtractSiftPt(Mat& srcImg, Mat& tarImg, vector<KeyPoint>& srcPts,
	vector<KeyPoint>& tarPts, Mat& desSrc, Mat& desTar, int pts_num)
{
	Ptr<SIFT> detector = SIFT::create(pts_num);
	detector->detectAndCompute(srcImg, noArray(), srcPts, desSrc);
	detector->detectAndCompute(tarImg, noArray(), tarPts, desTar);
}
//===============================================================

//特征点配对=====================================================
void KeyPtsMatch(Mat& desSrc, Mat& desTar, vector<DMatch>& good_matches)
{
	if (good_matches.size() != 0)
		good_matches.resize(0);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector<vector<DMatch> > knn_matches(0);
	matcher->knnMatch(desSrc, desTar, knn_matches, 2);
	int pts_num = knn_matches.size();
	if (pts_num < 10)
		return;
	good_matches.reserve(pts_num);
	const float ratio_thresh = 0.75f;
	for (int i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}
}
//===============================================================

//特征点配对=====================================================
void KeyPtsMatch_Pts(vector<KeyPoint>& srcKeyPts, vector<KeyPoint>& tarKeyPts,
	Mat& desSrc, Mat& desTar, vector<Point2f>& srcPts, vector<Point2f>& tarPts)
{
	if (srcPts.size() != 0)
		srcPts.resize(0);
	if (tarPts.size() != 0)
		tarPts.resize(0);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector<vector<DMatch> > knn_matches(0);
	matcher->knnMatch(desSrc, desTar, knn_matches, 2);
	int pts_num = knn_matches.size();
	if (pts_num < 10)
		return;
	srcPts.reserve(pts_num);
	tarPts.reserve(pts_num);
	const float ratio_thresh = 0.75f;
	for (int i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			srcPts.push_back(srcKeyPts[knn_matches[i][0].queryIdx].pt);
			tarPts.push_back(tarKeyPts[knn_matches[i][0].trainIdx].pt);
		}
	}
}
//===============================================================

//选择可靠的匹配点===============================================
void SelectGoodMatchPts(vector<KeyPoint>& srcPts, vector<KeyPoint>& tarPts, vector<DMatch>& good_matches,
	vector<Point2f>& obj, vector<Point2f>& scene, Mat& tranMat, vector<uchar>& inliers)
{
	if (good_matches.size() < 10)
		return;
	int pts_num = good_matches.size();
	if (obj.size() != pts_num)
		obj.resize(pts_num);
	if (scene.size() != pts_num)
		scene.resize(pts_num);
	for (int i = 0; i < pts_num; i++)
	{
		obj[i] = srcPts[good_matches[i].queryIdx].pt;
		scene[i] = tarPts[good_matches[i].trainIdx].pt;
	}
	tranMat = findHomography(obj, scene, inliers, RANSAC);
}
//===============================================================

//提取内点=======================================================
void ExtractInlinerPts(vector<DMatch>& good_matches, vector<uchar>& inliers, vector<DMatch>& inlinerPts)
{
	if (good_matches.size() < 10 || inliers.size() < 10)
		return;
	int inlinerPts_num = inliers.size();
	if (inlinerPts.size() != inlinerPts_num)
		inlinerPts.reserve(inlinerPts_num);
	for (int i = 0; i < inlinerPts_num; i++)
	{
		if (inliers[i])
		{
			inlinerPts.push_back(good_matches[i]);
		}
	}
}
//===============================================================

//提取内点=======================================================
void ExtractInlinerPts_Pts(vector<Point2f>& srcPts, vector<Point2f>& tarPts, vector<uchar>& inliers,
	vector<Point2f>& srcPts_o, vector<Point2f>& tarPts_o)
{
	if (srcPts.size() < 10 || tarPts.size() < 10 || inliers.size() < 10)
		return;
	int pts_num = srcPts.size();
	if (tarPts.size() != pts_num || inliers.size() != pts_num)
		return;
	srcPts_o.reserve(pts_num);
	tarPts_o.reserve(pts_num);
	for (int i = 0; i < pts_num; ++i)
	{
		if (inliers[i])
		{
			srcPts_o.push_back(srcPts[i]);
			tarPts_o.push_back(tarPts[i]);
		}
	}
}
//===============================================================

//最小二乘法优化=================================================
void SiftLSOpt(vector<Point2f>& srcPts, vector<Point2f>& tarPts, Mat& tranMat)
{
	int num_pts = srcPts.size();
	if (tarPts.size() != num_pts || num_pts == 0)
		return;
	double sum_sx = 0.0, sum_sy = 0.0, sum_tx = 0.0, sum_ty = 0.0f;
	for (int i = 0; i < num_pts; ++i)
	{
		sum_sx += srcPts[i].x; sum_sy += srcPts[i].y;
		sum_tx += tarPts[i].x; sum_ty += tarPts[i].y;
	}
	Point2d mean_sp(sum_sx / num_pts, sum_sy / num_pts);
	Point2d mean_tp(sum_tx / num_pts, sum_ty / num_pts);
	Mat A(cv::Size(2, 2), CV_64FC1, cv::Scalar(0));
	Mat B(cv::Size(2, 2), CV_64FC1, cv::Scalar(0));
	double* pA = A.ptr<double>();
	double* pB = B.ptr<double>();
	for (int i = 0; i < num_pts; ++i)
	{
		double x_s = srcPts[i].x - mean_sp.x;
		double y_s = srcPts[i].y - mean_sp.y;
		double x_t = tarPts[i].x - mean_tp.x;
		double y_t = tarPts[i].y - mean_tp.y;
		pA[0] += x_s * x_s;	pA[1] += x_s * y_s;	pA[3] += y_s * y_s;

		pB[0] += x_s * x_t;	pB[1] += y_s * x_t;  pB[2] += x_s * y_t; pB[3] += y_s * y_t;
	}
	pA[2] = pA[1]; 
	tranMat = Mat(cv::Size(3, 2), CV_64FC1, cv::Scalar(0));
	tranMat(cv::Rect(0,0,2,2)) = B * (A.inv());
	double* pTransMat = tranMat.ptr<double>();
	pTransMat[2] = mean_tp.x - (pTransMat[0] * mean_sp.x + pTransMat[1] * mean_sp.y);
	pTransMat[5] = mean_tp.y - (pTransMat[3] * mean_sp.x + pTransMat[4] * mean_sp.y);
}
//===============================================================

//提取SIFT角点===================================================
void SiftPtTest()
{
	Mat srcImg = imread("1.jpg", IMREAD_GRAYSCALE);
	Mat tarImg = imread("1_t.png", IMREAD_GRAYSCALE);

	//提取特征点
	vector<KeyPoint> srcPts, tarPts;
	Mat desSrc, desTar;
	int pts_num = 500;
	ExtractSiftPt(srcImg, tarImg, srcPts, tarPts, desSrc, desTar, pts_num);

	//特征点配对
	vector<Point2f> good_srcPts(0), good_tarPts(0);
	KeyPtsMatch_Pts(srcPts, tarPts, desSrc, desTar, good_srcPts, good_tarPts);

	//RANSAC算法选取特征点
	vector<uchar> inliers;
	Mat tranMat = findHomography(good_srcPts, good_tarPts, inliers, RANSAC);

	//选取局内点
	vector<Point2f> inliner_srcPts, inliner_tarPts;
	ExtractInlinerPts_Pts(good_srcPts, good_tarPts, inliers, inliner_srcPts, inliner_tarPts);

	Mat tranMat_;
	SiftLSOpt(inliner_srcPts, inliner_tarPts, tranMat_);
	Mat t_Mat;
	warpAffine(srcImg, t_Mat, tranMat_, srcImg.size());
}
//===============================================================