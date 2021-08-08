#pragma once
#include "utils.h"
#include "opencv2/features2d.hpp"

//图像转灰度图
void ImageToGray(Mat& srcImg, Mat& grayImg);

//提取SIFT角点
void ExtractSiftPt(Mat& srcImg, Mat& tarImg, vector<KeyPoint>& srcPts, 
	vector<KeyPoint>& tarPts, Mat& desSrc, Mat& desTar, int pts_num);

//特征点配对--通过DMatch
void KeyPtsMatch(Mat& desSrc, Mat& desTar, vector<DMatch>& good_matches);

//特征点配对--通过Point2f
void KeyPtsMatch_Pts(vector<KeyPoint>& srcKeyPts,vector<KeyPoint>& tarKeyPts, 
	Mat& desSrc, Mat& desTar, vector<Point2f>& srcPts, vector<Point2f>& tarPts);

//RANSAC算法选择可靠的匹配点
void SelectGoodMatchPts(vector<KeyPoint>& srcPts, vector<KeyPoint>& tarPts, vector<DMatch>& good_matches,
	vector<Point2f>& obj, vector<Point2f>& scene, Mat& tranMat, vector<uchar>& inliers);

//提取内点--通过DMatch
void ExtractInlinerPts(vector<DMatch>& good_matches, vector<uchar>& inliers, vector<DMatch>& inlinerPts);

//提取内点--通过Point2f
void ExtractInlinerPts_Pts(vector<Point2f>& srcPts, vector<Point2f>& tarPts, vector<uchar>& inliers, 
	vector<Point2f>& srcPts_o, vector<Point2f>& tarPts_o);

//最小二乘法优化:AX = B
void SiftLSOpt(vector<Point2f>& srcPts, vector<Point2f>& tarPts, Mat& tranMat);

//提取SIFT角点
void SiftPtTest();
