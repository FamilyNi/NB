#include "SiftMatch.h"

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
void ExtractSiftPt(Mat& srcIMg, Mat& targetImg, vector<KeyPoint>& srcPts, vector<KeyPoint>& targetPts)
{
	//Ptr<SIFT> detector = SIFT::create();

}
//===============================================================