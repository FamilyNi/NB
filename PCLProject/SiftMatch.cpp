#include "SiftMatch.h"

//ͼ��ת�Ҷ�ͼ===================================================
void ImageToGray(Mat& srcImg, Mat& grayImg)
{
	if (srcImg.channels() > 1)
		cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
	else
		grayImg = srcImg.clone();
}
//===============================================================

//��ȡSIFT�ǵ�===================================================
void ExtractSiftPt(Mat& srcIMg, Mat& targetImg, vector<KeyPoint>& srcPts, vector<KeyPoint>& targetPts)
{
	//Ptr<SIFT> detector = SIFT::create();

}
//===============================================================