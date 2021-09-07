#include "OpenCV_Utils.h"

//计算图像的灰度直方图==========================================================
void Img_ComputeImgHist(Mat& srcImg, Mat& hist)
{
	int row = srcImg.rows;
	int col = srcImg.cols;
	int channel = srcImg.channels();
	if (channel == 1)
		hist = Mat(cv::Size(1, 256), CV_64FC1, cv::Scalar(0));
	else if (channel == 3)
		hist = Mat(cv::Size(1, 256), CV_64FC3, cv::Scalar(0));

	uchar* pSrc = srcImg.ptr<uchar>(0);
	double* pHist = hist.ptr<double>(0);
	int step = channel * col;
	for (int y = 0; y < row; ++y)
	{
		int offset = y * step;
		for (int x = 0; x < step; x += channel)
		{
			for (int c_ = 0; c_ < channel; ++c_)
			{
				pHist[256 * c_ + (int)pSrc[offset + x + c_]] += 1;
			}
		}
	}
	hist /= ((double)(row * col));
}
//==============================================================================