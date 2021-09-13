#include "OpenCV_Utils.h"

//计算图像的灰度直方图==========================================================
void Img_ComputeImgHist(Mat& srcImg, Mat& hist)
{
	int row = srcImg.rows;
	int col = srcImg.cols;
	int channel = srcImg.channels();
	if (channel == 1)
		hist = Mat(cv::Size(256, 1), CV_64FC1, cv::Scalar(0));
	else if (channel == 3)
		hist = Mat(cv::Size(256, 3), CV_64FC3, cv::Scalar(0));

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

//绘制灰度直方图================================================================
void Img_DrawHistImg(Mat& hist)
{
	Mat normHist;
	cv::normalize(hist, normHist, 0, 1, cv::NORM_MINMAX);
	double* pNormHist = normHist.ptr<double>(0);
	Mat histImage(cv::Size(256, 256), CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < 256; i++)
	{
		cv::Point s_p(i, 255);
		cv::Point e_p(i, std::round(255 - 255 * pNormHist[i]));
		line(histImage, s_p, e_p, Scalar(0, 0, 255));
	}
	imshow("直方图", histImage);
	waitKey(0);
}
//==============================================================================