#include "ImageSeg.h"
#include "OpenCV_Utils.h"

//整体阈值分割=============================================================================
NB_API void Img_Seg(Mat& srcImg, Mat& dstImg, double thresVal, IMG_SEG mode)
{
	switch (mode)
	{
	case IMG_SEG_LIGHT:
		cv::threshold(srcImg, dstImg, thresVal, 255, THRESH_BINARY);
		break;
	case IMG_SEG_DARK:
		cv::threshold(srcImg, dstImg, thresVal, 255, THRESH_BINARY_INV);
		break;
	default:
		break;
	}
}
//=========================================================================================

//选择灰度区间=============================================================================
NB_API void Img_SelectGraySeg(Mat& srcImg, Mat& dstImg, uchar thresVal1, uchar thresVal2, IMG_SEG mode)
{
	Mat lookUpTable(1, 256, CV_8UC1);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
	{
		switch (mode)
		{
		case IMG_SEG_LIGHT:
			p[i] = (i > thresVal1 && i < thresVal2) ? 255 : 0;
			break;
		case IMG_SEG_DARK:
			p[i] = (i < thresVal1 && i > thresVal2) ? 255 : 0;
			break;
		default:
			break;
		}
	}
	cv::LUT(srcImg, lookUpTable, dstImg);
}
//=========================================================================================

//熵最大的阈值分割(熵越大系统越不稳定)=====================================================
NB_API void Img_MaxEntropySeg(Mat &srcImg, Mat &dstImg, IMG_SEG mode)
{
	Mat hist;
	Img_ComputeImgHist(srcImg, hist);
	int thresVal = 0;
	double entropy = 0;
	double* pHist = hist.ptr<double>();
	for (int index = 0; index < 256; ++index)
	{
		//计算背景熵
		float b_p = 0.0;
		for (int i = 0; i < index; ++i)
		{
			b_p += pHist[i];
		}
		float b_entropy = 0.0;
		for (int i = 0; i < index; ++i)
		{
			float p_i = pHist[i] / b_p;
			b_entropy -= p_i * log(p_i + 1e-8);
		}
		//计算前景熵
		float f_p = 1 - b_p;
		float f_entropy = 0.0;
		for (int i = index; i < 256; ++i)
		{
			float p_i = pHist[i] / f_p;
			f_entropy -= p_i * log(p_i + 1e-8);
		}
		if (entropy < (b_entropy + f_entropy))
		{
			entropy = b_entropy + f_entropy;
			thresVal = index;
		}
	}
	Img_Seg(srcImg, dstImg, thresVal, mode);
}
//=========================================================================================

//迭代自适应二值化=========================================================================
NB_API void Img_IterTresholdSeg(Mat &srcImg, Mat &dstImg, double eps, IMG_SEG mode)
{
	Mat hist;
	Img_ComputeImgHist(srcImg, hist);
	double* pHist = hist.ptr<double>();
	float thresVal = 0.0;
	for (int i = 0; i < 256; ++i)
		thresVal += (i * pHist[i]);
	float m1 = 0.0;
	float m2 = 0.0;
	float T = 0.0;
	while (abs(thresVal - T) < eps)
	{
		for (int i = 0; i < thresVal; ++i)
		{
			m1 += (i * pHist[i]);
		}
		for (int i = (int)thresVal; i < 256; ++i)
		{
			m2 += (i * pHist[i]);
		}
		T = thresVal;
		thresVal = (m1 + m2) / 2;
	}
	Img_Seg(srcImg, dstImg, thresVal, mode);
}
//=========================================================================================

//局部自适应阈值分割=======================================================================
NB_API void Img_LocAdapThresholdSeg(Mat& srcImg, Mat& dstImg, cv::Size size, double stdDevScale, double absThres, IMG_SEG mode)
{
	dstImg = Mat(srcImg.size(), CV_8UC1, cv::Scalar(0));
	Mat srcImg_16S = Mat(srcImg.size(), CV_16UC1, Scalar::all(0));
	srcImg.convertTo(srcImg_16S, srcImg_16S.type());

	//计算图像的均值
	Mat meanImg(srcImg.size(), CV_32FC1, cv::Scalar(0));
	cv::boxFilter(srcImg, meanImg, meanImg.type(), size);

	//计算图像的方差
	Mat II = srcImg_16S.mul(srcImg_16S);
	Mat mean_II(srcImg.size(), CV_32FC1, Scalar::all(0));
	cv::boxFilter(II, mean_II, mean_II.type(), size);
	
	int col = srcImg.cols;
	int row = srcImg.rows;
	uchar* pSrc = srcImg.ptr<uchar>(0);
	float* pMean = meanImg.ptr<float>(0);
	float* pVarImg = mean_II.ptr<float>(0);
	uchar* pDst = dstImg.ptr<uchar>(0);

#pragma omp parallel for
	for (int y = 0; y < row; ++y)
	{
		int offset = y * col;
		for (int x = 0; x < col; ++x)
		{
			int offset_ = offset + x;
			double srcVal = (double)pSrc[offset_];
			double meanVal = (double)pMean[offset_];
			double var = std::sqrt(pVarImg[offset_] - meanVal * meanVal) * stdDevScale;
			var = stdDevScale >= 0 ? std::max(var, absThres) : std::min(var, absThres);
			switch(mode){
			case IMG_SEG_LIGHT:
				pDst[offset_] = srcVal >= meanVal + var ? 255 : 0;
				break;
			case IMG_SEG_DARK:
				pDst[offset_] = srcVal < meanVal - var ? 255 : 0;
				break;
			case IMG_SEG_EQUL:
				pDst[offset_] = ((srcVal >= meanVal - var) && (srcVal <= meanVal + var)) ? 255 : 0;
				break;
			case IMG_SEG_NOTEQUL:
				pDst[offset_] = ((srcVal < meanVal - var) && (srcVal > meanVal + var)) ? 255 : 0;
				break;
			default: break;
			}
		}
	}
}
//=========================================================================================

//迟滞分割=================================================================================
NB_API void Img_HysteresisSeg(Mat& srcImg, Mat& dstImg, double thresVal1, double thresVal2)
{
	int col = srcImg.cols;
	int row = srcImg.rows;
	Mat mask(row, col, CV_8UC1, cv::Scalar(0));
	dstImg = Mat(row, col, CV_8UC1, cv::Scalar(0));
	uchar* pSrc = srcImg.data;
	uchar* pMask = mask.data;
	uchar* pDst = dstImg.data;
	int idxes[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };


	for (int y = 0; y < row; ++y)
	{
		pSrc += col; pMask += col; pDst += col;
		for (int x = 0; x < col; ++x)
		{
			if (pSrc[x] >= thresVal2)
				pDst[x] = 255;
			else if (pSrc[x] > thresVal1)
			{
				for (int i = 0; i < 8; ++i)
				{
					int x_ = std::min(col - 1, std::max(0, x + idxes[i][0]));
					int y_ = std::min(row - 1, std::max(0, y + idxes[i][1]));
					if (dstImg.at<uchar>(y_, x_) > thresVal2)
					{
						pDst[x] = 255;
						break;
					}
				}
			}
		}
	}
}
//=========================================================================================

//Halcon中的点检测=========================================================================
NB_API void Img_DotImgSeg(Mat& srcImg, Mat& dstImg, int size, IMG_SEG mode)
{
	int length = size + 2;
	double center = length / 2;
	double radius1_2 = (size / 2.0) * (size / 2.0);
	double radisu2_2 = (length / 2.0) * (length / 2.0);
	double positive = 0;
	double negtive = 0;
	Mat kenerl = Mat(length, length, CV_32FC1, Scalar(0));
	float *pKenerl = kenerl.ptr<float>(0);
	for (int y = 0; y < length; ++y)
	{
		double y_2 = (y - center) * (y - center);
		for (int x = 0; x < length; ++x)
		{
			double x_2 = (x - center) * (x - center);
			if (x_2 + y_2 < radius1_2)
				positive--;
			else if (x_2 + y_2 < radisu2_2)
				negtive++;
		}
	}
	double sum = 0.0f;
	for (int y = 0; y < length; ++y, pKenerl += length)
	{
		double y_2 = (y - center) * (y - center);
		for (int x = 0; x < length; ++x)
		{
			double x_2 = (x - center) * (x - center);
			if (x_2 + y_2 < radius1_2)
				pKenerl[x] = negtive;
			else if (x_2 + y_2 < radisu2_2)
			{
				pKenerl[x] = positive;
				sum -= positive;
			}
		}
	}
	sum = std::max(sum, EPS);
	switch (mode)
	{
	case IMG_SEG::IMG_SEG_LIGHT:
		kenerl /= sum; break;
	case IMG_SEG::IMG_SEG_DARK:
		kenerl /= -sum; break;
	}
	dstImg = Mat(srcImg.size(), CV_8UC1, Scalar(0));
	cv::filter2D(srcImg, dstImg, CV_8UC1, kenerl);
}
//=========================================================================================

//区域生长=================================================================================
NB_API void Img_RegionGrowSeg(Mat& srcImg, Mat& labels, int dist_c, int dist_r, int thresVal, int minRegionSize)
{
	if (!labels.empty())
		labels.release();
	int half_c = dist_c / 2;
	int half_r = dist_r / 2;

	Mat padded;
	copyMakeBorder(srcImg, padded, half_r, half_r, half_c, half_c, cv::BORDER_REFLECT, Scalar::all(0));
	Mat mask(padded.size(), CV_8UC1, cv::Scalar(0));
	int minArea = dist_c * dist_c;
	uchar* pPadded = padded.ptr<uchar>(0);
	int col = padded.cols - half_c;
	int row = padded.rows - half_r;
	int half_c_1 = half_c + 1;
	int half_r_1 = half_r + 1;
	int step = padded.cols * padded.channels();
	uchar* pMask = mask.ptr<uchar>(0);
	queue<cv::Point> seeds;
	seeds.push(cv::Point(half_c, half_r));
	vector<vector<cv::Point>> regions;
	while (!seeds.empty())
	{
		queue<cv::Point> seed;
		if (mask.at<uchar>(seeds.front()) == 0)
		{
			seed.push(seeds.front());
		}
		seeds.pop();
		vector<cv::Point> region;
		while (!seed.empty())
		{
			int c_x = seed.front().x;
			int c_y = seed.front().y;
			seed.pop();
			int ref_val = (int)(pPadded[c_y * step + c_x]);
			for (int y = c_y - dist_r; y <= c_y + dist_r; y += dist_r)
			{
				if (y >= half_r && y <= row)
				{
					int offset_y = y * step;
					for (int x = c_x - dist_c; x <= c_x + dist_c; x += dist_c)
					{
						if (x >= half_c && x <= col)
						{
							int offset = offset_y + x;
							int val = abs((int)(pPadded[offset]) - ref_val);
							if (val < thresVal && pMask[offset] == 0)
							{
								cv::Point pt_(x, y);
								pMask[offset] = 255;
								region.push_back(pt_);
								seed.push(pt_);
							}
							else
							{
								seeds.push(cv::Point(x, y));
							}
						}
					}
				}
			}
		}
		if (region.size() * minArea > minRegionSize)
			regions.push_back(region);
	}
	Mat region(padded.size(), CV_8UC3, cv::Scalar(0));
	for (int i = 0; i < regions.size(); ++i)
	{
		for (int j = 0; j < regions[i].size(); ++j)
		{
			int end_y = std::min(regions[i][j].y + half_r + 1, padded.rows);
			int end_x = std::min(regions[i][j].x + half_c + 1, padded.cols);
			cv::Scalar color(0,0,0);
			color(i % 3) = 255;
			region(Range(regions[i][j].y - half_r, end_y), Range(regions[i][j].x - half_c, end_x)) = color;
		}
	}
}
//=========================================================================================

//各向异性的图像分割=======================================================================
void NB_AnisImgSeg(Mat &srcImg, Mat &dstImg, int WS, double C_Thr, int lowThr, int highThr)
{
	Mat grad_x, grad_y, grad_xy;
	Sobel(srcImg, grad_x, CV_32F, 1, 0, 3);
	Sobel(srcImg, grad_y, CV_32F, 0, 1, 3);
	grad_xy = grad_x.mul(grad_y);

	Mat grad_xx = grad_x.mul(grad_x);
	Mat grad_yy = grad_y.mul(grad_y);

	Mat J11, J22, J12;
	boxFilter(grad_xx, J11, CV_32F, Size(WS, WS));
	boxFilter(grad_yy, J22, CV_32F, Size(WS, WS));
	boxFilter(grad_xy, J12, CV_32F, Size(WS, WS));

	Mat tmp1 = J11 + J22;
	Mat tmp2 = J11 - J22;
	tmp2 = tmp2.mul(tmp2);
	Mat tmp3 = J12.mul(J12);
	Mat tmp4;
	sqrt(tmp2 + 4.0 * tmp3, tmp4);

	Mat lambda1 = tmp1 + tmp4;
	Mat lambda2 = tmp1 - tmp4;

	Mat imgCoherency, imgOrientation;
	divide(lambda1 - lambda2, lambda1 + lambda2, imgCoherency);
	phase(J22 - J11, 2.0*J12, imgOrientation, true);
	imgOrientation = 0.5*imgOrientation;


	Mat imgCoherencyBin;
	imgCoherencyBin = imgCoherency > C_Thr;
	Mat imgOrientationBin;
	inRange(imgOrientation, Scalar(lowThr), Scalar(highThr), imgOrientationBin);
	Mat imgBin;
	imgBin = imgCoherencyBin & imgOrientationBin;
	normalize(imgCoherency, imgCoherency, 0, 255, NORM_MINMAX);
	normalize(imgOrientation, imgOrientation, 0, 255, NORM_MINMAX);
}
//=========================================================================================


void ImgSegTest()
{
	string imgPath = "C:/Users/Administrator/Desktop/testimage/05.jpg";
	Mat srcImg = imread(imgPath, 0);

	Mat dstImg;
	Img_RegionGrowSeg(srcImg, dstImg, 3, 3, 5, 1000);

	Mat t = dstImg.clone();
}