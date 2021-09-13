#include "ImageSeg.h"
#include "OpenCV_Utils.h"

//������ֵ�ָ�=============================================================================
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

//ѡ��Ҷ�����=============================================================================
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

//��������ֵ�ָ�(��Խ��ϵͳԽ���ȶ�)=====================================================
NB_API void Img_MaxEntropySeg(Mat &srcImg, Mat &dstImg, IMG_SEG mode)
{
	Mat hist;
	Img_ComputeImgHist(srcImg, hist);
	int thresVal = 0;
	double entropy = 0;
	double* pHist = hist.ptr<double>();
	for (int index = 0; index < 256; ++index)
	{
		//���㱳����
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
		//����ǰ����
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

//��������Ӧ��ֵ��=========================================================================
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

//�ֲ�����Ӧ��ֵ�ָ�=======================================================================
NB_API void Img_LocAdapThresholdSeg(Mat& srcImg, Mat& dstImg, cv::Size size, double stdDevScale, double absThres, IMG_SEG mode)
{
	dstImg = Mat(srcImg.size(), CV_8UC1, cv::Scalar(0));
	Mat srcImg_16S = Mat(srcImg.size(), CV_16UC1, Scalar::all(0));
	srcImg.convertTo(srcImg_16S, srcImg_16S.type());

	//����ͼ��ľ�ֵ
	Mat meanImg(srcImg.size(), CV_32FC1, cv::Scalar(0));
	cv::boxFilter(srcImg, meanImg, meanImg.type(), size);

	//����ͼ��ķ���
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

//����ֱ��ͼ�Ķ���ֵ�ָ�===================================================================
NB_API void Img_HistMultiplySeg(Mat& srcImg, Mat& dstImg, double sigma)
{
	Mat hist;
	Img_ComputeImgHist(srcImg, hist);
	cv::GaussianBlur(hist, hist, cv::Size(3, 1), sigma);

}
//=========================================================================================

//���ͷָ�=================================================================================
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

//Halcon�еĵ���=========================================================================
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

//��������=================================================================================
NB_API void Img_RegionGrowSeg(Mat& srcImg, Mat& labels, int dist_c, int dist_r, int thresVal)
{
	if (!labels.empty())
		labels.release();
	labels = Mat(srcImg.size(), CV_32SC1, cv::Scalar(0));
	Mat mask(srcImg.size(), CV_8UC1, cv::Scalar(0));
	int col = srcImg.cols;
	int row = srcImg.rows;
	int label = 0;
	int half_c = dist_c / 2;
	int half_r = dist_r / 2;

	queue<cv::Point> seeds;
	seeds.push(cv::Point(half_c, half_r));
	while (!seeds.empty())
	{
		queue<cv::Point> seed;
		if (mask.at<uchar>(seeds.front()) == 0)
		{
			++label;
			cv::Point& s_pt = seeds.front();
			seed.push(s_pt);
		}
		seeds.pop();
		while (!seed.empty())
		{
			int c_x = seed.front().x;
			int c_y = seed.front().y;
			seed.pop();
			int ref_val = (int)(srcImg.at<uchar>(c_y, c_x));
			for (int y_ = c_y - dist_r; y_ <= c_y + dist_r; y_ += dist_r)
			{
				int y = std::min(row - 1, std::max(0, y_));
				for (int x_ = c_x - dist_c; x_ <= c_x + dist_c; x_ += dist_c)
				{
					int x = std::min(col - 1, std::max(0, x_));
					int s_x = std::max(0, x - half_c);
					int e_x = std::min(col - 1, x + half_c + 1);
					int s_y = std::max(0, y - half_r);
					int e_y = std::min(row - 1, y + half_r + 1);
					//�жϸ������Ƿ�������
					if (cv::countNonZero(labels(Range(s_y, e_y), Range(s_x, e_x))) < (e_x - s_x) * (e_y - s_y))
					{
						uchar& isSeed = mask.at<uchar>(y, x);
						int val = (int)(srcImg.at<uchar>(y, x));
						if (abs(ref_val - val) < thresVal)
						{
							labels(Range(s_y, e_y), Range(s_x, e_x)) = label;
							if (isSeed == 0)
							{
								seed.push(cv::Point(x, y));
								isSeed = 255;
							}
						}
					}
				}
			}
		}
		cv::Point min_p(0,0);
		cv::minMaxLoc(labels, NULL, NULL, &min_p, NULL);
		if (min_p.x > 0 || min_p.y > 0)
		{
			seeds.push(min_p);
		}
		else
			break;
		//cv::minMaxLoc
	}
}
//=========================================================================================

//�������Ե�ͼ��ָ�=======================================================================
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
	string imgPath = "C:/Users/Administrator/Desktop/2.jpg";
	Mat srcImg = imread(imgPath, 0);

	Mat dstImg;
	Img_HysteresisSeg(srcImg, dstImg, 10, 30);

	Mat t = dstImg.clone();
}