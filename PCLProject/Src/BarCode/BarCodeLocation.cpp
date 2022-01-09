#include "../../include/BarCode/BarCodeLocation.h"

const float PI = static_cast<float>(CV_PI);

//初始化图像=============================================================================================
void BarCodeDectet::ImgInit(const Mat &img)
{
	//resize(img, m_ResizeImg, cv::Size(640, 480), 0, 0, INTER_AREA);
	//m_ResizeCoff_W = img.cols / 640;
	//m_ResizeCoff_H = img.rows / 480;
	//m_ImgW = 640;
	//m_ImgH = 480;
	const double min_side = std::min(img.size().width, img.size().height);
	if (min_side > 512.0)
	{
		m_ResizeCoff = min_side / 512.0;
		m_ImgW = cvRound(img.size().width / m_ResizeCoff);
		m_ImgH = cvRound(img.size().height / m_ResizeCoff);
		Size new_size(m_ImgW, m_ImgH);
		resize(img, m_ResizeImg, new_size, 0, 0, INTER_AREA);
	}
	else
	{
		m_ResizeCoff = 1.0;
		m_ImgW = img.cols;
		m_ImgH = img.rows;
		m_ResizeImg = img.clone();
	}
}
//=======================================================================================================

//图像预处理=============================================================================================
void BarCodeDectet::ImgPreProcess()
{
	Mat grad_x, grad_y;
	Scharr(m_ResizeImg, grad_x, CV_32FC1, 1, 0);
	Scharr(m_ResizeImg, grad_y, CV_32FC1, 0, 1);
	Mat amplitude(cv::Size(m_ImgW, m_ImgH), CV_8UC1, cv::Scalar(0));

	float thresVal = 16 * 16 * 16;
	Mat grad_xx(cv::Size(m_ImgW, m_ImgH), CV_32FC1, cv::Scalar(0));
	Mat grad_yy(cv::Size(m_ImgW, m_ImgH), CV_32FC1, cv::Scalar(0));
	Mat grad_xy(cv::Size(m_ImgW, m_ImgH), CV_32FC1, cv::Scalar(0));
	float xx = 0.0f, yy = 0.0f;
	for (int y = 0; y < m_ImgH; ++y)
	{
		float *pGrad_x = grad_x.ptr<float>(y);
		float *pGrad_y = grad_y.ptr<float>(y);
		uchar *pAmp = amplitude.ptr<uchar>(y);

		float *pGrad_xx = grad_xx.ptr<float>(y);
		float *pGrad_yy = grad_yy.ptr<float>(y);
		float *pGrad_xy = grad_xy.ptr<float>(y);

		for (int x = 0; x < m_ImgW; ++x)
		{
			float xx = pGrad_x[x] * pGrad_x[x];
			float yy = pGrad_y[x] * pGrad_y[x];
			if (xx + yy > thresVal)
			{
				pGrad_xx[x] = xx;
				pGrad_yy[x] = yy;
				pAmp[x] = 1;
				pGrad_xy[x] = pGrad_x[x] * pGrad_y[x];
			}
		}
	}
	cv::integral(grad_xx, m_InterImg_xx, CV_32FC1);
	cv::integral(grad_yy, m_InterImg_yy, CV_32FC1);	
	cv::integral(grad_xy, m_InterImg_xy, CV_32FC1);
	cv::integral(amplitude, m_InterEdges, CV_32FC1);
}
//=======================================================================================================

//计算梯度方向一致性=====================================================================================
float BarCodeDectet::ComputeGradCoh_(float* const pInterImgData, int x, int y, int w_size)
{
	int img_w = m_ImgW + 1;
	float left_up = *(pInterImgData + (y - w_size) * img_w + x - w_size);
	float right_up = *(pInterImgData + (y - w_size) * img_w + x);
	float left_down = *(pInterImgData + y * img_w + x - w_size);
	float right_down = *(pInterImgData + y * img_w + x);
	return (right_down + left_up - right_up - left_down);
}
//=======================================================================================================

//计算梯度一致性=========================================================================================
void BarCodeDectet::ComputeGradCoh(int w_size)
{
	const float THRESHOLD_COHERENCE = 0.9f * 0.9f;
	const float THRESHOLD_AREA = float(w_size * w_size) * 0.42f;
	m_CohMat = Mat(cv::Size(m_ImgW / w_size, m_ImgH / w_size), CV_8UC1, cv::Scalar(0));
	m_AngMat = Mat(cv::Size(m_ImgW / w_size, m_ImgH / w_size), CV_32FC1, cv::Scalar(0));
	m_EdgesNumMat = Mat(cv::Size(m_ImgW / w_size, m_ImgH / w_size), CV_32FC1, cv::Scalar(0));
	float xx = 0.0f, yy = 0.0f, xy = 0.0f, coh = 0.0f, edges = 0.0f;
	float *pXX = m_InterImg_xx.ptr<float>(0);
	float *pYY = m_InterImg_yy.ptr<float>(0);
	float *pXY = m_InterImg_xy.ptr<float>(0);
	float *pEdges = m_InterEdges.ptr<float>(0);
	for (int y = w_size; y < m_ImgH; y += w_size)
	{
		uchar* pCohMat = m_CohMat.ptr<uchar>(y / w_size - 1);
		float* pAngMat = m_AngMat.ptr<float>(y / w_size - 1);
		float* pEdgesNum = m_EdgesNumMat.ptr<float>(y / w_size - 1);
		for (int x = w_size; x < m_ImgW; x += w_size)
		{
			edges = ComputeGradCoh_(pEdges, x, y, w_size);
			if (edges < THRESHOLD_AREA)
				continue;
			xx = ComputeGradCoh_(pXX, x, y, w_size);
			yy = ComputeGradCoh_(pYY, x, y, w_size);
			xy = ComputeGradCoh_(pXY, x, y, w_size);
			coh = ((xx - yy) * (xx - yy) + 4 * xy * xy) / ((xx + yy) * (xx + yy) + EPS);
			if (coh > THRESHOLD_COHERENCE)
			{
				pEdgesNum[x / w_size - 1] = edges;
				pCohMat[x / w_size - 1] = 255;
				pAngMat[x / w_size - 1] = std::atan2(xx - yy, 2 * xy) / 2.0f;
			}
		}
	}
}
//=======================================================================================================

//图像腐蚀===============================================================================================
void BarCodeDectet::ImageErode()
{
	Mat image = m_CohMat.clone();
	for (int y = 1; y < image.rows - 1; ++y)
	{
		uchar *pImage_up = image.ptr<uchar>(y - 1);
		uchar *pImage = image.ptr<uchar>(y);
		uchar *pImage_down = image.ptr<uchar>(y + 1);
		uchar *pCohMat = m_CohMat.ptr<uchar>(y);
		int sum = 0;
		for (int x = 1; x < image.cols - 1; ++x)
		{
			sum += (pImage_up[x - 1] == 255) && (pImage_down[x + 1] == 255) ? 1 : 0;
			sum += (pImage_up[x] == 255) && (pImage_down[x] == 255) ? 1 : 0;
			sum += (pImage_up[x + 1] == 255) && (pImage_down[x - 1] == 255) ? 1 : 0;
			sum += (pImage[x + 1] == 255) && (pImage[x - 1] == 255) ? 1 : 0;
			if (sum < 2)
				pCohMat[x] = 0;
		}
	}
	Mat testImg = m_CohMat.clone();
}
//=======================================================================================================

//区域生长===============================================================================================
void BarCodeDectet::RegionGrowing(int w_size)
{
	queue<Point2i> seeds;
	uchar *pCohMat = m_CohMat.ptr<uchar>(0);
	const float THRESHOLD_RADIAN = PI / 30, THRESHOLD_ORTCOH = 0.9f, 
		THRESHOLD_EDGES = 35, THRESHOLD_WIDTH = 95 / w_size;
	const int THRESHOLD_BLOCK = 35;
	const int DIR[8][2] = { {-1, -1},{0,  -1}, {1,  -1}, {1,  0},
						{1,  1}, {0,  1}, {-1, 1}, {-1, 0} };
	Mat testImg = m_CohMat;
	for (int y = 1; y < m_CohMat.rows - 1; ++y)
	{
		pCohMat += m_CohMat.cols;
		for (int x = 1; x < m_CohMat.cols - 1; ++x)
		{
			if (pCohMat[x] == 0)
				continue;
			vector<Point2i> region;
			pCohMat[x] = 0;
			seeds.push(Point2i(x, y));

			float cur_theta = 2.0 * m_AngMat.at<float>(seeds.front());
			float sum_sin = std::sin(cur_theta), sum_cos = std::cos(cur_theta);
			float edges_num = m_EdgesNumMat.at<float>(seeds.front());
			while (!seeds.empty())
			{
				Point2i pt = seeds.front();
				float src_theta = m_AngMat.at<float>(pt);
				region.push_back(pt);
				seeds.pop();
				for (auto offset : DIR)
				{
					Point2i pt_ = Point2i(pt.x + offset[0], pt.y + offset[1]);
					if (pt_.x < 0 || pt_.y < 0 || pt_.x > m_CohMat.cols - 1 || pt_.y > m_CohMat.rows - 1)
						continue;
					if (m_CohMat.at<uchar>(pt_) == 0)
						continue;
					cur_theta = m_AngMat.at<float>(pt_);
					if (abs(src_theta - cur_theta) < THRESHOLD_RADIAN
						|| abs(src_theta - cur_theta) > PI - THRESHOLD_RADIAN)
					{
						sum_sin += std::sin(2.0f * cur_theta);
						sum_cos += std::cos(2.0f * cur_theta);
						edges_num += m_EdgesNumMat.at<float>(pt_);
						seeds.push(pt_);
						m_CohMat.at<uchar>(pt_) = 0;
						region.push_back(pt_);
					}
				}
			}
			if (region.size() < THRESHOLD_BLOCK)
				continue;
			float ortCoh = (sum_sin * sum_sin + sum_cos * sum_cos) / static_cast<float>(region.size());
			if (ortCoh < THRESHOLD_ORTCOH)
				continue;
			RotatedRect rotRect = cv::minAreaRect(region);
			float w_h_ratio = rotRect.size.width / rotRect.size.height;
			if (w_h_ratio < 0.25 || w_h_ratio > 4)
				continue;
			float length = rotRect.size.width > rotRect.size.height ? rotRect.size.width : rotRect.size.height;
			if (length < THRESHOLD_WIDTH)
				continue;
			if (edges_num < rotRect.size.area() * float(w_size * w_size) * 0.5f)
				continue;

			rotRect.size.width *= static_cast<float>(w_size) * 1.1;
			rotRect.size.height *= static_cast<float>(w_size) * 1.1;
			rotRect.center.x = (rotRect.center.x + 0.5f) * static_cast<float>(w_size);
			rotRect.center.y = (rotRect.center.y + 0.5f) * static_cast<float>(w_size);
			m_LocationBoxes.push_back(rotRect);
			m_BboxScores.push_back(edges_num);
		}
	}
}
//=======================================================================================================

//非极大值抑制===========================================================================================
void BarCodeDectet::BoxesNMS(float scoreThreshold)
{
	//选出大于阈值的box
	std::vector<std::pair<float, int>> boxes;
	for (int i = 0; i < m_LocationBoxes.size(); ++i)
	{
		if (m_BboxScores[i] > scoreThreshold)
		{
			boxes.push_back(std::pair<float, int>(m_BboxScores[i], i));
		}
	}

	//非极大值拟制
	if (boxes.size() > 0)
	{
		vector<bool> flags(boxes.size(), false);
		for (int i = 0; i < boxes.size(); ++i)
		{
			if (!flags[i])
			{
				int i_ = i;
				flags[i] = true;
				for (int j = i + 1; j < boxes.size(); ++j)
				{
					if (!flags[j])
					{
						std::vector<Point2f> inter;
						int res = rotatedRectangleIntersection(m_LocationBoxes[boxes[i].second],
							m_LocationBoxes[boxes[j].second], inter);
						if (res)
						{
							i_ = boxes[i].first > boxes[j].first ? i : j;
						}
						flags[j] = true;
					}
				}
				m_BoxId.push_back(i_);
			}
		}
	}
}
//=======================================================================================================

//定位===================================================================================================
void BarCodeDectet::Location(const Mat &image)
{
	m_LocationBoxes.clear();
	m_BboxScores.clear();
	m_BoxId.clear();
	ImgInit(image);
	ImgPreProcess();
	for (int w_size = 5; w_size < 41; w_size += 10)
	{
		ComputeGradCoh(w_size);
		ImageErode();
		RegionGrowing(w_size);
	}
	BoxesNMS(0.0f);
}
//=======================================================================================================

void BarCodeTest()
{
	for (int i = 50; i < 150; ++i)
	{
		string imgFile = "F:/nbcode/image/BarcodeTestData/1 (" + to_string(i) + ").jpg";
		Mat image = imread(imgFile, 0);
		BarCodeDectet barCode;
		barCode.Location(image);
		//BarCodeLocation barCode;
		//barCode.Init(image);
		//barCode.Localization();
		Mat colorImg = imread(imgFile, 1);
		for (int j = 0; j < barCode.m_BoxId.size(); ++j)
		{
			cv::Point2f* vertices = new cv::Point2f[4];
			barCode.m_LocationBoxes[barCode.m_BoxId[j]].points(vertices);

			std::vector<cv::Point> contour;

			for (int k = 0; k < 4; k++)
			{
				//Point2f pt = Point2f(vertices[i].x * barCode.m_ResizeCoff_W, vertices[i].y * barCode.m_ResizeCoff_H);
				contour.push_back(vertices[k] * barCode.m_ResizeCoff);
			}

			std::vector<std::vector<cv::Point>> contours;
			contours.push_back(contour);
			cv::drawContours(colorImg, contours, 0, cv::Scalar(0, 0, 255), 2);
		}
		int ttt = 0;
	}
}