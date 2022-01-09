#include "../BaseOprFile/utils.h"

class BarCodeDectet
{
public:
	BarCodeDectet()
	{}

	//初始化图像
	void ImgInit(const Mat &img);

	//图像预处理
	void ImgPreProcess();

	//计算梯度一致性
	float ComputeGradCoh_(float* const pInterImgData, int x, int y, int w_size);

	//计算梯度方向一致性
	void ComputeGradCoh(int w_size);

	//图像腐蚀
	void ImageErode();

	//区域生长
	void RegionGrowing(int w_size);

	//非极大值抑制
	void BoxesNMS(float scoreThreshold);

	//条形码定位
	void Location(const Mat &image);

private:
	int m_ImgW;
	int m_ImgH;
	Mat m_ResizeImg;
	Mat m_InterImg_xx;
	Mat m_InterImg_yy;
	Mat m_InterImg_xy;
	Mat m_InterEdges;
	Mat m_CohMat;
	Mat m_AngMat;
	Mat m_EdgesNumMat;

public:
	float m_ResizeCoff;
	vector<RotatedRect> m_LocationBoxes;
	vector<float> m_BboxScores;
	vector<int> m_BoxId;
};

//图像预处理
void ImgPreProcess(Mat &srcImg, Mat & dstImg);

void BarCodeTest();