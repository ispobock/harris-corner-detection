#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/contrib/contrib.hpp>
#include "string"
#include "iostream"
#include "math.h"

using namespace std;
using namespace cv;

string img_path; //图片路径
int window_size = 5; //窗口大小
const double alpha = 0.05; //R值计算参数，默认
const double threshold_level = 0.01; //R值阈值，默认
const string save_path = "imgs/"; //结果保存路径，默认在imgs/下

void convertToGray(Mat img, Mat& gray) {
	// BGR三通道图转化为灰度图
	if (img.channels() == 3) {
		cvtColor(img, gray, CV_BGR2GRAY);
	}
	else {
		gray = img.clone();
	}
	return;
}

void harrisCorner(Mat gray, Mat& R, Mat& lambda_max, Mat& lambda_min) {
	/*
	此函数输入一张灰度图像，用于得到R, lambda_max, lambda_min
	@param: gray: 灰度图像，
	@param: R: R值矩阵，
	@param: lambda_max: 最大特征值矩阵，
	@param: lambda_min: 最小特征值矩阵
	*/

	// 对每个像素点求x,y两个方向的梯度
	Mat Ix, Iy;
	Sobel(gray, Ix, CV_64F, 1, 0, 3); //求x方向梯度
	Sobel(gray, Iy, CV_64F, 0, 1, 3); //求y方向梯度

	// 梯度相乘，得到M矩阵的元素
	Mat Ix2, Iy2, IxIy;
	Ix2 = Ix.mul(Ix);
	Iy2 = Iy.mul(Iy);
	IxIy = Ix.mul(Iy);

	// 通过卷积的方式，对窗口内的梯度求和
	Mat Ix2_sum, Iy2_sum, IxIy_sum;
	Mat kernel(window_size, window_size, CV_64F, 1);
	filter2D(Ix2, Ix2_sum, CV_64F, kernel);
	filter2D(Iy2, Iy2_sum, CV_64F, kernel);
	filter2D(IxIy, IxIy_sum, CV_64F, kernel);

	// 计算指标R，最大特征值矩阵、最小特征值矩阵
	double A, B, C, d, t;
	for (int i = 0; i < gray.rows; i++) {
		for (int j = 0; j < gray.cols; j++) {
			A = Ix2_sum.at<double>(i, j);
			B = IxIy_sum.at<double>(i, j);
			C = Iy2_sum.at<double>(i, j);
			d = A * C - B * B;
			t = A + C;
			R.at<double>(i, j) = d - alpha * t * t;
			lambda_max.at<double>(i, j) = (t + sqrt(t * t - 4 * d)) / 2;
			lambda_min.at<double>(i, j) = (t - sqrt(t * t - 4 * d)) / 2;
		}
	}
	return;
}

// 将矩阵的元素映射到[0,255]之间，再进行绘图
void showMat(Mat M, string windowName) {
	double max, min;
	minMaxLoc(M, &min, &max, NULL, NULL);
	if (max > 255 || min < 0) {
		M = (M - min) / (max - min) * 255;
	}
	imshow(windowName, M);
	string img_name = img_path.substr(0, img_path.length() - 4);
	imwrite(save_path + img_name + "_" + windowName + ".png", M);
	return;
}

void showRHeatmap(Mat R) {
	double R_max, R_min;
	minMaxLoc(R, &R_min, &R_max, NULL, NULL);

	// 得到R值热力图，需要做个映射
	Mat R_colored;
	R_colored = (R - R_min) / (R_max - R_min) * 255;
	R_colored.convertTo(R_colored, CV_8UC1);
	applyColorMap(R_colored, R_colored, COLORMAP_JET);

	showMat(R_colored, "R_HeatMap");
	return;
}

// 为R值设定阈值
void setThreshold(Mat& R) {
	double R_max;
	minMaxLoc(R, NULL, &R_max, NULL, NULL);
	double threshold = threshold_level * R_max;
	for (int i = 0; i < R.rows; i++) {
		for (int j = 0; j < R.cols; j++) {
			if (R.at<double>(i, j) < threshold) {
				R.at<double>(i, j) = 0;
			}
		}
	}
	return;
}

// 局部最大值抑制
void getLocalMax(Mat R, Mat& localMax) {
	Mat dilated;
	dilate(R, dilated, Mat());
	compare(R, dilated, localMax, CMP_EQ);
	localMax = localMax - 255 + (dilated > 0);
	return;
}

// 将角点画到图像上
void drawPointsToImg(Mat img, Mat localMax) {
	localMax.convertTo(localMax, CV_64F);
	for (int i = 0; i < localMax.rows; i++) {
		for (int j = 0; j < localMax.cols; j++) {
			if (localMax.at<double>(i, j) != 0) {
				Point p = Point(j, i);
				circle(img, p, 1, Scalar(0, 255, 0), -1);
			}
		}
	}
	showMat(img, "Image_with_Corners");
	return;
}

int main() {
	// 读取图片路径，窗口大小
	cout << "Please input a image path (e.g. test.png): " << endl;
	cin >> img_path;
	cout << "Please input window size of Harris Corner Detection (e.g. 5): " << endl;
	cin >> window_size;

	// 读入图像并转为灰度图
	Mat img = imread(img_path);
	Mat gray;
	convertToGray(img, gray);

	// 初始化R, lambda_max, lambda_min，并进行Harris角点检测
	Mat R(gray.size(), CV_64F);
	Mat lambda_max(gray.size(), CV_64F), lambda_min(gray.size(), CV_64F);
	harrisCorner(gray, R, lambda_max, lambda_min);
	
	// 将最大特征值矩阵和最小特征值矩阵映射至[0,255]之内
	showMat(lambda_max, "max_lambda"); 
	showMat(lambda_min, "min_lambda");

	// 画出R值热力图
	showRHeatmap(R);

	// 设定Threshold
	setThreshold(R); 
	showMat(R, "R_with_Threshold");

	// 得出局部最大值点
	Mat localMax;
	getLocalMax(R, localMax);
	showMat(localMax, "localMax");

	// 将局部最大值点绘制到原始图像上
	drawPointsToImg(img, localMax);

	waitKey(0);

	return 0;
}