#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include <cuda.h>  
#include <cuda_device_runtime_api.h>  
#include <opencv2/opencv.hpp>  
#include <stdio.h>  
#include <iostream>  
#include <time.h>

using namespace std;
using namespace cv;

//out1_data是小图，out2_data是大图
int Cubic_return(uchar *out1_data, uchar *out2_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	float f_src_row=0;    //高
	int   i_src_row=0;

	float f_src_col=0;    //宽
	int   i_src_col=0;    

	for (int y = 0; y < out_rows; y++)
	{
		for (int x = 0; x < out_cols; x++)
		{

			f_src_row = y / scale;
			i_src_row = (int)f_src_row;

			f_src_col = x / scale;
			i_src_col = (int)f_src_col;

			if (i_src_row >= rows || i_src_col >= cols)
				return 0;
			else
			{
				for (int i = 0; i < 3; i++)
				{
					out2_data[channels*i_src_col + i + i_src_row*cols*channels] = out1_data[channels*x + i + y*out_cols*channels];
				}
			}
		}
	}
	return 0;
}



int main()
{
	float scale = 0.5f;

	char* src_path = "map.png";
	Mat src = cv::imread(src_path, 1);   //src为原图
	cv::imshow("原始图像", src);

	char* out_path1 = "distance_6_scale=0.5.jpg";
	Mat out = cv::imread(out_path1, 1);   //小图
	cv::imshow("Cubic处理后图像", out);

	int rows = src.rows;              //原始图像的高度rows
	int cols = src.cols;              //原始图像的宽度cols
	int channels = src.channels();    //原始图像的通道数channels
	int out_rows = out.rows;	       //变换后图像高度rows
	int out_cols = out.cols;          //变换后图像宽度cols

														
	uchar *out_data = out.ptr<uchar>(0);   //指向了Cubic处理后的结果
	uchar *src_data = src.ptr<uchar>(0);    //指向了最终输出的结果

	Cubic_return(out_data, src_data, scale, rows, cols, channels, out_rows, out_cols);

	cv::imshow("return后的结果", src);
	imwrite("distance_return_6_scale=0.5.jpg",src);
	cv::waitKey(0);
	return 0;
}
