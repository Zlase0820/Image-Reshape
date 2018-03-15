#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include <cuda.h>  
#include <cuda_device_runtime_api.h>  
#include <opencv2/opencv.hpp>  
#include <stdio.h>  
#include <iostream>  
#include <time.h>
#include <math.h>

using namespace std;
using namespace cv;

// 宏定义 循环的次数
const int loop = 1;

//宏定义参数
const double aa = 0.0000147169;
const double bb = 0.00380683;
const double cc = 0.984714;


// src_cpu_data原图像指针； out_cpu_data扩充后图像指针； scale扩充倍数
int CPU_distance(uchar *src_cpu_data, uchar *out_cpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	float f_src_row;   //高
	int   i_src_row;

	float f_src_col;   //宽
	int   i_src_col;

	for (int y = 0; y < out_rows; y++)
	{
		for (int x = 0; x < out_cols; x++)
		{
			f_src_row = y / scale;   
			i_src_row = (int)f_src_row;
			if(f_src_row- i_src_row>0.5)  
			{  i_src_row++; }

			f_src_col = x / scale;
			i_src_col = (int)f_src_col;
			if (f_src_col - i_src_col>0.5)
			{  i_src_col++; }

			if (x == 0 || y == 0 || x == (out_cols - 1) || y == (out_rows - 1))   //最外侧边界保护
			{
				out_cpu_data[channels * x + y*out_cols*channels] = src_cpu_data[i_src_col * channels + i_src_row * cols * channels];    //做+1到下一个数据
				out_cpu_data[channels * x + 1 + y*out_cols*channels] = src_cpu_data[i_src_col * channels + 1 + i_src_row * cols * channels];
				out_cpu_data[channels * x + 2 + y*out_cols*channels] = src_cpu_data[i_src_col * channels + 2 + i_src_row * cols * channels];;
			}
			else
			{
				int DN = 0;
				for (int ii = 0; ii < 3; ii++)
				{
					DN = (int)( aa* src_cpu_data[channels * (i_src_col - 1) + ii + (i_src_row - 1) * cols *channels]
						+ bb * src_cpu_data[channels * (i_src_col)+ ii+(i_src_row - 1) * cols *channels]
						+ aa * src_cpu_data[channels * (i_src_col + 1) +ii+ (i_src_row - 1) * cols *channels]
						+ bb * src_cpu_data[channels * (i_src_col - 1) +ii+ (i_src_row)* cols *channels]
						+ cc * src_cpu_data[channels * (i_src_col)+ii+(i_src_row)* cols *channels]
						+ bb * src_cpu_data[channels * (i_src_col + 1) +ii+ (i_src_row)* cols *channels]
						+ aa * src_cpu_data[channels * (i_src_col + 1) +ii+ (i_src_row + 1) * cols *channels]
						+ bb * src_cpu_data[channels * (i_src_col)+ii+(i_src_row + 1) * cols *channels]
						+ aa * src_cpu_data[channels * (i_src_col + 1) +ii+ (i_src_row + 1) * cols *channels]);
					out_cpu_data[channels * x + ii + y*out_cols*channels] = DN;
				}
			}
		}
	}
	return 0;
}




int main(int argc, char **argv)
{
	cudaSetDevice(1);
	float scale = 0.25f;
	char* src_path = "map.png";

	Mat src;
	if (argc > 1)
	{
		int width = atoi(argv[1]);
		scale = atof(argv[2]);
		src = Mat::ones(width, width, CV_8UC3);
	}
	else {
		src = cv::imread(src_path,1);   //src为原图
	}

	cv::imshow("原始图像", src);
	printf("img_width = %d, scale = %lf\n", src.rows, scale);

	int rows = src.rows;              //原始图像的高度rows
	int cols = src.cols;              //原始图像的宽度cols
	int channels = src.channels();    //原始图像的通道数channels
	int out_rows = src.rows*scale;    //变换后图像高度rows
	int out_cols = src.cols*scale;    //变换后图像宽度cols


/*-------------------------------CPU图像处理-----------------------------*/
	Mat out(out_rows, out_cols, CV_8UC3);       //要输出的图像

	uchar *src_cpu_data = src.ptr<uchar>(0);   //指向了src第一行第一个元素
	uchar *out_cpu_data = out.ptr<uchar>(0);   //指向了out第一行第一个元素

	cudaEvent_t start, stop;   //使用CUDA的计时函数
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	clock_t clock_start;
	clock_t clock_end;
	clock_start = clock();

	for (int time = 0; time < loop; time++)         //运行loop次，取平均值
		CPU_distance(src_cpu_data, out_cpu_data, scale, rows, cols, channels, out_rows, out_cols);

	clock_end = clock();

	float clock_diff_sec = ((clock_end - clock_start)*1.0 / CLOCKS_PER_SEC * 1000.0 / loop);
	printf("CPU Time = %lf\n", clock_diff_sec);  

	cv::imshow("CPU处理后图像", out);
	imwrite("distance_6_scale=0.25.jpg", out);    //保存CPU处理后的结果



	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cv::waitKey(0);
	 
	return 0;
}
