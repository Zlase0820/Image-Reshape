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

// �궨�� ѭ���Ĵ���
const int loop = 1;

//�궨�����
const double aa = 0.0000147169;
const double bb = 0.00380683;
const double cc = 0.984714;


// src_cpu_dataԭͼ��ָ�룻 out_cpu_data�����ͼ��ָ�룻 scale���䱶��
int CPU_distance(uchar *src_cpu_data, uchar *out_cpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	float f_src_row;   //��
	int   i_src_row;

	float f_src_col;   //��
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

			if (x == 0 || y == 0 || x == (out_cols - 1) || y == (out_rows - 1))   //�����߽籣��
			{
				out_cpu_data[channels * x + y*out_cols*channels] = src_cpu_data[i_src_col * channels + i_src_row * cols * channels];    //��+1����һ������
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
		src = cv::imread(src_path,1);   //srcΪԭͼ
	}

	cv::imshow("ԭʼͼ��", src);
	printf("img_width = %d, scale = %lf\n", src.rows, scale);

	int rows = src.rows;              //ԭʼͼ��ĸ߶�rows
	int cols = src.cols;              //ԭʼͼ��Ŀ��cols
	int channels = src.channels();    //ԭʼͼ���ͨ����channels
	int out_rows = src.rows*scale;    //�任��ͼ��߶�rows
	int out_cols = src.cols*scale;    //�任��ͼ����cols


/*-------------------------------CPUͼ����-----------------------------*/
	Mat out(out_rows, out_cols, CV_8UC3);       //Ҫ�����ͼ��

	uchar *src_cpu_data = src.ptr<uchar>(0);   //ָ����src��һ�е�һ��Ԫ��
	uchar *out_cpu_data = out.ptr<uchar>(0);   //ָ����out��һ�е�һ��Ԫ��

	cudaEvent_t start, stop;   //ʹ��CUDA�ļ�ʱ����
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	clock_t clock_start;
	clock_t clock_end;
	clock_start = clock();

	for (int time = 0; time < loop; time++)         //����loop�Σ�ȡƽ��ֵ
		CPU_distance(src_cpu_data, out_cpu_data, scale, rows, cols, channels, out_rows, out_cols);

	clock_end = clock();

	float clock_diff_sec = ((clock_end - clock_start)*1.0 / CLOCKS_PER_SEC * 1000.0 / loop);
	printf("CPU Time = %lf\n", clock_diff_sec);  

	cv::imshow("CPU�����ͼ��", out);
	imwrite("distance_6_scale=0.25.jpg", out);    //����CPU�����Ľ��



	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cv::waitKey(0);
	 
	return 0;
}
