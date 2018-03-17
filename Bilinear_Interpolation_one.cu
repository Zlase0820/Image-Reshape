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

const int loop = 10;

// 宏定义 Block的尺寸为 32*8
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y  8


// src_cpu_data原图像指针；out_cpu_data扩充后图像指针；scale扩充倍数
int Bilinear_Interpolation(uchar *src_cpu_data, uchar *out_cpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	float f_src_row;
	int   i_src_row;
	float cha_src_row;

	float f_src_col;    //原图横坐标float型
	int   i_src_col;    //原图横坐标int型
	float cha_src_col;

	for (int y = 0; y < out_rows; y++)
	{
		for (int x = 0; x < out_cols; x++)
		{
			int a = 0, b = 0;
			f_src_row = y / scale;
			i_src_row = (int)f_src_row;
			cha_src_row = f_src_row - i_src_row;   //v

			f_src_col = x / scale;
			i_src_col = (int)f_src_col;
			cha_src_col = f_src_col - i_src_col;   //u

			//加和中的四列分别表示左上角的点、右上角的点、左下角的点、右下角的点
			out_cpu_data[channels*x + y*out_cols*channels] =
				src_cpu_data[channels * i_src_col + i_src_row * cols * channels] * (1 - cha_src_col)*(1 - cha_src_row)
				+ src_cpu_data[channels * (i_src_col + 1) + i_src_row * cols * channels] * cha_src_col*(1 - cha_src_row)
				+ src_cpu_data[channels * i_src_col + (i_src_row + 1) * cols * channels] * (1 - cha_src_col)* cha_src_row
				+ src_cpu_data[channels * (i_src_col + 1) + (i_src_row + 1) * cols * channels] * cha_src_col * cha_src_row;
		}
	}
	return 0;
}


__global__ void BI_kernel(uchar *src_gpu_data, uchar *out_gpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;  //x
	const int y = blockIdx.y * blockDim.y + threadIdx.y;  //y   

	if (x >= out_cols || y >= out_rows)
		return;

	float f_src_col;
	int   i_src_col;
	float cha_src_col;

	float f_src_row;
	int   i_src_row;
	float cha_src_row;

	f_src_col = x / scale;
	f_src_row = y / scale;
	i_src_col = (int)f_src_col;
	i_src_row = (int)f_src_row;
	cha_src_col = f_src_col - i_src_col;  //v
	cha_src_row = f_src_row - i_src_row;  //u

	uchar *out_ptr = out_gpu_data + channels*x + y*out_cols*channels; //输出图像的坐标点
	uchar *src_ptr1 = src_gpu_data + channels * i_src_col + i_src_row * cols * channels;  //对应坐标左上角点
	uchar *src_ptr2 = src_gpu_data + channels * (i_src_col + 1) + i_src_row * cols * channels;  //对应坐标右上角点
	uchar *src_ptr3 = src_gpu_data + channels * i_src_col + (i_src_row + 1) * cols * channels;  //对应坐标左下角点
	uchar *src_ptr4 = src_gpu_data + channels * (i_src_col + 1) + (i_src_row + 1) * cols * channels;  //对应坐标右下角点

	*(out_ptr++) =
		(*src_ptr1++) * (1 - cha_src_col)*(1 - cha_src_row)
		+ (*src_ptr2++) * cha_src_col*(1 - cha_src_row)
		+ (*src_ptr3++) * (1 - cha_src_col)* cha_src_row
		+ (*src_ptr4++) * cha_src_col * cha_src_row;
}


void arr_diff(uchar *src, uchar *dst, int size)
{
	int aa = 0;
	for (int i = 0; i < size; i++)
	{
		if (src[i] != dst[i])
		{
			printf("diff, index=%d, %d != %d\n", i, src[i], dst[i]);
			aa++;
		}
	}
	cout << "错误像素点总数量为：" << aa << endl;
}


int main(int argc, char **argv)
{
	cudaSetDevice(1);
	float scale = 0.5f;
	char* src_path = "map.png";

	Mat src;
	if (argc > 1)
	{
		int width = atoi(argv[1]);
		scale = atof(argv[2]);
		src = Mat::ones(width, width, CV_8UC1);
	}
	else {
		src = cv::imread(src_path, 0);   //src为原图
	}
	// cv::imshow("原始图像", src);
	printf("img_width = %d, scale = %lf\n", src.rows, scale);

	int rows = src.rows;              //原始图像的高度rows
	int cols = src.cols;              //原始图像的宽度cols
	int channels = src.channels();    //原始图像的通道数channels
	int out_rows = src.rows*scale;    //变换后图像高度rows
	int out_cols = src.cols*scale;    //变换后图像宽度cols

/*-------------------------------CPU图像处理-----------------------------*/
	Mat out(out_rows, out_cols, CV_8UC1);  //要输出的图像

	uchar *src_cpu_data = src.ptr<uchar>(0);   //指向了src第一行第一个元素
	uchar *out_cpu_data = out.ptr<uchar>(0);   //指向了out第一行第一个元素

	cudaEvent_t start, stop;   //使用CUDA的计时函数
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	clock_t clock_start;
	clock_t clock_end;
	clock_start = clock();

	for (int time = 0; time < loop; time++)
		Bilinear_Interpolation(src_cpu_data, out_cpu_data, scale, rows, cols, channels, out_rows, out_cols);
	clock_end = clock();


	float clock_diff_sec = ((clock_end - clock_start)*1.0 / CLOCKS_PER_SEC * 1000.0 / loop);
	printf("CPU Time = %lf\n", clock_diff_sec);


	cv::imshow("CPU处理后图像", out);   //CPU处理后的结果
//  imwrite("Bilinear_interpolation.jpg",out);    //保存CPU处理后的结果



/*----------------------GPU图像处理(1线程处理1个像素点)-------------------------*/
	Mat gpu_out(out_rows, out_cols, CV_8UC1);
	uchar *src_gpu_data_host = src.ptr<uchar>(0);       //指向读入进来的原始图像
	uchar *src_gpu_data_device = NULL;                  //用于在GPU中计算的原始图像指针
	uchar *out_gpu_data_host = gpu_out.ptr<uchar>(0);   //指向最后输出的图像
	uchar *out_gpu_data_device = NULL;                  //在GPU中用于输出计算的指针

	cudaMalloc((void**)&src_gpu_data_device, sizeof(unsigned char) * rows * cols * channels);  //给输入指针内分配显存
	cudaMalloc((void**)&out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols * channels); //给输出指针分配显存
																									  //开线程（线程数量会比像素点稍微多一些）
	dim3 bdim(DEF_BLOCK_X, DEF_BLOCK_Y);
	dim3 gdim;
	gdim.x = (out_cols + bdim.x - 1) / bdim.x;
	gdim.y = (out_rows + bdim.y - 1) / bdim.y;

	cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
	BI_kernel << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
	cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);

	cudaEventRecord(start, 0);  //计时函数

	for (int time = 0; time < loop; time++)
	{
		cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
		BI_kernel << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
		cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "GPU time=" << elapsedTime / loop << endl;   //输出GPU运行的时间 

    cv::imshow("GPU处理后图像", gpu_out);//GPU的结果进行输出
//	imwrite("Bilinear.jpg",gpu_out);  //保存GPU处理的结果

	arr_diff(out_cpu_data, out_gpu_data_host, out_cols*out_rows*channels);



	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cv::waitKey(0);
	return 0;
}
