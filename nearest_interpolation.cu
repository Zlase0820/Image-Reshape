#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include <cuda.h>  
#include <cuda_device_runtime_api.h>  
#include <opencv2\gpu\gpu.hpp>  
#include <opencv2\gpu\gpumat.hpp>  
#include <opencv2\opencv.hpp>  
#include <opencv.hpp>  
#include <stdio.h>  
#include <iostream>  
#include "opencv2/gpu/device/common.hpp"  
#include "opencv2/gpu/device/reduce.hpp"  
#include "opencv2/gpu/device/functional.hpp"  
#include "opencv2/gpu/device/warp_shuffle.hpp"  
#include "windows.h"

#include "Binarize.h"

using namespace std;
using namespace cv;
using namespace gpu;

// 宏定义 Block的尺寸为 16*2
#define DEF_BLOCK_X  16
#define DEF_BLOCK_Y  2


// src_cpu_data原图像指针；out_cpu_data扩充后图像指针；scale扩充倍数
int NearestInterpolation(uchar *src_cpu_data, uchar *out_cpu_data, float scale,int rows,int cols,int channels,int out_rows,int out_cols)
{
	float f_src_row;
	int   i_src_row;

	float f_src_col;    //原图横坐标float型
	int   i_src_col;    //原图横坐标int型

	for (int y = 0; y < out_rows; y++)   
	{
		for (int x = 0; x < out_cols ; x++)
		{
			int a = 0, b = 0;
			f_src_row = y / scale;
			i_src_row = (int)f_src_row;
			
			f_src_col = x / scale;
			i_src_col = (int)f_src_col;			
			
			if ((f_src_row - i_src_row) >= 0.5 && i_src_row <(rows - 1) )    //i_out_row <(rows - 1)只是为了防止超出边框
				a=1;

			if ((f_src_col - i_src_col) >= 0.5 && i_src_col < (cols - 1))
				b=1;

			out_cpu_data [ 3 * x + y*out_cols*channels] = src_cpu_data [ 3 * (i_src_col + a) + (i_src_row + b)*cols*channels];
			out_cpu_data[3 * x + y*out_cols*channels+1] = src_cpu_data[3 * (i_src_col + a) + (i_src_row + b)*cols*channels+1];
			out_cpu_data[3 * x + y*out_cols*channels+2] = src_cpu_data[3 * (i_src_col + a) + (i_src_row + b)*cols*channels+2];

		}
	}
	return 0;
}


template <int nthreads>
__global__ void NI_kernel(uchar *src_gpu_data, uchar *out_gpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;  //x
	const int y = blockIdx.y * blockDim.y + threadIdx.y;  //y	

	float f_src_col;
	int   i_src_col;
	float cha_col;

	float f_src_row;
	int   i_src_row;
	float cha_row;

	int a = 0, b = 0;

	f_src_col = x / scale;
	f_src_row = y / scale;
	i_src_col = (int)f_src_col;
	i_src_row = (int)f_src_row;
	cha_col   = f_src_col - i_src_col;
	cha_row   = f_src_row - i_src_row;

//-----------新版本----------
//旧版本每一个线程需要需要比较4次，新版本每个线程只需要比较2次。
//旧版本赋值过程太繁琐，已简化为三行。
//旧版本使用了GpuMat，本版本使用指针进行传入，避免了黑盒调试

	if (cha_col >= 0.5)  a++;
	if (cha_row >= 0.5)  b++;
	out_gpu_data [ channels*x + y*out_cols*channels] = src_gpu_data [channels*(i_src_col + a) + (i_src_row + b)*cols*channels];
	out_gpu_data[channels*x + y*out_cols*channels+1] = src_gpu_data[channels*(i_src_col + a) + (i_src_row + b)*cols*channels+1];
	out_gpu_data[channels*x + y*out_cols*channels+2] = src_gpu_data[channels*(i_src_col + a) + (i_src_row + b)*cols*channels+2];
}



__global__ void NI2_kernel(uchar *src_gpu_data, uchar *out_gpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	const int x = (blockIdx.x * blockDim.x + threadIdx.x)*4;          //x
	const int y = blockIdx.y * blockDim.y + threadIdx.y  ;    //y	

	if (x >= out_cols || y >= out_rows)
		return;

	float f_src_col;
	int   i_src_col;
	float cha_col;

	float f_src_row;
	int   i_src_row;
	float cha_row;

	int a = 0, b = 0;

	f_src_col = x / scale;
	f_src_row = y / scale;
	i_src_col = (int)f_src_col;
	i_src_row = (int)f_src_row;
	cha_col = f_src_col - i_src_col;
	cha_row = f_src_row - i_src_row;

	for (int i = 0; i < 4; i++)
	{
		int xx = x + i;
		if (cha_col >= 0.5)  a++;
		if (cha_row >= 0.5)  b++;

		out_gpu_data[channels*xx + y*out_cols*channels] = src_gpu_data[channels*(i_src_col + a+i) + (i_src_row + b )*cols*channels];
		out_gpu_data[channels*xx + y*out_cols*channels+1] = src_gpu_data[channels*(i_src_col + a+i) + (i_src_row + b )*cols*channels+1];
		out_gpu_data[channels*xx + y*out_cols*channels+2] = src_gpu_data[channels*(i_src_col + a+i) + (i_src_row + b )*cols*channels+2];
	}

}







int main()
{
	float scale = 0.6f;   
	char* src_path = "teddy512.bmp";  

	Mat src = cv::imread(src_path, CV_LOAD_IMAGE_COLOR);   //src为原图


	cv::imshow("原始图像", src);

	int rows = src.rows;              //原始图像的高度rows
	int cols = src.cols;              //原始图像的宽度cols
	int channels = src.channels();    //原始图像的通道数channels
	int out_rows = src.rows*scale;    //变换后图像高度rows
	int out_cols = src.cols*scale;    //变换后图像宽度cols

/*-------------------------------CPU图像处理-----------------------------
	Mat out(out_rows, out_cols, CV_8UC3);  //要输出的图像

	uchar *src_cpu_data = src.ptr<uchar>(0);   //指向了src第一行第一个元素
	uchar *out_cpu_data = out.ptr<uchar>(0);   //指向了out第一行第一个元素
	
	LARGE_INTEGER cpu_t1, cpu_t2, cpu_tc;
	QueryPerformanceFrequency(&cpu_tc);
	QueryPerformanceCounter(&cpu_t1);
	
	for (int time = 0; time < 100; time++)         //运行100次，取平均值
	{
		NearestInterpolation(src_cpu_data, out_cpu_data, scale,rows,cols,channels,out_rows,out_cols);
	}
	
	QueryPerformanceCounter(&cpu_t2);
	std::cout << "使用CPU做最邻近插值法的时间：" << (cpu_t2.QuadPart - cpu_t1.QuadPart) * 1.0 * 1000 / cpu_tc.QuadPart /100 << "ms" << endl;
	
	cv::imshow("CPU处理后图像", out);	
*/


/*----------------------GPU图像处理(1线程处理1个像素点)-------------------------
	Mat gpu_out(out_rows, out_cols, CV_8UC3);
	uchar *src_gpu_data_host = src.ptr<uchar>(0);       //指向读入进来的原始图像
	uchar *src_gpu_data_device = NULL;                         //用于在GPU中计算的原始图像指针
	uchar *out_gpu_data_host = gpu_out.ptr<uchar>(0);   //指向最后输出的图像
	uchar *out_gpu_data_device = NULL;                         //在GPU中用于输出计算的指针

	cudaMalloc((void**)&src_gpu_data_device, sizeof(unsigned char) * rows * cols * channels);  //给输入指针内分配显存
	cudaMalloc((void**)&out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols * channels); //给输出指针分配显存

	//将数据传入GPU中
	cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);

	//开线程，启动跟out图像像素点一样多的线程
	const int nthreads = 256;
	dim3 bdim(nthreads, 1);
	dim3 gdim(divUp(out_cols, bdim.x), divUp(out_rows, bdim.y));

	//计时函数
	LARGE_INTEGER gpu_t1, gpu_t2, gpu_tc;
	QueryPerformanceFrequency(&gpu_tc);
	QueryPerformanceCounter(&gpu_t1);
	for (int time = 0; time < 100; time++)
	{
		NI_kernel<nthreads> << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
		cudaDeviceSynchronize();
	}
	QueryPerformanceCounter(&gpu_t2);
	cout << "使用GPU做最邻近插值法的时间：" << (gpu_t2.QuadPart - gpu_t1.QuadPart) * 1.0 * 1000 / gpu_tc.QuadPart / 100 << "ms" << endl;
	
	//将结果传回内存
	cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);
	cv::imshow("GPU处理后图像", gpu_out);//GPU的结果进行输出
*/


/*-----------------------GPU图像处理（1线程处理横向4个像素点）--------------*/
	Mat gpu_out(out_rows, out_cols, CV_8UC3);
	uchar *src_gpu_data_host = src.ptr<uchar>(0);       //指向读入进来的原始图像
	uchar *src_gpu_data_device = NULL;                         //用于在GPU中计算的原始图像指针
	uchar *out_gpu_data_host = gpu_out.ptr<uchar>(0);   //指向最后输出的图像
	uchar *out_gpu_data_device = NULL;                         //在GPU中用于输出计算的指针

	cudaMalloc((void**)&src_gpu_data_device, sizeof(unsigned char) * rows * cols * channels);  //给输入指针内分配显存
	cudaMalloc((void**)&out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols * channels); //给输出指针分配显存

																									  //将数据传入GPU中
	cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);

	//开线程，启动输出图像1/4的线程
	dim3 bdim, gdim;
	bdim.x = DEF_BLOCK_X;
	bdim.y = DEF_BLOCK_Y;
	gdim.x = (out_cols + bdim.x * 4 - 1) /(bdim.x * 4);
	gdim.y = (out_rows + bdim.y - 1) /bdim.y;


	//计时函数
	LARGE_INTEGER gpu_t1, gpu_t2, gpu_tc;
	QueryPerformanceFrequency(&gpu_tc);
	QueryPerformanceCounter(&gpu_t1);
	for (int time = 0; time < 100; time++)
	{
		NI2_kernel << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
		cudaDeviceSynchronize();
	}
	QueryPerformanceCounter(&gpu_t2);
	cout << "使用GPU做最邻近插值法的时间：" << (gpu_t2.QuadPart - gpu_t1.QuadPart) * 1.0 * 1000 / gpu_tc.QuadPart / 100 << "ms" << endl;


	//将结果传回内存
	cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);

	cv::imshow("GPU处理后图像", gpu_out);//GPU的结果进行输出





	cv::waitKey(0);
	return 0;
}