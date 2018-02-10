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
//#define DEF_BLOCK_X  16
//#define DEF_BLOCK_Y  2


// src_cpu_data原图像指针；out_cpu_data扩充后图像指针；scale扩充倍数
int NearestInterpolation(uchar* &src_cpu_data, uchar* &out_cpu_data, float scale,int rows,int cols,int channels,int out_rows,int out_cols)
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

			*(out_cpu_data + 3 * x + y*out_cols*channels) = *(src_cpu_data + 3 * (i_src_col + a) + (i_src_row + b)*cols*channels);
			*(out_cpu_data + 3 * x + y*out_cols*channels+1) = *(src_cpu_data + 3 * (i_src_col + a) + (i_src_row + b)*cols*channels+1);
			*(out_cpu_data + 3 * x + y*out_cols*channels+2) = *(src_cpu_data + 3 * (i_src_col + a) + (i_src_row + b)*cols*channels+2);
		}
	}
	return 0;
}


template <int nthreads>
__global__ void NI_kernel(uchar* &src_gpu_data, uchar* &out_gpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;  //x
	const int y = blockIdx.y * blockDim.y + threadIdx.y;  //y	

	printf("1");
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
	*(out_gpu_data + x + y*out_cols*channels) = *(src_gpu_data + (i_src_col + a) + (i_src_row + b)*cols*channels);
	


//	*(out_gpu_data + 3 * x + y*out_cols*channels) = *(src_gpu_data + 3 * (i_src_col + a) + (i_src_row + b)*cols*channels);
//	*(out_gpu_data + 3 * x + y*out_cols*channels + 1) = *(src_gpu_data + 3 * (i_src_col + a) + (i_src_row + b)*cols*channels + 1);
//	*(out_gpu_data + 3 * x + y*out_cols*channels + 2) = *(src_gpu_data + 3 * (i_src_col + a) + (i_src_row + b)*cols*channels + 2);
	
}



int main()
{
	float scale = 0.6f;   
	char* src_path = "teddy.bmp";  

//	Mat src = cv::imread(src_path, CV_LOAD_IMAGE_COLOR);   //src为原图
	Mat src = cv::imread(src_path, 0);   //src为原图

	cv::imshow("原始图像", src);

	int rows = src.rows;              //原始图像的高度rows
	int cols = src.cols;              //原始图像的宽度cols
	int channels = src.channels();    //原始图像的通道数channels
	int out_rows = src.rows*scale;    //变换后图像高度rows
	int out_cols = src.cols*scale;    //变换后图像宽度cols

/*-------------------------------CPU图像处理-----------------------------*/
	Mat out(out_rows, out_cols, CV_8UC1);  //要输出的图像

	uchar *src_cpu_data = src.ptr<uchar>(0);   //指向了src第一行第一个元素
	uchar *out_cpu_data = out.ptr<uchar>(0);   //指向了out第一行第一个元素
	
	LARGE_INTEGER cpu_t1, cpu_t2, cpu_tc;
	QueryPerformanceFrequency(&cpu_tc);
	QueryPerformanceCounter(&cpu_t1);
	
	for (int time = 0; time < 100; time++)         //运行100次，取平均值
	{
//		NearestInterpolation(src_cpu_data, out_cpu_data, scale,rows,cols,channels,out_rows,out_cols);
	}
	
	QueryPerformanceCounter(&cpu_t2);
	std::cout << "使用CPU做最邻近插值法的时间：" << (cpu_t2.QuadPart - cpu_t1.QuadPart) * 1.0 * 1000 / cpu_tc.QuadPart /100 << "ms" << endl;



/*----------------------GPU图像处理-------------------------*/
	Mat gpu_out(out_rows, out_cols, CV_8UC3);
	uchar *src_gpu_data = src.ptr<uchar>(0);       //指向读入进来的原始图像
	uchar *out_gpu_data = gpu_out.ptr<uchar>(0);   //指向新生成的GPU型

	const int nthreads = 256;
	dim3 bdim(nthreads, 1);
	dim3 gdim(divUp(out.cols, bdim.x), divUp(out.rows, bdim.y));

	LARGE_INTEGER gpu_t1, gpu_t2, gpu_tc;
	QueryPerformanceFrequency(&gpu_tc);
	QueryPerformanceCounter(&gpu_t1);
//	for (int time = 0; time < 100; time++)
//	{
		NI_kernel<nthreads> << <gdim, bdim >> > (src_gpu_data, out_gpu_data, scale, rows, cols, channels, out_rows, out_cols);
		cudaDeviceSynchronize();
//	}
	QueryPerformanceCounter(&gpu_t2);
	cout << "使用GPU做最邻近插值法的时间：" << (gpu_t2.QuadPart - gpu_t1.QuadPart) * 1.0 * 1000 / gpu_tc.QuadPart / 100 << "ms" << endl;




	cv::imshow("GPU处理后图像", gpu_out);//GPU的结果进行输出
	cv::imshow("CPU处理后图像", out);

	cv::waitKey(0);
	return 0;
}