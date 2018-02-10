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



using namespace std;
using namespace cv;
using namespace gpu;

// �궨�� Block�ĳߴ�Ϊ 16*2
#define DEF_BLOCK_X  16
#define DEF_BLOCK_Y  2


// src_cpu_dataԭͼ��ָ�룻out_cpu_data�����ͼ��ָ�룻scale���䱶��
int Bilinear_Interpolation(uchar *src_cpu_data, uchar *out_cpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	float f_src_row;
	int   i_src_row;
	float cha_src_row;

	float f_src_col;    //ԭͼ������float��
	int   i_src_col;    //ԭͼ������int��
	float cha_src_col;  

	for (int y = 0; y < out_rows; y++)
	{
		for (int x = 0; x < out_cols; x++)
		{
			int a = 0, b = 0;
			f_src_row = y / scale;
			i_src_row = (int)f_src_row;
			cha_src_row = f_src_row - i_src_row;

			f_src_col = x / scale;
			i_src_col = (int)f_src_col;
			cha_src_col = f_src_col - i_src_col;

			//�Ӻ��е����зֱ��ʾ���Ͻǵĵ㡢���Ͻǵĵ㡢���½ǵĵ㡢���½ǵĵ�
			out_cpu_data[channels*x + y*out_cols*channels] =
				src_cpu_data[channels * i_src_col + i_src_row * cols * channels] * (1 - cha_src_col)*(1 - cha_src_row)
				+ src_cpu_data[channels * (i_src_col + 1) + i_src_row * cols * channels] * cha_src_col*(1 - cha_src_row)
				+ src_cpu_data[channels * i_src_col + (i_src_row + 1) * cols * channels] * (1 - cha_src_col)* cha_src_row
				+ src_cpu_data[channels * (i_src_col + 1) + (i_src_row + 1) * cols * channels] * cha_src_col * cha_src_row;
			out_cpu_data[channels*x + y*out_cols*channels+1] =
				src_cpu_data[channels * i_src_col + i_src_row * cols * channels+1] * (1 - cha_src_col)*(1 - cha_src_row)
				+ src_cpu_data[channels * (i_src_col + 1) + i_src_row * cols * channels+1] * cha_src_col*(1 - cha_src_row)
				+ src_cpu_data[channels * i_src_col + (i_src_row + 1) * cols * channels+1] * (1 - cha_src_col)* cha_src_row
				+ src_cpu_data[channels * (i_src_col + 1) + (i_src_row + 1) * cols * channels+1] * cha_src_col * cha_src_row;
			out_cpu_data[channels*x + y*out_cols*channels + 2] =
				src_cpu_data[channels * i_src_col + i_src_row * cols * channels + 2] * (1 - cha_src_col)*(1 - cha_src_row)
				+ src_cpu_data[channels * (i_src_col + 1) + i_src_row * cols * channels + 2] * cha_src_col*(1 - cha_src_row)
				+ src_cpu_data[channels * i_src_col + (i_src_row + 1) * cols * channels + 2] * (1 - cha_src_col)* cha_src_row
				+ src_cpu_data[channels * (i_src_col + 1) + (i_src_row + 1) * cols * channels + 2] * cha_src_col * cha_src_row;

		}
	}
	return 0;
}


template <int nthreads>
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

	int a = 0, b = 0;

	f_src_col = x / scale;
	f_src_row = y / scale;
	i_src_col = (int)f_src_col;
	i_src_row = (int)f_src_row;
	cha_src_col = f_src_col - i_src_col;
	cha_src_row = f_src_row - i_src_row;
	
	out_gpu_data[channels*x + y*out_cols*channels] =
		src_gpu_data[channels * i_src_col + i_src_row * cols * channels] * (1 - cha_src_col)*(1 - cha_src_row)
		+ src_gpu_data[channels * (i_src_col + 1) + i_src_row * cols * channels] * cha_src_col*(1 - cha_src_row)
		+ src_gpu_data[channels * i_src_col + (i_src_row + 1) * cols * channels] * (1 - cha_src_col)* cha_src_row
		+ src_gpu_data[channels * (i_src_col + 1) + (i_src_row + 1) * cols * channels] * cha_src_col * cha_src_row;
	out_gpu_data[channels*x + y*out_cols*channels+1] =
		src_gpu_data[channels * i_src_col + i_src_row * cols * channels+1] * (1 - cha_src_col)*(1 - cha_src_row)
		+ src_gpu_data[channels * (i_src_col + 1) + i_src_row * cols * channels+1] * cha_src_col*(1 - cha_src_row)
		+ src_gpu_data[channels * i_src_col + (i_src_row + 1) * cols * channels+1] * (1 - cha_src_col)* cha_src_row
		+ src_gpu_data[channels * (i_src_col + 1) + (i_src_row + 1) * cols * channels+1] * cha_src_col * cha_src_row;
	out_gpu_data[channels*x + y*out_cols*channels+2] =
		src_gpu_data[channels * i_src_col + i_src_row * cols * channels+2] * (1 - cha_src_col)*(1 - cha_src_row)
		+ src_gpu_data[channels * (i_src_col + 1) + i_src_row * cols * channels+2] * cha_src_col*(1 - cha_src_row)
		+ src_gpu_data[channels * i_src_col + (i_src_row + 1) * cols * channels+2] * (1 - cha_src_col)* cha_src_row
		+ src_gpu_data[channels * (i_src_col + 1) + (i_src_row + 1) * cols * channels+2] * cha_src_col * cha_src_row;
	
}



__global__ void BI2_kernel(uchar *src_gpu_data, uchar *out_gpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;          //x
	const int y = blockIdx.y * blockDim.y + threadIdx.y;    //y	

	if (x >= out_cols || y >= out_rows)
		return;

	float f_src_col;
	int   i_src_col;
	float cha_src_col;

	float f_src_row;
	int   i_src_row;
	float cha_src_row;

	int a = 0, b = 0;

	f_src_col = x / scale;
	f_src_row = y / scale;
	i_src_col = (int)f_src_col;
	i_src_row = (int)f_src_row;
	cha_src_col = f_src_col - i_src_col;
	cha_src_row = f_src_row - i_src_row;

	for (int i = 0; i < 4; i++)
	{
		out_gpu_data[channels*(x+i) + y*out_cols*channels] =
			src_gpu_data[channels * (i_src_col+i) + i_src_row * cols * channels] * (1 - cha_src_col)*(1 - cha_src_row)
			+ src_gpu_data[channels * (i_src_col + 1+i) + i_src_row * cols * channels] * cha_src_col*(1 - cha_src_row)
			+ src_gpu_data[channels * (i_src_col+i) + (i_src_row + 1) * cols * channels] * (1 - cha_src_col)* cha_src_row
			+ src_gpu_data[channels * (i_src_col + 1+i) + (i_src_row + 1) * cols * channels] * cha_src_col * cha_src_row;
		out_gpu_data[channels*(x + i) + y*out_cols*channels+1] =
			src_gpu_data[channels * (i_src_col + i) + i_src_row * cols * channels+1] * (1 - cha_src_col)*(1 - cha_src_row)
			+ src_gpu_data[channels * (i_src_col + 1 + i) + i_src_row * cols * channels+1] * cha_src_col*(1 - cha_src_row)
			+ src_gpu_data[channels * (i_src_col + i) + (i_src_row + 1) * cols * channels+1] * (1 - cha_src_col)* cha_src_row
			+ src_gpu_data[channels * (i_src_col + 1 + i) + (i_src_row + 1) * cols * channels+1] * cha_src_col * cha_src_row;
		out_gpu_data[channels*(x + i) + y*out_cols*channels+2] =
			src_gpu_data[channels * (i_src_col + i) + i_src_row * cols * channels+2] * (1 - cha_src_col)*(1 - cha_src_row)
			+ src_gpu_data[channels * (i_src_col + 1 + i) + i_src_row * cols * channels+2] * cha_src_col*(1 - cha_src_row)
			+ src_gpu_data[channels * (i_src_col + i) + (i_src_row + 1) * cols * channels+2] * (1 - cha_src_col)* cha_src_row
			+ src_gpu_data[channels * (i_src_col + 1 + i) + (i_src_row + 1) * cols * channels+2] * cha_src_col * cha_src_row;
	}
}




int main()
{
	float scale = 0.6f;
	char* src_path = "teddy.bmp";

	Mat src = cv::imread(src_path, CV_LOAD_IMAGE_COLOR);   //srcΪԭͼ


	cv::imshow("ԭʼͼ��", src);

	int rows = src.rows;              //ԭʼͼ��ĸ߶�rows
	int cols = src.cols;              //ԭʼͼ��Ŀ��cols
	int channels = src.channels();    //ԭʼͼ���ͨ����channels
	int out_rows = src.rows*scale;    //�任��ͼ��߶�rows
	int out_cols = src.cols*scale;    //�任��ͼ����cols

/*-------------------------------CPUͼ����-----------------------------
	Mat out(out_rows, out_cols, CV_8UC3);  //Ҫ�����ͼ��

	uchar *src_cpu_data = src.ptr<uchar>(0);   //ָ����src��һ�е�һ��Ԫ��
	uchar *out_cpu_data = out.ptr<uchar>(0);   //ָ����out��һ�е�һ��Ԫ��

	LARGE_INTEGER cpu_t1, cpu_t2, cpu_tc;
	QueryPerformanceFrequency(&cpu_tc);
	QueryPerformanceCounter(&cpu_t1);
	for (int time = 0; time < 100; time++)         //����100�Σ�ȡƽ��ֵ
	{
		Bilinear_Interpolation(src_cpu_data, out_cpu_data, scale,rows,cols,channels,out_rows,out_cols);
	}
	QueryPerformanceCounter(&cpu_t2);
	std::cout << "ʹ��CPU�����ڽ���ֵ����ʱ�䣺" << (cpu_t2.QuadPart - cpu_t1.QuadPart) * 1.0 * 1000 / cpu_tc.QuadPart /100 << "ms" << endl;

	cv::imshow("CPU�����ͼ��", out);
*/


/*----------------------GPUͼ����(1�̴߳���1�����ص�)-------------------------
	Mat gpu_out(out_rows, out_cols, CV_8UC3);
	uchar *src_gpu_data_host = src.ptr<uchar>(0);       //ָ����������ԭʼͼ��
	uchar *src_gpu_data_device = NULL;                  //������GPU�м����ԭʼͼ��ָ��
	uchar *out_gpu_data_host = gpu_out.ptr<uchar>(0);   //ָ����������ͼ��
	uchar *out_gpu_data_device = NULL;                  //��GPU��������������ָ��

	cudaMalloc((void**)&src_gpu_data_device, sizeof(unsigned char) * rows * cols * channels);  //������ָ���ڷ����Դ�
	cudaMalloc((void**)&out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols * channels); //�����ָ������Դ�

	//�����ݴ���GPU��
	cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);

	//���̣߳�������outͼ�����ص�һ������߳�
	const int nthreads = 256;
	dim3 bdim(nthreads, 1);
	dim3 gdim(divUp(out_cols, bdim.x), divUp(out_rows, bdim.y));

	//��ʱ����
	LARGE_INTEGER gpu_t1, gpu_t2, gpu_tc;
	QueryPerformanceFrequency(&gpu_tc);
	QueryPerformanceCounter(&gpu_t1);
	for (int time = 0; time < 100; time++)
	{
		BI_kernel<nthreads> << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
		cudaDeviceSynchronize();
	}
	QueryPerformanceCounter(&gpu_t2);
	cout << "ʹ��GPU�����ڽ���ֵ����ʱ�䣺" << (gpu_t2.QuadPart - gpu_t1.QuadPart) * 1.0 * 1000 / gpu_tc.QuadPart / 100 << "ms" << endl;

	//����������ڴ�
	cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);
	cv::imshow("GPU�����ͼ��", gpu_out);//GPU�Ľ���������
*/
	

/*-----------------------GPUͼ����1�̴߳������4�����ص㣩--------------*/
	Mat gpu_out(out_rows, out_cols, CV_8UC3);
	uchar *src_gpu_data_host = src.ptr<uchar>(0);       //ָ����������ԭʼͼ��
	uchar *src_gpu_data_device = NULL;                         //������GPU�м����ԭʼͼ��ָ��
	uchar *out_gpu_data_host = gpu_out.ptr<uchar>(0);   //ָ����������ͼ��
	uchar *out_gpu_data_device = NULL;                         //��GPU��������������ָ��

	cudaMalloc((void**)&src_gpu_data_device, sizeof(unsigned char) * rows * cols * channels);  //������ָ���ڷ����Դ�
	cudaMalloc((void**)&out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols * channels); //�����ָ������Դ�

	//�����ݴ���GPU��
	cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);

	//���̣߳��������ͼ��1/4���߳�
	dim3 bdim, gdim;
	bdim.x = DEF_BLOCK_X;
	bdim.y = DEF_BLOCK_Y;
	gdim.x = (out_cols + bdim.x * 4 - 1) / (bdim.x * 4);
	gdim.y = (out_rows + bdim.y - 1) / bdim.y;

	//��ʱ����
	LARGE_INTEGER gpu_t1, gpu_t2, gpu_tc;
	QueryPerformanceFrequency(&gpu_tc);
	QueryPerformanceCounter(&gpu_t1);
	for (int time = 0; time < 100; time++)
	{
		BI2_kernel << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
		cudaDeviceSynchronize();
	}
	QueryPerformanceCounter(&gpu_t2);
	cout << "ʹ��GPU�����ڽ���ֵ����ʱ��(1�̴߳������4�����ص�)��" << (gpu_t2.QuadPart - gpu_t1.QuadPart) * 1.0 * 1000 / gpu_tc.QuadPart / 100 << "ms" << endl;

	//����������ڴ�
	cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);
	cv::imshow("GPU�����ͼ��", gpu_out);//GPU�Ľ���������





	cv::waitKey(0);
	return 0;
}