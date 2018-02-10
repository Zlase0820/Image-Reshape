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

// �궨�� Block�ĳߴ�Ϊ 16*2
//#define DEF_BLOCK_X  16
//#define DEF_BLOCK_Y  2


// src_cpu_dataԭͼ��ָ�룻out_cpu_data�����ͼ��ָ�룻scale���䱶��
int NearestInterpolation(uchar* &src_cpu_data, uchar* &out_cpu_data, float scale,int rows,int cols,int channels,int out_rows,int out_cols)
{
	float f_src_row;
	int   i_src_row;

	float f_src_col;    //ԭͼ������float��
	int   i_src_col;    //ԭͼ������int��

	for (int y = 0; y < out_rows; y++)   
	{
		for (int x = 0; x < out_cols ; x++)
		{
			int a = 0, b = 0;
			f_src_row = y / scale;
			i_src_row = (int)f_src_row;
			
			f_src_col = x / scale;
			i_src_col = (int)f_src_col;			
			
			if ((f_src_row - i_src_row) >= 0.5 && i_src_row <(rows - 1) )    //i_out_row <(rows - 1)ֻ��Ϊ�˷�ֹ�����߿�
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

//-----------�°汾----------
//�ɰ汾ÿһ���߳���Ҫ��Ҫ�Ƚ�4�Σ��°汾ÿ���߳�ֻ��Ҫ�Ƚ�2�Ρ�
//�ɰ汾��ֵ����̫�������Ѽ�Ϊ���С�
//�ɰ汾ʹ����GpuMat�����汾ʹ��ָ����д��룬�����˺ںе���

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

//	Mat src = cv::imread(src_path, CV_LOAD_IMAGE_COLOR);   //srcΪԭͼ
	Mat src = cv::imread(src_path, 0);   //srcΪԭͼ

	cv::imshow("ԭʼͼ��", src);

	int rows = src.rows;              //ԭʼͼ��ĸ߶�rows
	int cols = src.cols;              //ԭʼͼ��Ŀ��cols
	int channels = src.channels();    //ԭʼͼ���ͨ����channels
	int out_rows = src.rows*scale;    //�任��ͼ��߶�rows
	int out_cols = src.cols*scale;    //�任��ͼ����cols

/*-------------------------------CPUͼ����-----------------------------*/
	Mat out(out_rows, out_cols, CV_8UC1);  //Ҫ�����ͼ��

	uchar *src_cpu_data = src.ptr<uchar>(0);   //ָ����src��һ�е�һ��Ԫ��
	uchar *out_cpu_data = out.ptr<uchar>(0);   //ָ����out��һ�е�һ��Ԫ��
	
	LARGE_INTEGER cpu_t1, cpu_t2, cpu_tc;
	QueryPerformanceFrequency(&cpu_tc);
	QueryPerformanceCounter(&cpu_t1);
	
	for (int time = 0; time < 100; time++)         //����100�Σ�ȡƽ��ֵ
	{
//		NearestInterpolation(src_cpu_data, out_cpu_data, scale,rows,cols,channels,out_rows,out_cols);
	}
	
	QueryPerformanceCounter(&cpu_t2);
	std::cout << "ʹ��CPU�����ڽ���ֵ����ʱ�䣺" << (cpu_t2.QuadPart - cpu_t1.QuadPart) * 1.0 * 1000 / cpu_tc.QuadPart /100 << "ms" << endl;



/*----------------------GPUͼ����-------------------------*/
	Mat gpu_out(out_rows, out_cols, CV_8UC3);
	uchar *src_gpu_data = src.ptr<uchar>(0);       //ָ����������ԭʼͼ��
	uchar *out_gpu_data = gpu_out.ptr<uchar>(0);   //ָ�������ɵ�GPU��

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
	cout << "ʹ��GPU�����ڽ���ֵ����ʱ�䣺" << (gpu_t2.QuadPart - gpu_t1.QuadPart) * 1.0 * 1000 / gpu_tc.QuadPart / 100 << "ms" << endl;




	cv::imshow("GPU�����ͼ��", gpu_out);//GPU�Ľ���������
	cv::imshow("CPU�����ͼ��", out);

	cv::waitKey(0);
	return 0;
}