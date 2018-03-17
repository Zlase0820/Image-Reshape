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

// �궨�� Block�ĳߴ�Ϊ 32*8
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y  8


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
			cha_src_row = f_src_row - i_src_row;   //v

			f_src_col = x / scale;
			i_src_col = (int)f_src_col;
			cha_src_col = f_src_col - i_src_col;   //u

			//�Ӻ��е����зֱ��ʾ���Ͻǵĵ㡢���Ͻǵĵ㡢���½ǵĵ㡢���½ǵĵ�
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

	uchar *out_ptr = out_gpu_data + channels*x + y*out_cols*channels; //���ͼ��������
	uchar *src_ptr1 = src_gpu_data + channels * i_src_col + i_src_row * cols * channels;  //��Ӧ�������Ͻǵ�
	uchar *src_ptr2 = src_gpu_data + channels * (i_src_col + 1) + i_src_row * cols * channels;  //��Ӧ�������Ͻǵ�
	uchar *src_ptr3 = src_gpu_data + channels * i_src_col + (i_src_row + 1) * cols * channels;  //��Ӧ�������½ǵ�
	uchar *src_ptr4 = src_gpu_data + channels * (i_src_col + 1) + (i_src_row + 1) * cols * channels;  //��Ӧ�������½ǵ�

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
	cout << "�������ص�������Ϊ��" << aa << endl;
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
		src = cv::imread(src_path, 0);   //srcΪԭͼ
	}
	// cv::imshow("ԭʼͼ��", src);
	printf("img_width = %d, scale = %lf\n", src.rows, scale);

	int rows = src.rows;              //ԭʼͼ��ĸ߶�rows
	int cols = src.cols;              //ԭʼͼ��Ŀ��cols
	int channels = src.channels();    //ԭʼͼ���ͨ����channels
	int out_rows = src.rows*scale;    //�任��ͼ��߶�rows
	int out_cols = src.cols*scale;    //�任��ͼ����cols

/*-------------------------------CPUͼ����-----------------------------*/
	Mat out(out_rows, out_cols, CV_8UC1);  //Ҫ�����ͼ��

	uchar *src_cpu_data = src.ptr<uchar>(0);   //ָ����src��һ�е�һ��Ԫ��
	uchar *out_cpu_data = out.ptr<uchar>(0);   //ָ����out��һ�е�һ��Ԫ��

	cudaEvent_t start, stop;   //ʹ��CUDA�ļ�ʱ����
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


	cv::imshow("CPU�����ͼ��", out);   //CPU�����Ľ��
//  imwrite("Bilinear_interpolation.jpg",out);    //����CPU�����Ľ��



/*----------------------GPUͼ����(1�̴߳���1�����ص�)-------------------------*/
	Mat gpu_out(out_rows, out_cols, CV_8UC1);
	uchar *src_gpu_data_host = src.ptr<uchar>(0);       //ָ����������ԭʼͼ��
	uchar *src_gpu_data_device = NULL;                  //������GPU�м����ԭʼͼ��ָ��
	uchar *out_gpu_data_host = gpu_out.ptr<uchar>(0);   //ָ����������ͼ��
	uchar *out_gpu_data_device = NULL;                  //��GPU��������������ָ��

	cudaMalloc((void**)&src_gpu_data_device, sizeof(unsigned char) * rows * cols * channels);  //������ָ���ڷ����Դ�
	cudaMalloc((void**)&out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols * channels); //�����ָ������Դ�
																									  //���̣߳��߳�����������ص���΢��һЩ��
	dim3 bdim(DEF_BLOCK_X, DEF_BLOCK_Y);
	dim3 gdim;
	gdim.x = (out_cols + bdim.x - 1) / bdim.x;
	gdim.y = (out_rows + bdim.y - 1) / bdim.y;

	cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
	BI_kernel << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
	cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);

	cudaEventRecord(start, 0);  //��ʱ����

	for (int time = 0; time < loop; time++)
	{
		cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
		BI_kernel << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
		cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "GPU time=" << elapsedTime / loop << endl;   //���GPU���е�ʱ�� 

    cv::imshow("GPU�����ͼ��", gpu_out);//GPU�Ľ���������
//	imwrite("Bilinear.jpg",gpu_out);  //����GPU����Ľ��

	arr_diff(out_cpu_data, out_gpu_data_host, out_cols*out_rows*channels);



	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cv::waitKey(0);
	return 0;
}
