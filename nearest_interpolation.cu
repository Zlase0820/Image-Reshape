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


// �궨�� Block�ĳߴ�
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y  8


// src_cpu_dataԭͼ��ָ�룻out_cpu_data�����ͼ��ָ�룻scale���䱶��
int NearestInterpolation(uchar *src_cpu_data, uchar *out_cpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	float f_src_row;
	int   i_src_row;

	float f_src_col;    //ԭͼ������float��
	int   i_src_col;    //ԭͼ������int��

	for (int y = 0; y < out_rows; y++)
	{
		for (int x = 0; x < out_cols; x++)
		{
			int a = 0, b = 0;
			f_src_row = y / scale;
			i_src_row = (int)f_src_row;

			f_src_col = x / scale;
			i_src_col = (int)f_src_col;

			if ((f_src_row - i_src_row) >= 0.5 && i_src_row <(rows - 1))    //i_out_row <(rows - 1)ֻ��Ϊ�˷�ֹ�����߿�
				a = 1;

			if ((f_src_col - i_src_col) >= 0.5 && i_src_col < (cols - 1))
				b = 1;

			out_cpu_data[3 * x + y*out_cols*channels] = src_cpu_data[3 * (i_src_col + a) + (i_src_row + b)*cols*channels];
			out_cpu_data[3 * x + y*out_cols*channels + 1] = src_cpu_data[3 * (i_src_col + a) + (i_src_row + b)*cols*channels + 1];
			out_cpu_data[3 * x + y*out_cols*channels + 2] = src_cpu_data[3 * (i_src_col + a) + (i_src_row + b)*cols*channels + 2];

		}
	}
	return 0;
}

__global__ void NI_kernel(uchar *src_gpu_data, uchar *out_gpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;  //x
	const int y = blockIdx.y * blockDim.y + threadIdx.y;  //y    

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

	if (cha_col >= 0.5 && i_src_col < (out_cols - 1))  a++;
	if (cha_row >= 0.5 && i_src_row < (out_rows - 1))  b++;
	uchar *src_ptr = src_gpu_data + channels*(i_src_col + a) + (i_src_row + b)*cols*channels;
	uchar *out_ptr = out_gpu_data + channels*x + y*out_cols*channels;
	*(out_ptr++) = *(src_ptr++);
	*(out_ptr++) = *(src_ptr++);
	*out_ptr = *src_ptr;
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
	if (aa == 0)
	{
		cout << "CPU��GPU�����Ľ����ȫ��ͬ" << endl;
	}
	else
	{
		cout << "��������ص�һ����" << aa << "��"<<endl;
	}
}

const int loop = 10;

int main(int argc, char **argv)
{
	float scale = 0.3f;
	char* src_path = "flower.jpg";
	Mat src;
	if (argc > 1)
	{
		int width = atoi(argv[1]);
		scale = atof(argv[2]);
		src = Mat::ones(width, width, CV_8UC3);
	}
	else {

		src = cv::imread(src_path, CV_LOAD_IMAGE_COLOR);   //srcΪԭͼ
	}
	cv::imshow("ԭʼͼ��", src);
	printf("img_width = %d, scale = %lf\n", src.rows, scale);
	cudaSetDevice(1);

	int rows = src.rows;              //ԭʼͼ��ĸ߶�rows
	int cols = src.cols;              //ԭʼͼ��Ŀ��cols
	int channels = src.channels();    //ԭʼͼ���ͨ����channels
	int out_rows = src.rows*scale;    //�任��ͼ��߶�rows
	int out_cols = src.cols*scale;    //�任��ͼ����cols

/*-------------------------------CPUͼ����-----------------------------*/
	Mat out(out_rows, out_cols, CV_8UC3);  //Ҫ�����ͼ��

	uchar *src_cpu_data = src.ptr<uchar>(0);   //ָ����src��һ�е�һ��Ԫ��
	uchar *out_cpu_data = out.ptr<uchar>(0);   //ָ����out��һ�е�һ��Ԫ��

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	clock_t clock_start;
	clock_t clock_end;
	clock_start = clock();

	for (int time = 0; time < loop; time++)         //����100�Σ�ȡƽ��ֵ
		NearestInterpolation(src_cpu_data, out_cpu_data, scale, rows, cols, channels, out_rows, out_cols);
	clock_end = clock();


	float clock_diff_sec = ((clock_end - clock_start)*1.0 / CLOCKS_PER_SEC * 1000.0 / loop);
	printf("CPU Time = %lf\n", clock_diff_sec);
  

	cv::imshow("CPU�����ͼ��", out);    
//	imwrite("Nearest.jpg", out);



/*----------------------GPUͼ����(1�̴߳���1�����ص�)-------------------------*/
	Mat gpu_out(out_rows, out_cols, CV_8UC3);
	uchar *src_gpu_data_host = src.ptr<uchar>(0);       //ָ����������ԭʼͼ��
	uchar *src_gpu_data_device = NULL;                         //������GPU�м����ԭʼͼ��ָ��
	uchar *out_gpu_data_host = gpu_out.ptr<uchar>(0);   //ָ����������ͼ��
	uchar *out_gpu_data_device = NULL;                         //��GPU��������������ָ��

	cudaMalloc((void**)&src_gpu_data_device, sizeof(unsigned char) * rows * cols * channels);  //������ָ���ڷ����Դ�
	cudaMalloc((void**)&out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols * channels); //�����ָ������Դ�

	//���̣߳�������outͼ�����ص�һ������߳�
	dim3 bdim(DEF_BLOCK_X, DEF_BLOCK_Y);
	dim3 gdim;
	gdim.x = (out_cols + bdim.x - 1) / bdim.x;
	gdim.y = (out_rows + bdim.y - 1) / bdim.y;

	cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
	NI_kernel << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
	//����������ڴ�
	cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);
	cudaEventRecord(start, 0);

	for (int time = 0; time < loop; time++)
	{
		//�����ݴ���GPU��
		cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
		NI_kernel << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
		//����������ڴ�
		cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "GPU time=" << elapsedTime / loop << endl;
	arr_diff(out_cpu_data, out_gpu_data_host, out_cols*out_rows*channels);

	cv::imshow("GPU�����ͼ��", gpu_out);  //GPU�Ľ���������
	cv::imwrite("Nearest.jpg", gpu_out);   //GPU�Ľ���������

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	waitKey(0);
	return 0;
}
