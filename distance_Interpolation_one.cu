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

// �궨�� Block�ĳߴ�Ϊ 32*8
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y  8

// �궨�� �� zla��ȡֵ0.5��1     
#define rou  1.0




float dis(float x1, float y1, float x, float y)
{
	return pow(pow(x1 - x, 2) + pow(y1 - y, 2), 0.5); //����֮���ŷ�Ͼ���
}

float ww(float x)
{
	float wi = pow(2.71828182845905, -x / (rou*rou));  //ÿһ�����wֵ
	return wi;
}

float DDis(float wi, float di,float W)
{
	return pow(wi / W*di*di, 0.5);    //���ŷʽ����Di
}


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
//				out_cpu_data[channels * x + 1 + y*out_cols*channels] = src_cpu_data[i_src_col * channels + 1 + i_src_row * cols * channels];
//				out_cpu_data[channels * x + 2 + y*out_cols*channels] = src_cpu_data[i_src_col * channels + 2 + i_src_row * cols * channels];;
			}
			else
			{
				float di[9] = { 0 };
				float wi[9] = { 0 };
				float Di[9] = { 0 };
				float W = 0;
				float D = 0;
				int DN = 0;

				di[0] = dis(i_src_col - 1, i_src_row - 1, f_src_col, f_src_row);  //���� ����
				di[1] = dis(i_src_col, i_src_row - 1, f_src_col, f_src_row);      //��
				di[2] = dis(i_src_col + 1, i_src_row - 1, f_src_col, f_src_row);  //����
				di[3] = dis(i_src_col - 1, i_src_row, f_src_col, f_src_row);      //��
				di[4] = dis(i_src_col, i_src_row, f_src_col, f_src_row);          //��
				di[5] = dis(i_src_col + 1, i_src_row, f_src_col, f_src_row);      //��
				di[6] = dis(i_src_col - 1, i_src_row + 1, f_src_col, f_src_row);  //����
				di[7] = dis(i_src_col, i_src_row + 1, f_src_col, f_src_row);      //��
				di[8] = dis(i_src_col + 1, i_src_row + 1, f_src_col, f_src_row);  //����

				for (int temp2 = 0; temp2 < 9; temp2++)
				{
					wi[temp2] = ww(di[temp2]);   //9�����wֵ
					W += wi[temp2];              //��W��ֵ
				}

				for (int temp3 = 0; temp3 < 9; temp3++)
				{
					Di[temp3] = DDis(wi[temp3], di[temp3], W);    //���ŷʽ����
					D += Di[temp3];                               //��ŷʽ����֮��
				}

				int ii = 0;
//				for (int ii = 0; ii < 3; ii++)
//				{
					DN = (int)((Di[0] * src_cpu_data[channels * (i_src_col - 1) + ii + (i_src_row - 1) * cols *channels]
						+ Di[1] * src_cpu_data[channels * (i_src_col)+ ii+(i_src_row - 1) * cols *channels]
						+ Di[2] * src_cpu_data[channels * (i_src_col + 1) +ii+ (i_src_row - 1) * cols *channels]
						+ Di[3] * src_cpu_data[channels * (i_src_col - 1) +ii+ (i_src_row)* cols *channels]
						+ Di[4] * src_cpu_data[channels * (i_src_col)+ii+(i_src_row)* cols *channels]
						+ Di[5] * src_cpu_data[channels * (i_src_col + 1) +ii+ (i_src_row)* cols *channels]
						+ Di[6] * src_cpu_data[channels * (i_src_col + 1) +ii+ (i_src_row + 1) * cols *channels]
						+ Di[7] * src_cpu_data[channels * (i_src_col)+ii+(i_src_row + 1) * cols *channels]
						+ Di[8] * src_cpu_data[channels * (i_src_col + 1) +ii+ (i_src_row + 1) * cols *channels]) / D);
					out_cpu_data[channels * x + ii + y*out_cols*channels] = DN;
//				}
			}
		}
	}
	return 0;
}



__device__ float gpu_dis(float x1, float y1, float x, float y)
{
	return powf(powf(x1 - x, 2) + powf(y1 - y, 2), 0.5); //����֮���ŷ�Ͼ���
}

__device__ float gpu_ww(float x)
{
	float wi = powf(2.71828182845905, -x / (rou*rou));  //ÿһ�����wֵ
	return wi;
}

__device__ float gpu_DDis(float wi, float di, float W)
{
	return powf(wi / W*di*di, 0.5);    //���ŷʽ����Di
}


__global__ void GPU_distance(uchar *src_gpu_data, uchar *out_gpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;  //x
	const int y = blockIdx.y * blockDim.y + threadIdx.y;  //y   

	if (x >= out_cols || y >= out_rows)
		return;

	float f_src_col;
	int   i_src_col;

	float f_src_row;
	int   i_src_row;


	f_src_col = x / scale;
	f_src_row = y / scale;
	i_src_col = (int)f_src_col;
	i_src_row = (int)f_src_row;
	if (f_src_col - i_src_col>0.5)   //����ԭʼ��λ��������
	{i_src_col++;}
	if (f_src_row - i_src_row>0.5)
	{i_src_row++;}

	if (x == 0 || y == 0 || x == (out_cols -1) || y == (out_rows-1))   //�����߽籣��
	{
		out_gpu_data[channels * x + y*out_cols*channels] = src_gpu_data[i_src_col * channels + i_src_row * cols * channels];
//		out_gpu_data[channels * x + 1 + y*out_cols*channels] = src_gpu_data[i_src_col * channels + 1 + i_src_row * cols * channels];
//		out_gpu_data[channels * x + 2 + y*out_cols*channels] = src_gpu_data[i_src_col * channels + 2 + i_src_row * cols * channels];;
	}
	else
	{
		float di[9] = { 0,0,0,0,0,0,0,0,0 };
		float wi[9] = { 0,0,0,0,0,0,0,0,0 };
		float Di[9] = { 0,0,0,0,0,0,0,0,0 };
		float W = 0;
		float D = 0;
		int DN = 0;

		di[0] = gpu_dis(i_src_col - 1, i_src_row - 1, f_src_col, f_src_row);  //���Ͼ���
		di[1] = gpu_dis(i_src_col, i_src_row - 1, f_src_col, f_src_row);      //�Ͼ���
		di[2] = gpu_dis(i_src_col + 1, i_src_row - 1, f_src_col, f_src_row);  //���Ͼ���
		di[3] = gpu_dis(i_src_col - 1, i_src_row, f_src_col, f_src_row);      //�����
		di[4] = gpu_dis(i_src_col, i_src_row, f_src_col, f_src_row);          //�о���
		di[5] = gpu_dis(i_src_col + 1, i_src_row, f_src_col, f_src_row);      //��
		di[6] = gpu_dis(i_src_col - 1, i_src_row + 1, f_src_col, f_src_row);  //����
		di[7] = gpu_dis(i_src_col, i_src_row + 1, f_src_col, f_src_row);      //��
		di[8] = gpu_dis(i_src_col + 1, i_src_row + 1, f_src_col, f_src_row);  //����

		for (int temp2 = 0; temp2 < 9; temp2++)
		{
			wi[temp2] = gpu_ww(di[temp2]);   //9�����wֵ
			W += wi[temp2];                  //��W��ֵ
		}

		for (int temp3 = 0; temp3 < 9; temp3++)
		{
			Di[temp3] = gpu_DDis(wi[temp3], di[temp3], W);  //���ŷʽ����
			D += Di[temp3];                                 //��ŷʽ����֮��
		}

		int ii = 0;
//		for (int ii = 0; ii < 3; ii++)
//		{
			DN = (int)((Di[0] * src_gpu_data[channels * (i_src_col - 1) + ii + (i_src_row - 1) * cols *channels]
				+ Di[1] * src_gpu_data[channels * (i_src_col)+ii + (i_src_row - 1) * cols *channels]
				+ Di[2] * src_gpu_data[channels * (i_src_col + 1) + ii + (i_src_row - 1) * cols *channels]
				+ Di[3] * src_gpu_data[channels * (i_src_col - 1) + ii + (i_src_row)* cols *channels]
				+ Di[4] * src_gpu_data[channels * (i_src_col)+ii + (i_src_row)* cols *channels]
				+ Di[5] * src_gpu_data[channels * (i_src_col + 1) + ii + (i_src_row)* cols *channels]
				+ Di[6] * src_gpu_data[channels * (i_src_col + 1) + ii + (i_src_row + 1) * cols *channels]
				+ Di[7] * src_gpu_data[channels * (i_src_col)+ii + (i_src_row + 1) * cols *channels]
				+ Di[8] * src_gpu_data[channels * (i_src_col + 1) + ii + (i_src_row + 1) * cols *channels]) / D);
			out_gpu_data[channels * x + ii + y*out_cols*channels] = DN;
//		}
	}
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
		src = cv::imread(src_path,0);   //srcΪԭͼ
	}
	cv::imshow("ԭʼͼ��", src);
	printf("img_width = %d, scale = %lf\n", src.rows, scale);

	int rows = src.rows;              //ԭʼͼ��ĸ߶�rows
	int cols = src.cols;              //ԭʼͼ��Ŀ��cols
	int channels = src.channels();    //ԭʼͼ���ͨ����channels
	int out_rows = src.rows*scale;    //�任��ͼ��߶�rows
	int out_cols = src.cols*scale;    //�任��ͼ����cols


/*-------------------------------CPUͼ����-----------------------------*/
	Mat out(out_rows, out_cols, CV_8UC1);       //Ҫ�����ͼ��

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
//	imwrite("distance_interpolation_0.5.jpg", out);    //����CPU�����Ľ��


/*----------------------GPUͼ����(1�̴߳���1�����ص�)-------------------------*/
	Mat gpu_out(out_rows, out_cols, CV_8UC1);
	uchar *src_gpu_data_host = src.ptr<uchar>(0);       //ָ����������ԭʼͼ��
	uchar *src_gpu_data_device = NULL;                  //������GPU�м����ԭʼͼ��ָ��
	uchar *out_gpu_data_host = gpu_out.ptr<uchar>(0);   //ָ����������ͼ��
	uchar *out_gpu_data_device = NULL;                  //��GPU��������������ָ��

	cudaMalloc((void**)&src_gpu_data_device, sizeof(unsigned char) * rows * cols * channels);         //������ָ���ڷ����Դ�
	cudaMalloc((void**)&out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols * channels); //�����ָ������Դ�

	//���̣߳��߳�����������ص���΢��һЩ��
	dim3 bdim(DEF_BLOCK_X, DEF_BLOCK_Y);
	dim3 gdim;
	gdim.x = (out_cols + bdim.x - 1) / bdim.x;
	gdim.y = (out_rows + bdim.y - 1) / bdim.y;

	//�����ݴ�����GPU��
	cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
	GPU_distance << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
	//�����ݴ��Դ��д��ͻ��ڴ�
	cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);  //��ʼ��ʱ
	for (int time = 0; time < loop; time++)
	{
		//�����ݴ�����GPU��
		cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
		GPU_distance << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
		//�����ݴ��Դ��д��ͻ��ڴ�
		cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "GPU time=" << elapsedTime / loop << endl;   //���GPU���е�ʱ�� 

	cv::imshow("GPU�����ͼ��", gpu_out);    //GPU�Ľ���������
//	imwrite("distance.jpg", gpu_out);    //����GPU����Ľ��

	arr_diff(out_cpu_data, out_gpu_data_host, out_cols*out_rows*channels);



	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cv::waitKey(0);
	 
	return 0;
}
