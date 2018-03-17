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

// 宏定义 Block的尺寸为 32*8
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y  8

// 宏定义 肉 zla可取值0.5，1     
#define rou  1.0




float dis(float x1, float y1, float x, float y)
{
	return pow(pow(x1 - x, 2) + pow(y1 - y, 2), 0.5); //两点之间的欧氏距离
}

float ww(float x)
{
	float wi = pow(2.71828182845905, -x / (rou*rou));  //每一个点的w值
	return wi;
}

float DDis(float wi, float di,float W)
{
	return pow(wi / W*di*di, 0.5);    //求大欧式距离Di
}


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

				di[0] = dis(i_src_col - 1, i_src_row - 1, f_src_col, f_src_row);  //左上 距离
				di[1] = dis(i_src_col, i_src_row - 1, f_src_col, f_src_row);      //上
				di[2] = dis(i_src_col + 1, i_src_row - 1, f_src_col, f_src_row);  //右上
				di[3] = dis(i_src_col - 1, i_src_row, f_src_col, f_src_row);      //左
				di[4] = dis(i_src_col, i_src_row, f_src_col, f_src_row);          //中
				di[5] = dis(i_src_col + 1, i_src_row, f_src_col, f_src_row);      //右
				di[6] = dis(i_src_col - 1, i_src_row + 1, f_src_col, f_src_row);  //左下
				di[7] = dis(i_src_col, i_src_row + 1, f_src_col, f_src_row);      //下
				di[8] = dis(i_src_col + 1, i_src_row + 1, f_src_col, f_src_row);  //右下

				for (int temp2 = 0; temp2 < 9; temp2++)
				{
					wi[temp2] = ww(di[temp2]);   //9个点的w值
					W += wi[temp2];              //求W的值
				}

				for (int temp3 = 0; temp3 < 9; temp3++)
				{
					Di[temp3] = DDis(wi[temp3], di[temp3], W);    //求大欧式距离
					D += Di[temp3];                               //大欧式距离之和
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
	return powf(powf(x1 - x, 2) + powf(y1 - y, 2), 0.5); //两点之间的欧氏距离
}

__device__ float gpu_ww(float x)
{
	float wi = powf(2.71828182845905, -x / (rou*rou));  //每一个点的w值
	return wi;
}

__device__ float gpu_DDis(float wi, float di, float W)
{
	return powf(wi / W*di*di, 0.5);    //求大欧式距离Di
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
	if (f_src_col - i_src_col>0.5)   //根据原始点位四舍五入
	{i_src_col++;}
	if (f_src_row - i_src_row>0.5)
	{i_src_row++;}

	if (x == 0 || y == 0 || x == (out_cols -1) || y == (out_rows-1))   //最外侧边界保护
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

		di[0] = gpu_dis(i_src_col - 1, i_src_row - 1, f_src_col, f_src_row);  //左上距离
		di[1] = gpu_dis(i_src_col, i_src_row - 1, f_src_col, f_src_row);      //上距离
		di[2] = gpu_dis(i_src_col + 1, i_src_row - 1, f_src_col, f_src_row);  //右上距离
		di[3] = gpu_dis(i_src_col - 1, i_src_row, f_src_col, f_src_row);      //左距离
		di[4] = gpu_dis(i_src_col, i_src_row, f_src_col, f_src_row);          //中距离
		di[5] = gpu_dis(i_src_col + 1, i_src_row, f_src_col, f_src_row);      //右
		di[6] = gpu_dis(i_src_col - 1, i_src_row + 1, f_src_col, f_src_row);  //左下
		di[7] = gpu_dis(i_src_col, i_src_row + 1, f_src_col, f_src_row);      //下
		di[8] = gpu_dis(i_src_col + 1, i_src_row + 1, f_src_col, f_src_row);  //右下

		for (int temp2 = 0; temp2 < 9; temp2++)
		{
			wi[temp2] = gpu_ww(di[temp2]);   //9个点的w值
			W += wi[temp2];                  //求W的值
		}

		for (int temp3 = 0; temp3 < 9; temp3++)
		{
			Di[temp3] = gpu_DDis(wi[temp3], di[temp3], W);  //求大欧式距离
			D += Di[temp3];                                 //大欧式距离之和
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
		src = cv::imread(src_path,0);   //src为原图
	}
	cv::imshow("原始图像", src);
	printf("img_width = %d, scale = %lf\n", src.rows, scale);

	int rows = src.rows;              //原始图像的高度rows
	int cols = src.cols;              //原始图像的宽度cols
	int channels = src.channels();    //原始图像的通道数channels
	int out_rows = src.rows*scale;    //变换后图像高度rows
	int out_cols = src.cols*scale;    //变换后图像宽度cols


/*-------------------------------CPU图像处理-----------------------------*/
	Mat out(out_rows, out_cols, CV_8UC1);       //要输出的图像

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
//	imwrite("distance_interpolation_0.5.jpg", out);    //保存CPU处理后的结果


/*----------------------GPU图像处理(1线程处理1个像素点)-------------------------*/
	Mat gpu_out(out_rows, out_cols, CV_8UC1);
	uchar *src_gpu_data_host = src.ptr<uchar>(0);       //指向读入进来的原始图像
	uchar *src_gpu_data_device = NULL;                  //用于在GPU中计算的原始图像指针
	uchar *out_gpu_data_host = gpu_out.ptr<uchar>(0);   //指向最后输出的图像
	uchar *out_gpu_data_device = NULL;                  //在GPU中用于输出计算的指针

	cudaMalloc((void**)&src_gpu_data_device, sizeof(unsigned char) * rows * cols * channels);         //给输入指针内分配显存
	cudaMalloc((void**)&out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols * channels); //给输出指针分配显存

	//开线程（线程数量会比像素点稍微多一些）
	dim3 bdim(DEF_BLOCK_X, DEF_BLOCK_Y);
	dim3 gdim;
	gdim.x = (out_cols + bdim.x - 1) / bdim.x;
	gdim.y = (out_rows + bdim.y - 1) / bdim.y;

	//将数据传送至GPU中
	cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
	GPU_distance << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
	//将数据从显存中传送回内存
	cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);  //开始计时
	for (int time = 0; time < loop; time++)
	{
		//将数据传送至GPU中
		cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
		GPU_distance << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
		//将数据从显存中传送回内存
		cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "GPU time=" << elapsedTime / loop << endl;   //输出GPU运行的时间 

	cv::imshow("GPU处理后图像", gpu_out);    //GPU的结果进行输出
//	imwrite("distance.jpg", gpu_out);    //保存GPU处理的结果

	arr_diff(out_cpu_data, out_gpu_data_host, out_cols*out_rows*channels);



	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cv::waitKey(0);
	 
	return 0;
}
