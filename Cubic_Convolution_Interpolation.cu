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

const int loop = 1;

// 宏定义 Block的尺寸为 32*8
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y  8

//宏定义锐化值 a=-0.5，-0.75，-1，-2
#define ii -0.5


float Rx(float Rx)
{
	if (fabs(Rx) <= 1)
		return 1 - (ii + 3)*pow(Rx, 2) + (ii + 2)*pow(fabs(Rx), 3);
	else
		if (fabs(Rx) <= 2 && fabs(Rx)>1)
			return -4 * ii + 8 * ii*fabs(Rx) - 5 * ii*pow(Rx, 2) + ii*pow(fabs(Rx), 3);
		else
		{
			cout << "Rx计算中出现错误，输入任意键退出程序" << endl;
			getchar();
			return 0;
		}
}


int CPU_Cubic(uchar *src_cpu_data, uchar *out_cpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	float f_src_row;   //高
	int   i_src_row;
	float cha_src_row;

	float f_src_col;   //宽
	int   i_src_col;
	float cha_src_col;


	for (int y = 0; y < out_rows; y++)
	{
		for (int x = 0; x < out_cols; x++)
		{
			f_src_row = y / scale;
			i_src_row = (int)f_src_row;
			cha_src_row = f_src_row - i_src_row;

			f_src_col = x / scale;
			i_src_col = (int)f_src_col;
			cha_src_col = f_src_col - i_src_col;

			double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
			/*----计算部分----*/
			if (x == 0 || y == 0 || x == (out_cols - 1) || x == (out_cols - 2) || y == (out_rows - 1) || y == (out_rows - 2))
			{
				out_cpu_data[channels * x + y*out_cols*channels] = src_cpu_data[i_src_col * channels + i_src_row * cols * channels];
				out_cpu_data[channels * x + 1 + y*out_cols*channels] = src_cpu_data[i_src_col * channels + 1 + i_src_row * cols * channels];
				out_cpu_data[channels * x + 2 + y*out_cols*channels] = src_cpu_data[i_src_col * channels + 2 + i_src_row * cols * channels];;
			}
			else
			{
				for (int m = -1; m <= 2; m++)
				{
					for (int n = -1; n <= 2; n++)
					{
						sum1 += src_cpu_data[(i_src_col + m) * channels + (i_src_row + n) * cols * channels] * Rx(cha_src_col - m)*Rx(cha_src_row - n);
						sum2 += src_cpu_data[(i_src_col + m) * channels + 1 + (i_src_row + n) * cols * channels] * Rx(cha_src_col - m)*Rx(cha_src_row - n);
						sum3 += src_cpu_data[(i_src_col + m) * channels + 2 + (i_src_row + n) * cols * channels] * Rx(cha_src_col - m)*Rx(cha_src_row - n);
					}
				}
				if (sum1<0)   sum1 = 0;        //R 计算后的值会超出0-255的范围
				if (sum1>255) sum1 = 255;
				if (sum2<0)   sum2 = 0;        //G 计算后的值会超出0-255的范围
				if (sum2>255) sum2 = 255;
				if (sum3<0)   sum3 = 0;        //B 计算后的值会超出0-255的范围
				if (sum3>255) sum3 = 255;

				out_cpu_data[channels * x + y*out_cols*channels] = sum1;
				out_cpu_data[channels * x + 1 + y*out_cols*channels] = sum2;
				out_cpu_data[channels * x + 2 + y*out_cols*channels] = sum3;
			}
		}
	}
	return 0;
}

__device__ float GPU_Rx(float Rx)
{
	if (fabs(Rx) <= 1)
		return 1 - (ii + 3)*pow(Rx, 2) + (ii + 2)*pow(fabs(Rx), 3);
	else
		if (fabs(Rx) <= 2 && fabs(Rx)>1)
			return -4 * ii + 8 * ii*fabs(Rx) - 5 * ii*pow(Rx, 2) + ii*pow(fabs(Rx), 3);
}


__global__ void GPU_Cubic(uchar *src_gpu_data, uchar *out_gpu_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
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
	cha_src_col = f_src_col - i_src_col;
	cha_src_row = f_src_row - i_src_row;

	float sum1 = 0, sum2 = 0, sum3 = 0;
	/*----计算部分----*/
	for (int m = -1; m <= 2; m++)
	{
		for (int n = -1; n <= 2; n++)
		{
			sum1 += src_gpu_data[(i_src_col + m) * channels + (i_src_row + n) * cols * channels] * GPU_Rx(cha_src_col - m)*GPU_Rx(cha_src_row - n);
			sum2 += src_gpu_data[(i_src_col + m) * channels + 1 + (i_src_row + n) * cols * channels] * GPU_Rx(cha_src_col - m)*GPU_Rx(cha_src_row - n);
			sum3 += src_gpu_data[(i_src_col + m) * channels + 2 + (i_src_row + n) * cols * channels] * GPU_Rx(cha_src_col - m)*GPU_Rx(cha_src_row - n);
		}
	}
	if (sum1<0)   sum1 = 0;        //R 计算后的值会超出0-255的范围
	if (sum1>255) sum1 = 255;
	if (sum2<0)   sum2 = 0;        //G 计算后的值会超出0-255的范围
	if (sum2>255) sum2 = 255;
	if (sum3<0)   sum3 = 0;        //B 计算后的值会超出0-255的范围
	if (sum3>255) sum3 = 255;

	out_gpu_data[channels * x + y*out_cols*channels] = sum1;
	out_gpu_data[channels * x + 1 + y*out_cols*channels] = sum2;
	out_gpu_data[channels * x + 2 + y*out_cols*channels] = sum3;

}


void arr_diff(uchar *src, uchar *dst, int size)
{
	 int aa=0;
	 for (int i = 0; i < size; i++) 
	    {
	        if (src[i] != dst[i])
	        {
	            printf("diff, index=%d, %d != %d\n", i, src[i], dst[i]);  
				 aa++; 
	        }
	    }
	 cout<<"错误像素点总数量为："<<aa<<endl;
}




int main(int argc, char **argv)
{
	cudaSetDevice(1);
	float scale = 0.3f;
	char* src_path = "flower.jpg";

	Mat src;
	if (argc > 1)
	{
		int width = atoi(argv[1]);
		scale = atof(argv[2]);
		/*src = Mat(width, width, CV_8UC3)*/
		src = Mat::ones(width, width, CV_8UC3);
	}
	else {
		src = cv::imread(src_path, CV_LOAD_IMAGE_COLOR);   //src为原图
	}
	//cv::imshow("原始图像", src);
	printf("img_width = %d, scale = %lf\n", src.rows, scale);

	int rows = src.rows;              //原始图像的高度rows
	int cols = src.cols;              //原始图像的宽度cols
	int channels = src.channels();    //原始图像的通道数channels
	int out_rows = src.rows*scale;    //变换后图像高度rows
	int out_cols = src.cols*scale;    //变换后图像宽度cols

/*-------------------------------CPU图像处理-----------------------------*/
	Mat out(out_rows, out_cols, CV_8UC3);       //要输出的图像

	uchar *src_cpu_data = src.ptr<uchar>(0);   //指向了src第一行第一个元素
	uchar *out_cpu_data = out.ptr<uchar>(0);   //指向了out第一行第一个元素

	cudaEvent_t start, stop;   //使用CUDA的计时函数
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	clock_t clock_start;
	clock_t clock_end;
	clock_start = clock();

	for (int time = 0; time < loop; time++)         //运行100次，取平均值
		CPU_Cubic(src_cpu_data, out_cpu_data, scale, rows, cols, channels, out_rows, out_cols);

	clock_end = clock();

	float clock_diff_sec = ((clock_end - clock_start)*1.0 / CLOCKS_PER_SEC * 1000.0 /100 );
	printf("CPU Time = %lf\n", clock_diff_sec);

	cv::imshow("CPU处理后图像", out); 
//	imwrite("Cubic_Interpolation.jpg",out);    //保存CPU处理后的结果

/*----------------------GPU图像处理(1线程处理1个像素点)-------------------------*/
	Mat gpu_out(out_rows, out_cols, CV_8UC3);
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
	GPU_Cubic << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
	//将数据从显存中传送回内存
	cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);  //开始计时
	for (int time = 0; time < loop; time++)
	{
		//将数据传送至GPU中
		cudaMemcpy(src_gpu_data_device, src_gpu_data_host, sizeof(unsigned char) * rows * cols*channels, cudaMemcpyHostToDevice);
		GPU_Cubic << <gdim, bdim >> > (src_gpu_data_device, out_gpu_data_device, scale, rows, cols, channels, out_rows, out_cols);
		//将数据从显存中传送回内存
		cudaMemcpy(out_gpu_data_host, out_gpu_data_device, sizeof(unsigned char) * out_rows * out_cols*channels, cudaMemcpyDeviceToHost);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "GPU time=" << elapsedTime / loop << endl;   //输出GPU运行的时间 


	cv::imshow("GPU处理后图像", gpu_out);	//GPU的结果进行输出
	imwrite("Cubic.jpg",gpu_out);		//保存GPU处理的结果

	arr_diff(out_cpu_data, out_gpu_data_host, out_cols*out_rows*channels);


	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cv::waitKey(0);
	
	return 0;
}
