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


// in为原始图像； out为缩放后空白图； scale为缩放比例
int NearestInterpolation(Mat &in, Mat &out, float scale)
{
	float f_in_row;
	float in_row;

	float f_in_col;  //原图横坐标float型
	float in_col;    //原图横坐标int型

	for (int r = 0; r < out.rows; r++)   //height
	{
		for (int c = 0; c < out.cols; c++)  //width
		{
			f_in_row = r / scale;
			in_row = (int)f_in_row;
			if ((f_in_row - in_row) > 0.5 && in_row < (in.rows - 1))    //in.rows - 1 只是为了防止超出边框
				in_row++;

			f_in_col = c / scale;
			in_col = (int)f_in_col;
			if ((f_in_col - in_col) > 0.5 && in_col < (in.cols - 1))
				in_col++;

			out.at<Vec3b>(r, c) = in.at<Vec3b>(in_row, in_col);
		}
	}
	return 0;
}


template <int nthreads>
__global__ void NI_kernel(int out_height, int out_width, const PtrStepb src, PtrStepb out, float scale)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;  //x
	const int y = blockIdx.y * blockDim.y + threadIdx.y;  //y

	const uchar* src_y = (const uchar*)(src);
	uchar* out_y = (uchar*)(out);

	float src_x_f;
	int   src_x_i;
	float src_x_cha;

	float src_y_f;
	int   src_y_i;
	float src_y_cha;

	int a = 0;
	int b = 0;


	src_x_f = x / scale;
	src_y_f = y / scale;
	src_x_i = (int)src_x_f;
	src_y_i = (int)src_y_f;
	src_x_cha = src_x_f - src_x_i;
	src_y_cha = src_y_f - src_y_i;



	/*------------第一个版本-----------
	if (x < out_width && y < out_height)   //减少判断次数，三行赋值合并成一行；  一个线程处理4个像素点
	{
	if (src_x_cha<0.5 && src_y_cha<0.5)
	{
	out_y[3 * x + y*out.step] = src_y[3 * src_x_i + src_y_i*src.step];
	out_y[3 * x + y*out.step + 1] = src_y[3 * src_x_i + src_y_i*src.step + 1];
	out_y[3 * x + y*out.step + 2] = src_y[3 * src_x_i + src_y_i*src.step + 2];
	}
	if (src_x_cha>=0.5 && src_y_cha>=0.5)
	{
	out_y[3 * x + y*out.step] = src_y[3 * src_x_i + 3 + (src_y_i + 1)*src.step];
	out_y[3 * x + y*out.step + 1] = src_y[3 * src_x_i + (src_y_i + 1)*src.step + 1];
	out_y[3 * x + y*out.step + 2] = src_y[3 * src_x_i + (src_y_i + 1)*src.step + 2];
	}
	if (src_x_cha >= 0.5 && src_y_cha < 0.5)
	{
	out_y[3 * x + y*out.step] = src_y[3 * src_x_i + 3 + src_y_i*src.step];
	out_y[3 * x + y*out.step + 1] = src_y[3 * src_x_i + 3 + src_y_i*src.step + 1];
	out_y[3 * x + y*out.step + 2] = src_y[3 * src_x_i + 3 + src_y_i*src.step + 2];
	}
	if (src_x_cha < 0.5 && src_y_cha >= 0.5)
	{
	out_y[3 * x + y*out.step] = src_y[3 * src_x_i + (src_y_i+1)*src.step];
	out_y[3 * x + y*out.step + 1] = src_y[3 * src_x_i + (src_y_i + 1)*src.step + 1];
	out_y[3 * x + y*out.step + 2] = src_y[3 * src_x_i + (src_y_i + 1)*src.step + 2];
	}
	}
	*/

	//-----------第二个版本----------
	//旧版本每一个线程需要需要比较4次，新版本每个线程只需要比较2次。
	//旧版本赋值过程太繁琐，已简化为三行。

	if (x < out_width && y < out_height)
	{
		if (src_x_cha >= 0.5)  a++;
		if (src_y_cha >= 0.5)  b++;
		out_y[3 * x + y*out.step] = src_y[3 * src_x_i + 3 * a + (src_y_i + b) * src.step];
		out_y[3 * x + y*out.step + 1] = src_y[3 * src_x_i + 3 * a + (src_y_i + b)*src.step + 1];
		out_y[3 * x + y*out.step + 2] = src_y[3 * src_x_i + 3 * a + (src_y_i + b)*src.step + 2];
	}




}



int main()
{
	float scale = 0.6f;
	int i;

	char* src_path = "teddy.bmp";

	// initialize the data
	Mat src = cv::imread(src_path, CV_LOAD_IMAGE_COLOR);   //src为原图
	GpuMat d_src(src);
	imshow("原始图像", src);


	int out_rows = src.rows*scale;    //变换后图像高度rows(height)
	int out_cols = src.cols*scale;    //变换后图像宽度cols(width)

	Mat out(out_rows, out_cols, CV_8UC3);
	GpuMat d_out(out.size(), CV_8UC3);

	const int nthreads = 256;
	dim3 bdim(nthreads, 1);
	dim3 gdim(divUp(out.cols, bdim.x), divUp(out.rows, bdim.y));



	/*------------------------GPU图像处理-------------------------------*/
	LARGE_INTEGER gpu_t1, gpu_t2, gpu_tc;
	QueryPerformanceFrequency(&gpu_tc);
	QueryPerformanceCounter(&gpu_t1);

	for (i = 0; i < 100; i++)
	{
		NI_kernel<nthreads> << <gdim, bdim >> > (out_rows, out_cols, d_src, d_out, scale);
		cudaDeviceSynchronize();
	}

	QueryPerformanceCounter(&gpu_t2);
	cout << "使用GPU做最邻近插值法的时间：" << (gpu_t2.QuadPart - gpu_t1.QuadPart) * 1.0 * 1000 / gpu_tc.QuadPart / 100 << "ms" << endl;


	/*------------------------CPU图像处理--------------------------------*/
	LARGE_INTEGER cpu_t1, cpu_t2, cpu_tc;
	QueryPerformanceFrequency(&cpu_tc);
	QueryPerformanceCounter(&cpu_t1);

	for (i = 0; i < 100; i++)
	{
		NearestInterpolation(src, out, scale);
	}

	QueryPerformanceCounter(&cpu_t2);
	cout << "使用CPU做最邻近插值法的时间：" << (cpu_t2.QuadPart - cpu_t1.QuadPart) * 1.0 * 1000 / cpu_tc.QuadPart / 100 << "ms" << endl;



	Mat image(d_out);
	imshow("GPU处理后图像", image);//GPU的结果进行输出
	imshow("CPU处理后图像", out);

	waitKey(0);
	return 0;
}