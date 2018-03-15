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

//out1_data��Сͼ��out2_data�Ǵ�ͼ
int Cubic_return(uchar *out1_data, uchar *out2_data, float scale, int rows, int cols, int channels, int out_rows, int out_cols)
{
	float f_src_row=0;    //��
	int   i_src_row=0;

	float f_src_col=0;    //��
	int   i_src_col=0;    

	for (int y = 0; y < out_rows; y++)
	{
		for (int x = 0; x < out_cols; x++)
		{

			f_src_row = y / scale;
			i_src_row = (int)f_src_row;

			f_src_col = x / scale;
			i_src_col = (int)f_src_col;

			if (i_src_row >= rows || i_src_col >= cols)
				return 0;
			else
			{
				for (int i = 0; i < 3; i++)
				{
					out2_data[channels*i_src_col + i + i_src_row*cols*channels] = out1_data[channels*x + i + y*out_cols*channels];
				}
			}
		}
	}
	return 0;
}



int main()
{
	float scale = 0.5f;

	char* src_path = "map.png";
	Mat src = cv::imread(src_path, 1);   //srcΪԭͼ
	cv::imshow("ԭʼͼ��", src);

	char* out_path1 = "distance_6_scale=0.5.jpg";
	Mat out = cv::imread(out_path1, 1);   //Сͼ
	cv::imshow("Cubic�����ͼ��", out);

	int rows = src.rows;              //ԭʼͼ��ĸ߶�rows
	int cols = src.cols;              //ԭʼͼ��Ŀ��cols
	int channels = src.channels();    //ԭʼͼ���ͨ����channels
	int out_rows = out.rows;	       //�任��ͼ��߶�rows
	int out_cols = out.cols;          //�任��ͼ����cols

														
	uchar *out_data = out.ptr<uchar>(0);   //ָ����Cubic�����Ľ��
	uchar *src_data = src.ptr<uchar>(0);    //ָ������������Ľ��

	Cubic_return(out_data, src_data, scale, rows, cols, channels, out_rows, out_cols);

	cv::imshow("return��Ľ��", src);
	imwrite("distance_return_6_scale=0.5.jpg",src);
	cv::waitKey(0);
	return 0;
}
