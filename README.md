# Image-Reshape
CUDA/GPU

项目使用的软硬件如下：            
    Win10，GTX 1080;                                      
    CUDA8.0，OpenCV2.4.13，Visual Studio 2015；                                                                            
本项目主要有以下几个部分(每个程序都有CPU和GPU版本）：                        
1.nearest_interpolation.cu                                      
    该部分为原始图像使用最邻近插值对图像进行处理，算法详细内容请看【重采样算法说明.docx】。                                      
2.Bilinear_Interpolation.cu                                      
    该部分为原始图像使用双线性插值进行处理，算法详细内容请看【重采样算法说明.docx】。                                      
3.Cubic_Convolution_Interpolation.cu                                      
    该部分为原始图像使用双立方卷积法进行处理，算法详细内容请看【重采样算法说明.docx】。                                      
4.Cubic_Convolution_Interpolation_range.cu                                      
    该部分为原始图像进行图像重采样时，当一个点在原始图像的8邻域内的极值小于10时，认为这个点与原始点差距很小，采用8邻域取平均值的算法（三通道图像要求该点三个通道的极值都小于10）；如果极值大于10，这个点正常使用双立方卷积法进行处理，算法详细内容请看【重采样算法说明.docx】。               
5.distance_Interpolation.cu                                                                   
    该部分为图像重采样时使用一个距离算法，具体的算法比较复杂，算法详细内容请看【重采样算法说明.docx】。
6.distance_Interpolation_6.cu                          
    该部分在5的基础上，根据一个8邻域的值进行计算，算法详细内容请看【重采样算法说明.docx】。                          
7.return.cu                                  
    附加算法，将处理后的图像对应到原来的像素点上，算法详细内容请看【重采样算法说明.docx】。                           
8.run.bat                           
    windows下的指令，用来测试程序，测试的GPU加速效果存放在data.xlsx中。                           
    
后边出现_one的.cu则是对应该算法的单通道处理算法。
