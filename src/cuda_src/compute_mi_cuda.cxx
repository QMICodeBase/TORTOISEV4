#ifndef _COMPUTEMICUDA_CXX
#define _COMPUTEMICUDA_CXX

#include "compute_mi_cuda.h"
#include "mutual_information_common.h"



bool cudaImageMutualInformation64( float *h_JointEntropy, float *h_Entropy1, float *h_Entropy2,
   			          cudaPitchedPtr img1, cudaPitchedPtr img2,
                                  int3 sz,
                                  int NBins,
                                  float limarr0,float limarr1,float limarr2,float limarr3)
{
    cudaEvent_t start_device, stop_device, start_histogram, stop_histogram, start_entropy, stop_entropy;
  
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);
    cudaEventCreate(&start_histogram);
    cudaEventCreate(&stop_histogram);
    cudaEventCreate(&start_entropy);
    cudaEventCreate(&stop_entropy);
  
    cudaEventRecord(start_device,0);
    
    uint *d_JointHistogram, *d_Histogram1, *d_Histogram2;
    
    checkCudaErrors(cudaMalloc((void **)&d_Histogram1, NBins * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_Histogram2, NBins * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_JointHistogram, NBins*NBins* sizeof(uint)));
    
    
  

}



float  ComputeMICuda(CUDAIMAGE::Pointer fimg, CUDAIMAGE::Pointer mimg, int NBins,std::vector<float> lim_arr)
{

    float h_Entropy1 = 0.0;
    float h_Entropy2 = 0.0;
    float h_JointEntropy = 0.0;
    
    
    h_Histogram1 = (uint *)malloc(HISTOGRAM64_BIN_COUNT * sizeof(uint));
    h_Histogram2 = (uint *)malloc(HISTOGRAM64_BIN_COUNT * sizeof(uint));
    h_JointHistogram = (uint *)malloc(JOINT_HISTOGRAM64_BIN_COUNT * sizeof(uint));
      
     
    if (cudaImageMutualInformation64( &h_JointEntropy, &h_Entropy1, &h_Entropy2,
                                      cudaPitchedPtr img1, cudaPitchedPtr img2,
			              fimg->sz,
			              NBins,
			              lim_arr[0],lim_arr[1],lim_arr[2],lim_arr[3])
	{
	  std::cerr << "cudaImageMutualInformation64 Error\n";
	  return EXIT_FAILURE;
	}


    float MI = ( h_Entropy1 + h_Entropy2 ) / h_JointEntropy; 
    return -MI;
}

//void ComputeMICuda_cuda(cudaPitchedPtr fimg, cudaPitchedPtr mimg,
  //                   int3 sz);
                     

#endif
