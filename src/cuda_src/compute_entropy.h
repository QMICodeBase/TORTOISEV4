#ifndef _COMPUTEENTROPY_H
#define _COMPUTEENTROPY_H

#include "cuda_image.h"


float  ComputeEntropy(CUDAIMAGE::Pointer img, int Nbins, float low_lim, float high_lim);

void ComputeEntropy_cuda(cudaPitchedPtr img, const int3 sz, const int Nbins, const float low_lim, const float high_lim, float &value);
                     


void ComputeJointEntropy_cuda(cudaPitchedPtr img1, float low_lim1, float high_lim1, cudaPitchedPtr img2, float low_lim2, float high_lim2, const int3 sz, const int Nbins,  float &value1 , float &value2);
void ComputeJointEntropy(CUDAIMAGE::Pointer img1, float low_lim1, float high_lim1, CUDAIMAGE::Pointer img2, float low_lim2, float high_lim2, int Nbins,float &entropy_j,float &entropy_img2);

#endif
