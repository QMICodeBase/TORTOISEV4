#ifndef _WGPU_COMPUTEENTROPY_H
#define _WGPU_COMPUTEENTROPY_H
#include "gpu_image.h"
void ComputeJointEntropy(CUDAIMAGE::Pointer img1, float low_lim1, float high_lim1,
                         CUDAIMAGE::Pointer img2, float low_lim2, float high_lim2,
                         int Nbins, float &entropy_j, float &entropy_img1, float &entropy_img2);
#endif
