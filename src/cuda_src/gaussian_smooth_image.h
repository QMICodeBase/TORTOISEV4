#ifndef _GAUSSIANSMOOTHIMAGE_H
#define _GAUSSIANSMOOTHIMAGE_H

#include "cuda_image.h"





 CUDAIMAGE::Pointer GaussianSmoothImage(CUDAIMAGE::Pointer main_image, float std);

 void GaussianSmoothImage_cuda(cudaPitchedPtr data,
                     int3 data_sz,
                     int Ncomponents,
                     int kernel_sz,
                     float *h_kernel,
                     cudaPitchedPtr output );

 void AdjustFieldBoundary(cudaPitchedPtr orig_img,cudaPitchedPtr smooth_img,int3 data_sz, float weight1, float weight2);

#endif
