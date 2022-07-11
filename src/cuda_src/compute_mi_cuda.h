#ifndef _COMPUTEMICUDA_H
#define _COMPUTEMICUDA_H

#include "cuda_image.h"


float  ComputeMICuda(CUDAIMAGE::Pointer fimg, CUDAIMAGE::Pointer mimg,int bs,std::vector<float> lim_arr);

void ComputeMICuda_cuda(cudaPitchedPtr fimg, cudaPitchedPtr mimg,
                     int3 sz);
                     

#endif
