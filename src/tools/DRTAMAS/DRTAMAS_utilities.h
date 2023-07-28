#ifndef _DRTAMAS_UTILITIES_h
#define _DRTAMAS_UTILITIES_h


#include "cuda_image.h"



void  ComputeTRMapC_cuda(cudaPitchedPtr tensor_img, cudaPitchedPtr output, const int3 data_sz);
CUDAIMAGE::Pointer ComputeTRMapC(CUDAIMAGE::Pointer tensor_img);


void  LogTensor_cuda(cudaPitchedPtr tens, cudaPitchedPtr output, const int3 data_sz);
CUDAIMAGE::Pointer  LogTensor(CUDAIMAGE::Pointer tens);

void  ExpTensor_cuda(cudaPitchedPtr tens, cudaPitchedPtr output, const int3 data_sz);
CUDAIMAGE::Pointer  ExpTensor(CUDAIMAGE::Pointer tens);


void SplitImageComponents_cuda(cudaPitchedPtr tensor_img,
                          cudaPitchedPtr *output,
                          int3 data_sz,
                          int Ncomp);
std::vector<CUDAIMAGE::Pointer>  SplitImageComponents(CUDAIMAGE::Pointer img);



void CombineImageComponents_cuda(cudaPitchedPtr *img,
                          cudaPitchedPtr output,
                          int3 data_sz,
                          int Ncomp);
CUDAIMAGE::Pointer  CombineImageComponents(std::vector<CUDAIMAGE::Pointer> img);






#endif
