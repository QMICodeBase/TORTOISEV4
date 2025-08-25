#ifndef _DRTAMAS_UTILITIES_h
#define _DRTAMAS_UTILITIES_h


#include "cuda_image.h"

#include "itkEuler3DTransform.h"
using  TransformType=itk::Euler3DTransform<double>;

void  ComputeTRMapC_cuda(cudaPitchedPtr tensor_img, cudaPitchedPtr output, const int3 data_sz);
CUDAIMAGE::Pointer ComputeTRMapC(CUDAIMAGE::Pointer tensor_img);

void  ComputeFAMapC_cuda(cudaPitchedPtr tensor_img, cudaPitchedPtr output, const int3 data_sz);
CUDAIMAGE::Pointer ComputeFAMapC(CUDAIMAGE::Pointer tensor_img);


void  LogTensor_cuda(cudaPitchedPtr tens, cudaPitchedPtr output, const int3 data_sz);
CUDAIMAGE::Pointer  LogTensor(CUDAIMAGE::Pointer tens);

void  ExpTensor_cuda(cudaPitchedPtr tens, cudaPitchedPtr output, const int3 data_sz);
CUDAIMAGE::Pointer  ExpTensor(CUDAIMAGE::Pointer tens);

void  RotateTensors_cuda(cudaPitchedPtr tens, cudaPitchedPtr output, const int3 data_sz,  float mat_arr[]);
CUDAIMAGE::Pointer  RotateTensors(CUDAIMAGE::Pointer tens,TransformType::Pointer rigid_trans);


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
