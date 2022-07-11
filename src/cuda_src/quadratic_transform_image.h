#ifndef _QUADRATICTRANSFORMIMAGE_H
#define _QUADRATICTRANSFORMIMAGE_H

#include "cuda_image.h"


#include "itkOkanQuadraticTransform.h"
using  TransformType=itk::OkanQuadraticTransform<double,3,3>;

CUDAIMAGE::Pointer QuadraticTransformImageC(CUDAIMAGE::Pointer img, TransformType::Pointer tr, CUDAIMAGE::Pointer target_img);

void QuadraticTransformImage_cuda(cudaTextureObject_t tex,
                                  int3 img_sz,float3 img_res, float3 img_orig, float *img_dir,
                                  int3 target_sz,float3 target_res, float3 target_orig, float *target_dir,
                                  float mat_arr[],
                                  float params_arr[],
                                  cudaPitchedPtr output );

#endif
