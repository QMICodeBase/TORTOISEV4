#ifndef _RIGIDTRANSFORMIMAGE_H
#define _RIGIDTRANSFORMIMAGE_H

#include "cuda_image.h"


#include "itkEuler3DTransform.h"
using  TransformType=itk::Euler3DTransform<double>;

CUDAIMAGE::Pointer RigidTransformImageC(CUDAIMAGE::Pointer img, TransformType::Pointer tr, CUDAIMAGE::Pointer target_img);

void RigidTransformImage_cuda(cudaTextureObject_t tex,
                                  int3 img_sz,float3 img_res, float3 img_orig, float *img_dir,
                                  int3 target_sz,float3 target_res, float3 target_orig, float *target_dir,
                                  float mat_arr[],
                                  float params_arr[],
                                  cudaPitchedPtr output );

#endif
