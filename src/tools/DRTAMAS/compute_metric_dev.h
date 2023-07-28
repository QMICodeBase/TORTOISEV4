#ifndef _COMPUTEMETRICDEV_H
#define _COMPUTEMETRICDEV_H

#include "cuda_image.h"





float ComputeMetric_DEV(const CUDAIMAGE::Pointer fixed_img, const CUDAIMAGE::Pointer moving_img,
                          const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV   ,
                          CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM);




void ComputeMetric_DEV_cuda(cudaPitchedPtr fixed_img, cudaPitchedPtr moving_img,
		   int3 data_sz, float3 data_spc, 
		   float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
		   cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV,
   		   cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                   float &metric_value
);


void ComputeDeviatoricTensor_cuda(cudaPitchedPtr img,   int3 data_sz);

#endif
