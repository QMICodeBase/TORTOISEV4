#ifndef _COMPUTEMETRICMSJAC_H
#define _COMPUTEMETRICMSJAC_H

#include "cuda_image.h"

#include "itkGaussianOperator.h"


float ComputeMetric_CCJacS(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img, const CUDAIMAGE::Pointer str_img,
                          const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV   ,
                          CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                          float3 phase_vector,itk::GaussianOperator<float,3> &oper);


void ComputeMetric_CCJacS_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img, cudaPitchedPtr str_img,
     int3 data_sz, float3 data_spc,
     float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
     cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV,
     cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
     const float3 phase_vector,
     int kernel_sz, float *h_kernel, float &metric_value		   );




float ComputeMetric_MSJac(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img, 
                          const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV   ,
                          CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                          float3 phase_vector,itk::GaussianOperator<float,3> &oper);



void ComputeMetric_MSJac_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img,
		   int3 data_sz, float3 data_spc, 
		   float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
		   cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV,
   		   cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                   float3 phase_vector,int kernel_sz, float* h_kernel, float &metric_value
		   );




float ComputeMetric_CCSK(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img, const CUDAIMAGE::Pointer str_img,
                         CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM
                         );
void ComputeMetric_CCSK_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img, cudaPitchedPtr str_img,
                   int3 data_sz, float3 data_spc,
                   float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                   cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                   float &metric_value
                   );





float ComputeMetric_CC(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                         CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM
                         );
void ComputeMetric_CC_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                   int3 data_sz, float3 data_spc,
                   float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                   cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                   float &metric_value
                   );






#endif
