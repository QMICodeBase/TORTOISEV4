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



float ComputeMetric_CCJacSSingle(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img, const CUDAIMAGE::Pointer str_img,
                                 std::vector<CUDAIMAGE::Pointer>  up_grad_img, std::vector<CUDAIMAGE::Pointer>  down_grad_img,
                                 const CUDAIMAGE::Pointer def_FINV,
                                 CUDAIMAGE::Pointer &updateFieldFINV,
                                 float3 phase_vector,itk::GaussianOperator<float,3> &oper);


void ComputeMetric_CCJacSSingle_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img, cudaPitchedPtr str_img,
                                    cudaTextureObject_t up_grad_img_x, cudaTextureObject_t up_grad_img_y, cudaTextureObject_t up_grad_img_z,
                                    cudaTextureObject_t down_grad_img_x, cudaTextureObject_t down_grad_img_y, cudaTextureObject_t down_grad_img_z,
                                    int3 data_sz, float3 data_spc,
                                    float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                                    cudaPitchedPtr def_FINV,
                                    cudaPitchedPtr updateFieldFINV,
                                   float3 phase_vector,
                                     int kernel_sz, float* h_kernel, float &metric_value
);






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




float ComputeMetric_MSJacSingle(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                                std::vector<CUDAIMAGE::Pointer>  up_grad_img, std::vector<CUDAIMAGE::Pointer>  down_grad_img,
                          const CUDAIMAGE::Pointer def_FINV,
                          CUDAIMAGE::Pointer &updateFieldFINV,
                          float3 phase_vector,itk::GaussianOperator<float,3> &oper);


void ComputeMetric_MSJacSingle_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                                    cudaTextureObject_t up_grad_img_x, cudaTextureObject_t up_grad_img_y, cudaTextureObject_t up_grad_img_z,
                                    cudaTextureObject_t down_grad_img_x, cudaTextureObject_t down_grad_img_y, cudaTextureObject_t down_grad_img_z,
     int3 data_sz, float3 data_spc,
     float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
     cudaPitchedPtr def_FINV,
     cudaPitchedPtr updateFieldFINV,
     float3 phase_vector,int kernel_sz, float* h_kernel, float &metric_value
);





float ComputeMetric_CCSK(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img, const CUDAIMAGE::Pointer str_img,
                         CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                         float t=0.5);
void ComputeMetric_CCSK_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img, cudaPitchedPtr str_img,
                   int3 data_sz, float3 data_spc,
                   float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                   cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                   float &metric_value,
                   float t=0.5
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



CUDAIMAGE::Pointer ComputeDetImgMain(CUDAIMAGE::Pointer img, CUDAIMAGE::Pointer field, float3 phase_vector);
void ComputeDetImg_cuda(cudaPitchedPtr img, cudaPitchedPtr field,
                        int3 data_sz, float3 data_spc,
                        float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                        float3 phase_vector,
                        cudaPitchedPtr output);


float SumImage(CUDAIMAGE::Pointer im1);
float SumImage_cuda(cudaPitchedPtr im1, const int3 data_sz,const int ncomp);

#endif
