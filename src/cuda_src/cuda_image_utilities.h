#ifndef _CUDAIMAGEUTILITIES_H
#define _CUDAIMAGEUTILITIES_H

#include "cuda_image.h"


/* void ResampleImage_cuda(cudaPitchedPtr data,
                     int3 data_sz,float3 data_res,
                     float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                     float3 data_orig,
                     int3 virtual_sz,float3 virtual_res,
                     float virtual_d00,  float virtual_d01,float virtual_d02,float virtual_d10,float virtual_d11,float virtual_d12,float virtual_d20,float virtual_d21,float virtual_d22,
                     float3 virtual_orig,
                     int Ncomponents,
                     cudaPitchedPtr output );
*/


std::vector<CUDAIMAGE::Pointer> ComputeImageGradientImg(CUDAIMAGE::Pointer img);
void  ComputeImageGradient_cuda(cudaPitchedPtr img, const int3 data_sz, const float3 data_spc,
                                float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                                cudaPitchedPtr outputx,cudaPitchedPtr outputy,cudaPitchedPtr outputz);



CUDAIMAGE::Pointer AddImages(CUDAIMAGE::Pointer im1, CUDAIMAGE::Pointer im2);
void  AddImages_cuda(cudaPitchedPtr im1, cudaPitchedPtr im2, cudaPitchedPtr d_output, const int3 data_sz,const int ncomp);

CUDAIMAGE::Pointer MultiplyImage(CUDAIMAGE::Pointer im1, float factor);
void  MultiplyImage_cuda(cudaPitchedPtr im1, float factor, cudaPitchedPtr d_output, const int3 data_sz,const int ncomp);



CUDAIMAGE::Pointer NegateField(CUDAIMAGE::Pointer field);
void  NegateField_cuda(cudaPitchedPtr field, const int3 data_sz);

void ScaleUpdateField(CUDAIMAGE::Pointer  field,float scale_factor);
void ScaleUpdateField_cuda(cudaPitchedPtr data, int3 data_sz,float3 data_res,float scale_factor);


void AddToUpdateField(CUDAIMAGE::Pointer updateField,CUDAIMAGE::Pointer  updateField_temp,float weight,bool normalize=true);
void AddToUpdateField_cuda(cudaPitchedPtr total_data, cudaPitchedPtr to_add_data,float weight, int3 data_sz,int Ncomponents,bool normalize=true  );


void RestrictPhase(CUDAIMAGE::Pointer  field, float3 phase);
void RestrictPhase_cuda(cudaPitchedPtr field, int3 data_sz,float3 phase);


void ContrainDefFields(CUDAIMAGE::Pointer  ufield, CUDAIMAGE::Pointer  dfield);
void ContrainDefFields_cuda(cudaPitchedPtr ufield, cudaPitchedPtr dfield, int3 data_sz);



CUDAIMAGE::Pointer ComposeFields(CUDAIMAGE::Pointer main_field, CUDAIMAGE::Pointer update_field);
void ComposeFields_cuda(cudaPitchedPtr main_field,cudaPitchedPtr update_field,
             int3 data_sz,float3 data_res,
             float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
             float3 data_orig,
             cudaPitchedPtr output );



CUDAIMAGE::Pointer  InvertField(CUDAIMAGE::Pointer field,CUDAIMAGE::Pointer initial_estimate=nullptr);
void InvertField_cuda(cudaPitchedPtr field, int3 data_sz,float3 data_spc,
                      float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                      float3 data_orig,
                      cudaPitchedPtr output);


CUDAIMAGE::Pointer  PreprocessImage(CUDAIMAGE::Pointer img,float low_val, float up_val);
void PreprocessImage_cuda(cudaPitchedPtr img,
                      int3 data_sz,
                      float low_val, float up_val,
                      cudaPitchedPtr output);

void IntegrateVelocityFieldGPU(std::vector<CUDAIMAGE::Pointer> velocity_field, float lowt, float hight, CUDAIMAGE::Pointer output_field);
void IntegrateVelocityField_cuda(cudaPitchedPtr *velocity_field,
                                 cudaPitchedPtr output_field,
                                 float lowt, float hight,
                                 int NTimePoints,
                                 int3 data_sz,float3 data_res,
                                 float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                                 float3 data_orig);






#endif
