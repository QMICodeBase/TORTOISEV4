#ifndef _RESAMPLEIMAGE_H
#define _RESAMPLEIMAGE_H

#include "cuda_image.h"





 CUDAIMAGE::Pointer ResampleImage(CUDAIMAGE::Pointer main_image, CUDAIMAGE::Pointer field_image);
 void ResampleImage_cuda(cudaPitchedPtr data,
                     int3 data_sz,float3 data_res,
                     float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                     float3 data_orig,
                     int3 virtual_sz,float3 virtual_res,
                     float virtual_d00,  float virtual_d01,float virtual_d02,float virtual_d10,float virtual_d11,float virtual_d12,float virtual_d20,float virtual_d21,float virtual_d22,
                     float3 virtual_orig,
                     int Ncomponents,
                     cudaPitchedPtr output );
                     

                     
                     
                     
                     
                     
                     
                     

#endif
