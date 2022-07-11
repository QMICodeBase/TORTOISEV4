#ifndef _WARPIMAGE_H
#define _WARPIMAGE_H

#include "cuda_image.h"





 CUDAIMAGE::Pointer WarpImage(CUDAIMAGE::Pointer main_image, CUDAIMAGE::Pointer field_image);

 void WarpImage_cuda(cudaTextureObject_t tex,int3 sz,float3 res,
                     float d00,  float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                     cudaPitchedPtr field_ptr,
                     cudaPitchedPtr output );

#endif
