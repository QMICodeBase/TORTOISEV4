#ifndef WARP_IMAGE_CUDA_CU
#define WARP_IMAGE_CUDA_CU

#include <stdio.h>
#include <iostream>

#include "cuda_utils.h"



#define BLOCKSIZE 32
#define PER_SLICE 1

__constant__ float dir[9];






__global__ void
warp_image_kernel(cudaTextureObject_t tex, int3 sz, float3 res,
                  cudaPitchedPtr field_ptr,
                  cudaPitchedPtr output )
{



    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    //int id= __umul24(k, sy*sx) + __umul24(j, sx) + i;


    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<sz.x && j <sz.y && k<sz.z)
        {
            size_t opitch= output.pitch;
            size_t oslicePitch= opitch*sz.y*k;
            size_t ocolPitch= j*opitch;

            char *o_ptr= (char *)(output.ptr);
            char * slice_o= o_ptr+  oslicePitch;
            float * row_out= (float *)(slice_o+ ocolPitch);

            size_t fpitch= field_ptr.pitch;
            size_t fslicePitch= fpitch*sz.y*k;
            size_t fcolPitch= j*fpitch;

            char *f_ptr= (char *)(field_ptr.ptr);
            char * slice_f= f_ptr+  fslicePitch;
            float * row_f= (float *)(slice_f+ fcolPitch);


            float x= (dir[0]*i  + dir[1]*j + dir[2]*k)* res.x ;
            float y= (dir[3]*i  + dir[4]*j + dir[5]*k)* res.y ;
            float z= (dir[6]*i  + dir[7]*j + dir[8]*k)* res.z ;


            float xw= x + row_f[3*i];
            float yw= y + row_f[3*i+1];
            float zw= z + row_f[3*i+2];

            float iw = (dir[0]*xw  + dir[3]*yw  + dir[6]*zw)/ res.x ;
            float jw = (dir[1]*xw  + dir[4]*yw  + dir[7]*zw)/ res.y ;
            float kw = (dir[2]*xw  + dir[5]*yw  + dir[8]*zw)/ res.z ;

            row_out[i] =tex3D<float>(tex, iw+0.5, jw +0.5, kw+0.5);
        }
    }

}



void WarpImage_cuda(cudaTextureObject_t tex, int3 sz , float3 res,
                    float d00,  float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                    cudaPitchedPtr field_ptr,
                    cudaPitchedPtr output )
{




    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*sz.x / blockSize.x), std::ceil(1.*sz.y / blockSize.y), std::ceil(1.*sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*sz.x / blockSize.x), std::ceil(1.*sz.y / blockSize.y), std::ceil(1.*sz.z / blockSize.z/PER_SLICE) );
    }


    float h_dir[]= {d00,d01,d02,d10,d11,d12,d20,d21,d22};
    gpuErrchk(cudaMemcpyToSymbol(dir, &h_dir, 9 * sizeof(float)));

    warp_image_kernel<<< blockSize,gridSize>>>( tex, sz, res,
                                                field_ptr,
                                                output );


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


}


#endif
