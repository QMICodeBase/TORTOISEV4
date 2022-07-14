#ifndef _QUADRATICTRANSFORMIMAGE_CUDA_CU
#define _QUADRATICTRANSFORMIMAGE_CUDA_CU

#include <stdio.h>
#include <iostream>

#include "cuda_utils.h"



#define BLOCKSIZE 32
#define PER_SLICE 1



__global__ void
QuadraticTransformImage_kernel(cudaTextureObject_t tex, int3 target_sz, int3 img_sz,
                  float *dsmat, float *dsmat_inv,
                  float *drotmat, float *dparams, 
                  char phase, bool do_cubic, 
                  cudaPitchedPtr output )
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<target_sz.x && j <target_sz.y && k<target_sz.z)
        {
            size_t opitch= output.pitch;
            size_t oslicePitch= opitch*target_sz.y*k;
            size_t ocolPitch= j*opitch;

            char *o_ptr= (char *)(output.ptr);
            char * slice_o= o_ptr+  oslicePitch;
            float * row_out= (float *)(slice_o+ ocolPitch);

            float x= dsmat[0]*i + dsmat[1]*j + dsmat[2]*k+ dsmat[3];
            float y= dsmat[4]*i + dsmat[5]*j + dsmat[6]*k+ dsmat[7];
            float z= dsmat[8]*i + dsmat[9]*j + dsmat[10]*k+ dsmat[11];

                
            x-=  dparams[21];
            y-=  dparams[22];
            z-=  dparams[23];

            float x1= drotmat[0]*x + drotmat[1]*y + drotmat[2]*z + dparams[0];
            float y1= drotmat[3]*x + drotmat[4]*y + drotmat[5]*z + dparams[1];
            float z1= drotmat[6]*x + drotmat[7]*y + drotmat[8]*z + dparams[2];
                
            float new_phase_coord= dparams[6]*x1      +dparams[7] *y1      + dparams[8] *z1+
                                   dparams[9]*x1*y1 +dparams[10]*x1*z1 + dparams[11]*y1*z1+
                                   dparams[12]*(x1*x1-y1*y1) +dparams[13]*(2*z1*z1-x1*x1-y1*y1);

                           
            float cubic_change=0;
            if(do_cubic)
            {
                cubic_change = dparams[14]*x1*y1*z1 +
                              dparams[15]*z1*(x1*x1-y1*y1) +
                              dparams[16]*x1*(4*z1*z1-x1*x1-y1*y1) +
                              dparams[17]*y1*(4*z1*z1-x1*x1-y1*y1) +
                              dparams[18]*x1*(x1*x1-3*y1*y1) +
                              dparams[19]*y1*(3*x1*x1-y1*y1) +
                              dparams[20]*z1*(2*z1*z1-3*x1*x1-3*y1*y1) ;
            }
            if(phase==0)
                x1=new_phase_coord + cubic_change;
            if(phase==1)
                y1=new_phase_coord + cubic_change;
            if(phase==2)
                z1=new_phase_coord + cubic_change;


            float iw= dsmat_inv[0]*x1 + dsmat_inv[1]*y1 + dsmat_inv[2]*z1 + dsmat_inv[3];
            float jw= dsmat_inv[4]*x1 + dsmat_inv[5]*y1 + dsmat_inv[6]*z1 + dsmat_inv[7];
            float kw= dsmat_inv[8]*x1 + dsmat_inv[9]*y1 + dsmat_inv[10]*z1 + dsmat_inv[11];


            if(iw>=0 && iw<=img_sz.x-1 && jw>=0 && jw<=img_sz.y-1 && kw>=0 && kw<=img_sz.z-1 )
                row_out[i] =tex3D<float>(tex, iw+0.5, jw +0.5, kw+0.5);
        }

    }

}


void QuadraticTransformImage_cuda(cudaTextureObject_t tex,
                                  int3 img_sz,float3 img_res, float3 img_orig, float *img_dir,
                                  int3 target_sz,float3 target_res, float3 target_orig, float *target_dir,
                                  float rotmat_arr[],
                                  float params_arr[],
                                  cudaPitchedPtr output )
{
    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*target_sz.x / blockSize.x), std::ceil(1.*target_sz.y / blockSize.y), std::ceil(1.*target_sz.z / blockSize.z/PER_SLICE) );

    float smat[16]={0},smat_inv[16]={0};
    smat[0]= target_dir[0]*target_res.x;
    smat[1]= target_dir[1]*target_res.y;
    smat[2]= target_dir[2]*target_res.z;
    smat[4]= target_dir[3]*target_res.x;
    smat[5]= target_dir[4]*target_res.y;
    smat[6]= target_dir[5]*target_res.z;
    smat[8]= target_dir[6]*target_res.x;
    smat[9]= target_dir[7]*target_res.y;
    smat[10]=target_dir[8]*target_res.z;
    smat[3]= target_orig.x;
    smat[7]= target_orig.y;
    smat[11]= target_orig.z;
    smat[15]=1;    
    
    smat_inv[0]= img_dir[0]/img_res.x;
    smat_inv[1]= img_dir[3]/img_res.x;
    smat_inv[2]= img_dir[6]/img_res.x;
    smat_inv[4]= img_dir[1]/img_res.y;
    smat_inv[5]= img_dir[4]/img_res.y;
    smat_inv[6]= img_dir[7]/img_res.y;
    smat_inv[8]= img_dir[2]/img_res.z;
    smat_inv[9]= img_dir[5]/img_res.z;
    smat_inv[10]=img_dir[8]/img_res.z;    
    smat_inv[3]= -(smat_inv[0] *img_orig.x +smat_inv[1] *img_orig.y + smat_inv[2] *img_orig.z );
    smat_inv[7]= -(smat_inv[4] *img_orig.x +smat_inv[5] *img_orig.y + smat_inv[6] *img_orig.z );
    smat_inv[11]=-(smat_inv[8] *img_orig.x +smat_inv[9] *img_orig.y + smat_inv[10] *img_orig.z );
    smat_inv[15]=1;
                        

    float *dsmat,*dsmat_inv;
    cudaMalloc((void**)&dsmat, sizeof(float)*16);
    cudaMalloc((void**)&dsmat_inv, sizeof(float)*16);    
    cudaMemcpy(dsmat, smat, sizeof(float)*16, cudaMemcpyHostToDevice);
    cudaMemcpy(dsmat_inv, smat_inv, sizeof(float)*16, cudaMemcpyHostToDevice);
        
    float *d_rotmat,*d_params;
    cudaMalloc((void**)&d_rotmat, sizeof(float)*9);
    cudaMalloc((void**)&d_params, sizeof(float)*24);    

    cudaMemcpy(d_rotmat, rotmat_arr, sizeof(float)*9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, params_arr, sizeof(float)*24, cudaMemcpyHostToDevice);
    
    char phase=0;
    if( (fabs(params_arr[7]) > fabs(params_arr[6])) && (fabs(params_arr[7]) > fabs(params_arr[8]) ) )
        phase=1;
    if( (fabs(params_arr[8]) > fabs(params_arr[6])) && (fabs(params_arr[8]) > fabs(params_arr[7]) ) )
        phase=2;
        
    bool do_cubic=false;
    for(int p=14;p<=20;p++)
    {
        if(fabs(params_arr[p]) >1E-10)
        {
            do_cubic=true;
            break;
        }
    }              
    
    QuadraticTransformImage_kernel<<< blockSize,gridSize>>>( tex,
                                                target_sz, img_sz,
                                                dsmat, dsmat_inv,
                                                d_rotmat,d_params,
                                                phase,do_cubic,
                                                output );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    cudaFree(dsmat);
    cudaFree(dsmat_inv);
    cudaFree(d_rotmat);
    cudaFree(d_params);


}


#endif
