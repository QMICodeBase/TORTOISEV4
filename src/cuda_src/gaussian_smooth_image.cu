#ifndef _GAUSSIANSMOOTHIMAGE_CUDA_CU
#define _GAUSSIANSMOOTHIMAGE_CUDA_CU

#include <stdio.h>
#include <iostream>

#include "cuda_utils.h"



#define BLOCKSIZE 32
#define MAX_KERNEL_WIDTH 31
#define OTHER_DIM_THREADS 1

extern __constant__ int d_sz[3];
#define PER_SLICE 1

__constant__ float c_Kernel[MAX_KERNEL_WIDTH];



__global__ void
GaussianSmoothImageRow_kernel(cudaPitchedPtr data, cudaPitchedPtr output ,int Ncomponents, int kernel_sz)
{

  //  uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  //  uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  //  uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    uint i= threadIdx.x;
    uint j= blockIdx.y;
    uint k= blockIdx.z;


     if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
     {
         /*
         if(i==100 && j==100&& k==60)
         {
             printf("%d %d %d\n",blockDim.x,blockDim.y,blockDim.z );
             printf("%d %d %d\n",gridDim.x,gridDim.y,gridDim.z );
         }
         if(blockIdx.x!=0)
             printf("%d \n", blockIdx.x);
             */

         size_t dpitch= data.pitch;
         size_t dslicePitch= dpitch*d_sz[1]*k;
         size_t dcolPitch= j*dpitch;
         char *d_ptr= (char *)(data.ptr);
         char * slice_d= d_ptr+  dslicePitch;
         float * row_data= (float *)(slice_d+ dcolPitch);

         size_t opitch= output.pitch;
         size_t oslicePitch= opitch*d_sz[1]*k;
         size_t ocolPitch= j*opitch;
         char *o_ptr= (char *)(output.ptr);
         char * slice_o= o_ptr+  oslicePitch;
         float * row_out= (float *)(slice_o+ ocolPitch);


         const int MASK_WIDTH= kernel_sz;

         extern __shared__ float N_s[];

         N_s[threadIdx.x]=row_data[i* Ncomponents + blockIdx.x];
         __syncthreads();


         float val=0;
         for(int j =0; j < MASK_WIDTH; j++)
         {
             if((int)threadIdx.x+j-(MASK_WIDTH/2)>=0  && threadIdx.x+j-(MASK_WIDTH/2)<d_sz[0])
                 val+=N_s[threadIdx.x+j-(MASK_WIDTH/2)]*c_Kernel[j];
         }
         row_out[i* Ncomponents + blockIdx.x]=val;
     }
}


__global__ void
GaussianSmoothImageCol_kernel(cudaPitchedPtr data, cudaPitchedPtr output ,int Ncomponents, int kernel_sz)
{

  //  uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  //  uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
 //   uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    uint i= blockIdx.x;
    uint j= threadIdx.y;
    uint k= blockIdx.z;


     if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
     {
         size_t dpitch= data.pitch;
         size_t dslicePitch= dpitch*d_sz[1]*k;
         size_t dcolPitch= j*dpitch;
         char *d_ptr= (char *)(data.ptr);
         char * slice_d= d_ptr+  dslicePitch;
         float * row_data= (float *)(slice_d+ dcolPitch);

         size_t opitch= output.pitch;
         size_t oslicePitch= opitch*d_sz[1]*k;
         size_t ocolPitch= j*opitch;
         char *o_ptr= (char *)(output.ptr);
         char * slice_o= o_ptr+  oslicePitch;
         float * row_out= (float *)(slice_o+ ocolPitch);


         const int MASK_WIDTH= kernel_sz;

         extern __shared__ float N_s[];

         N_s[threadIdx.y]=row_data[i* Ncomponents + blockIdx.y];
         __syncthreads();


         float val=0;
         for(int j =0; j < MASK_WIDTH; j++)
         {
             if((int)threadIdx.y+j-(MASK_WIDTH/2)>=0  && threadIdx.y+j-(MASK_WIDTH/2)<d_sz[1])
                 val+=N_s[threadIdx.y+j-(MASK_WIDTH/2)]*c_Kernel[j];
         }
         row_out[i* Ncomponents + blockIdx.y]=val;
     }
}



__global__ void
GaussianSmoothImageSlice_kernel(cudaPitchedPtr data, cudaPitchedPtr output ,int Ncomponents, int kernel_sz)
{

 //   uint k = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
 //   uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
 //   uint i = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;


    uint i= blockIdx.z;
    uint j= blockIdx.y;
    uint k= threadIdx.x;

     if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
     {
         size_t dpitch= data.pitch;
         size_t dslicePitch= dpitch*d_sz[1]*k;
         size_t dcolPitch= j*dpitch;
         char *d_ptr= (char *)(data.ptr);
         char * slice_d= d_ptr+  dslicePitch;
         float * row_data= (float *)(slice_d+ dcolPitch);

         size_t opitch= output.pitch;
         size_t oslicePitch= opitch*d_sz[1]*k;
         size_t ocolPitch= j*opitch;
         char *o_ptr= (char *)(output.ptr);
         char * slice_o= o_ptr+  oslicePitch;
         float * row_out= (float *)(slice_o+ ocolPitch);


         const int MASK_WIDTH= kernel_sz;

         extern __shared__ float N_s[];

         N_s[threadIdx.x]=row_data[i* Ncomponents + blockIdx.x];
         __syncthreads();


         float val=0;
         for(int j =0; j < MASK_WIDTH; j++)
         {
             if((int)threadIdx.x+j-(MASK_WIDTH/2)>=0  && threadIdx.x+j-(MASK_WIDTH/2)<d_sz[2])
                 val+=N_s[threadIdx.x+j-(MASK_WIDTH/2)]*c_Kernel[j];
         }
         row_out[i* Ncomponents + blockIdx.x]=val;         
     }
}


void GaussianSmoothImage_cuda(cudaPitchedPtr data,
                     int3 data_sz,                     
                     int Ncomponents,
                     int kernel_sz,
                     float *h_kernel,
                     cudaPitchedPtr output )
{
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(c_Kernel, h_kernel, kernel_sz * sizeof(float)));





    cudaPitchedPtr buffer1,buffer2;
    cudaExtent extent =  make_cudaExtent(Ncomponents*sizeof(float)*data_sz.x,data_sz.y,data_sz.z);
    gpuErrchk(cudaMalloc3D(&buffer1, extent));
    gpuErrchk(cudaMalloc3D(&buffer2, extent));


    {
        const dim3 blockSize(Ncomponents, data_sz.y, data_sz.z);
        const dim3 gridSize(data_sz.x, 1,1 );
        gpuErrchk(cudaDeviceSynchronize());        
        GaussianSmoothImageRow_kernel<<< blockSize,gridSize, data_sz.x*sizeof(float)>>>   ( data,buffer1,Ncomponents,kernel_sz);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }


    {

        const dim3 blockSize(data_sz.x, Ncomponents,  data_sz.z);
        const dim3 gridSize(1,data_sz.y, 1 );
        gpuErrchk(cudaDeviceSynchronize());
        GaussianSmoothImageCol_kernel<<< blockSize,gridSize, data_sz.y*sizeof(float)>>>   ( buffer1,buffer2,Ncomponents,kernel_sz);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }


    {

        const dim3 blockSize(Ncomponents, data_sz.y, data_sz.x);
        const dim3 gridSize(data_sz.z, 1,1 );
        gpuErrchk(cudaDeviceSynchronize());
        GaussianSmoothImageSlice_kernel<<< blockSize,gridSize, data_sz.z*sizeof(float)>>>   ( buffer2,output,Ncomponents,kernel_sz);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }





    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(buffer1.ptr));
    gpuErrchk(cudaFree(buffer2.ptr));
}


__global__ void
AdjustFieldBoundary_kernel( cudaPitchedPtr orig_img,cudaPitchedPtr smooth_img, const float weight1, const float weight2)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
            {
                size_t pitch= orig_img.pitch;
                size_t slicePitch= pitch*d_sz[1]*k;
                size_t colPitch= j*pitch;

                char *orig_ptr= (char *)(orig_img.ptr);
                char * slice_orig= orig_ptr+  slicePitch;
                float * row_orig= (float *)(slice_orig+ colPitch);

                char *smooth_ptr= (char *)(smooth_img.ptr);
                char * slice_smooth= smooth_ptr+  slicePitch;
                float * row_smooth= (float *)(slice_smooth+ colPitch);


                if(i==0 || i==d_sz[0]-1 || j==0 || j==d_sz[1]-1 || k==0 || k==d_sz[2]-1 )
                {
                    row_smooth[3*i]=0;
                    row_smooth[3*i+1]=0;
                    row_smooth[3*i+2]=0;
                }
                else
                {
                    row_smooth[3*i]= row_smooth[3*i]*weight1 + row_orig[3*i]*weight2;
                    row_smooth[3*i+1]= row_smooth[3*i+1]*weight1 + row_orig[3*i+1]*weight2;
                    row_smooth[3*i+2]= row_smooth[3*i+2]*weight1 + row_orig[3*i+2]*weight2;
                }
           }
    }
}



void AdjustFieldBoundary(cudaPitchedPtr orig_img,cudaPitchedPtr smooth_img,int3 data_sz, float weight1, float weight2)
{
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));



    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    }

    AdjustFieldBoundary_kernel<<< blockSize,gridSize>>>( orig_img, smooth_img,weight1, weight2);


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());



}


#endif
