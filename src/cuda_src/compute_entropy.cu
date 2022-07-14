#ifndef _COMPUTEENTROPY_CU
#define _COMPUTEENTROPY_CU

#include <stdio.h>
#include <iostream>

#include "cuda_utils.h"

#define PADDING 2

#define GRIDSIZE 16
const int bSize = 256;


__global__ void
ComputePartialHistogram_kernel(cudaPitchedPtr data, int3 sz, const int NUM_BINS,float low_lim, float high_lim, unsigned int * out)
{

    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    // linear thread index within 3D block
    int t = threadIdx.x + threadIdx.y * blockDim.x +  threadIdx.z *blockDim.x*blockDim.y ;
    // linear block index within 3D grid
    int g = blockIdx.x + blockIdx.y * gridDim.x  + gridDim.x * gridDim.y * blockIdx.z;


    extern __shared__ unsigned int smem[];
    if(t < NUM_BINS)
        smem[t] = 0;
    __syncthreads();

    if(i<sz.x && j <sz.y && k<sz.z)
    {

        size_t dpitch= data.pitch;
        size_t dslicePitch= dpitch*sz.y*k;
        size_t dcolPitch= j*dpitch;
        char *d_ptr= (char *)(data.ptr);
        char * slice_d= d_ptr+  dslicePitch;
        float * row_data= (float *)(slice_d+ dcolPitch);
        
        float val = row_data[i];

        int bin_id=   floor((val - low_lim) * NUM_BINS / (high_lim-low_lim));
        if(bin_id>=NUM_BINS)
            bin_id=NUM_BINS-1;
        if(bin_id<0)
            bin_id=0;

        atomicAdd(&smem[bin_id], 1);

        
        __syncthreads();
         
    }
    unsigned int *out2= out;
    // go to corresponding block's slice
    out2+= g*NUM_BINS;

    if(t<NUM_BINS)
    {
        out2[t]= smem[t];
    }
}


__global__ void
AccumulatePartialHistogram_kernel(unsigned int * in, int Nblocks, int Nbins,float * out)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    
    float sm=0;
    for(int b=0;b<Nblocks;b++)
    {
        sm+= in[Nbins*b + i];
    }
    out[i]=sm;
}



__global__ void
ScalarFindSum2(const float *gArr, int arraySize,  float *gOut)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*bSize;
    const int gridSize = bSize*gridDim.x;
    float sum = 0;

    for (int i = gthIdx; i < arraySize; i += gridSize)
    {
        sum+= gArr[i ];
    }

    __shared__ float shArr[bSize];
    shArr[thIdx] = sum;
    __syncthreads();

    for (int size = bSize/2; size>0; size/=2)
    { //uniform
        if (thIdx<size)
        {
                shArr[thIdx] += shArr[thIdx+size];
        }
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
}




__global__ void
ConvertHistToBinwiseEntropy_kernel(float * in, int Nbins,float *sm)
{
    uint i =  threadIdx.x;

    if(i<Nbins)
    {
        float prob= in[i]/sm[0];
        if(prob > 1E-10)
            in[i]= prob * log(prob);
        else
            in[i]=0;
    }
}




void ComputeEntropy_cuda(cudaPitchedPtr img, int3 sz, const int Nbins, const float low_lim, const float high_lim, float &value)
{
    unsigned int curr_grid_size=GRIDSIZE;
    dim3 bs;
    bs.x=std::ceil(1.*sz.x / curr_grid_size);
    bs.y=std::ceil(1.*sz.y / curr_grid_size);
    bs.z=std::ceil(1.*sz.z / curr_grid_size);
    while( bs.x * bs.y * bs.z < Nbins)
    {
        curr_grid_size/=2;
        bs.x=std::ceil(1.*sz.x / curr_grid_size);
        bs.y=std::ceil(1.*sz.y / curr_grid_size);
        bs.z=std::ceil(1.*sz.z / curr_grid_size);
    }

    const dim3 gridSize(curr_grid_size,curr_grid_size,curr_grid_size);
    const dim3 blockSize(std::ceil(1.*sz.x / gridSize.x), std::ceil(1.*sz.y / gridSize.y), std::ceil(1.*sz.z / gridSize.z) );
    


    int Nblocks= gridSize.x*gridSize.y*gridSize.z;

    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, Nblocks * Nbins * sizeof(unsigned int));
    cudaMemset(d_part_hist, 0, Nblocks*Nbins * sizeof(unsigned int));
 
    ComputePartialHistogram_kernel <<< gridSize, blockSize,Nbins*sizeof(unsigned int) >>> (img,sz, Nbins, low_lim,high_lim,d_part_hist);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    
    float *d_hist;
    cudaMalloc(&d_hist,  Nbins * sizeof(float));
    cudaMemset(d_hist, 0, Nbins * sizeof(float));
    const dim3 blockSize2(Nbins);
    AccumulatePartialHistogram_kernel<<< 1, blockSize2 >>> (d_part_hist,Nblocks, Nbins,d_hist);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(d_part_hist);
    
    
    float* hist_sum;
    cudaMalloc((void**)&hist_sum, sizeof(float));
    ScalarFindSum2<<<1, bSize>>>(d_hist, Nbins, hist_sum);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());  


    ConvertHistToBinwiseEntropy_kernel<<<1, Nbins>>> (d_hist, Nbins, hist_sum);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());  
    cudaFree(hist_sum);
    
    float* d_entropy;
    cudaMalloc((void**)&d_entropy, sizeof(float));
    ScalarFindSum2<<<1, bSize>>>(d_hist, Nbins, d_entropy);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());  
    cudaFree(d_hist);
    
    cudaMemcpy(&value, d_entropy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_entropy);

}





__global__ void
AccumulateJointPartialHistogram_kernel(unsigned int * in, int Nblocks, int Nbins,float * out)
{
    uint row= blockIdx.x;
    uint col= threadIdx.x;

    float sm=0;
    int Nbins2=Nbins*Nbins;
    for(int b=0;b<Nblocks;b++)
    {
        sm+= in[Nbins2*b + row*Nbins + col ];
    }
    out[row*Nbins + col]=sm;
}






__global__ void
ComputeJointPartialHistogram_kernel(cudaPitchedPtr data1, float low_lim1, float high_lim1,cudaPitchedPtr data2, float low_lim2, float high_lim2,  int3 sz, const int NUM_BINS, unsigned int * out)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    // linear thread index within 3D block
    int t = threadIdx.x + threadIdx.y * blockDim.x +  threadIdx.z *blockDim.x*blockDim.y ;
    // linear block index within 3D grid
    int g = blockIdx.x + blockIdx.y * gridDim.x  + gridDim.x * gridDim.y * blockIdx.z;

    extern __shared__ unsigned int smem[];
    if(t < NUM_BINS)
    {
        for(int t1=0;t1<NUM_BINS;t1++)
            smem[t1*NUM_BINS + t] = 0;
    }
    __syncthreads();

    if(i<sz.x && j <sz.y && k<sz.z)
    {
        size_t dpitch1= data1.pitch;
        size_t dslicePitch1= dpitch1*sz.y*k;
        size_t dcolPitch1= j*dpitch1;
        char *d_ptr1= (char *)(data1.ptr);
        char * slice_d1= d_ptr1+  dslicePitch1;
        float * row_data1= (float *)(slice_d1+ dcolPitch1);

        size_t dpitch2= data2.pitch;
        size_t dslicePitch2= dpitch2*sz.y*k;
        size_t dcolPitch2= j*dpitch2;
        char *d_ptr2= (char *)(data2.ptr);
        char * slice_d2= d_ptr2+  dslicePitch2;
        float * row_data2= (float *)(slice_d2+ dcolPitch2);

        float val1 = row_data1[i];
        float val2 = row_data2[i];


        if(val1>=low_lim1 && val2>=low_lim2 && val1<=high_lim1 && val2<=high_lim2)
        {
            float m_MovingImageBinSize =(high_lim2 -low_lim2)/ (NUM_BINS-2*PADDING);
            float m_MovingImageNormalizedMin = low_lim2/m_MovingImageBinSize - PADDING;
            float m_FixedImageBinSize =(high_lim1 -low_lim1)/ (NUM_BINS-2*PADDING);
            float m_FixedImageNormalizedMin = low_lim1/m_FixedImageBinSize - PADDING;

            float movingImageParzenWindowTerm = val2 / m_MovingImageBinSize - m_MovingImageNormalizedMin;
            auto movingImageParzenWindowIndex = (int)( movingImageParzenWindowTerm );
            if( movingImageParzenWindowIndex < 2 )
            {
                movingImageParzenWindowIndex = 2;
            }
            else
            {
               if( movingImageParzenWindowIndex > NUM_BINS-3 )
               {
                  movingImageParzenWindowIndex = NUM_BINS-3;
               }
            }

            float fixedImageParzenWindowTerm = val1 / m_FixedImageBinSize - m_FixedImageNormalizedMin;
            auto fixedImageParzenWindowIndex = (int)( fixedImageParzenWindowTerm );
            if( fixedImageParzenWindowIndex < 2 )
            {
                fixedImageParzenWindowIndex = 2;
            }
            else
            {
               if( fixedImageParzenWindowIndex > NUM_BINS-3)
               {
                  fixedImageParzenWindowIndex = NUM_BINS-3;
               }
            }

            atomicAdd(&smem[fixedImageParzenWindowIndex*NUM_BINS + movingImageParzenWindowIndex], 1);





        }

       /*
        if(val1>=low_lim1 && val2>=low_lim2 && val1<high_lim1 && val2<high_lim2)
        {
            int bin_id1=   floor((val1 - low_lim1) * NUM_BINS / (high_lim1-low_lim1));
            int bin_id2=   floor((val2 - low_lim2) * NUM_BINS / (high_lim2-low_lim2));
            atomicAdd(&smem[bin_id1*NUM_BINS + bin_id2], 1);
        }
       */
    }
    __syncthreads();

    unsigned int *out2= out;
    // go to corresponding block's slice
    out2+= g*NUM_BINS*NUM_BINS;

    if(t<NUM_BINS)
    {
        for(int t1=0;t1<NUM_BINS;t1++)
            out2[t1*NUM_BINS+t]= smem[t1*NUM_BINS+t];
    }
}






__global__ void
ConvertJointHistToBinwiseEntropy_kernel(float * in, int Nbins,float *sm)
{
    uint row= blockIdx.x;
    uint col= threadIdx.x;

    float prob= in[row*Nbins+col]/sm[0];
    if(prob > 1E-10)
        in[row*Nbins+col]= prob * log(prob);
    else
        in[row*Nbins+col]=0;
}



__global__ void
ConvertJointHistogramToMovingHistogram_kernel(float * joint_hist, int Nbins,float *moving_hist)
{
    uint col= threadIdx.x;


    float sm=0;
    for(int r=0;r<Nbins;r++)
    {
        sm+= joint_hist[r*Nbins + col];
    }
    moving_hist[col]=sm;
}






void ComputeJointEntropy_cuda(cudaPitchedPtr img1, float low_lim1, float high_lim1, cudaPitchedPtr img2, float low_lim2, float high_lim2, const int3 sz, const int Nbins,  float &value1, float &value2)
{
    unsigned int curr_grid_size=GRIDSIZE;
    dim3 bs;
    bs.x=std::ceil(1.*sz.x / curr_grid_size);
    bs.y=std::ceil(1.*sz.y / curr_grid_size);
    bs.z=std::ceil(1.*sz.z / curr_grid_size);
    while( bs.x * bs.y * bs.z < Nbins)
    {
        curr_grid_size/=2;
        bs.x=std::ceil(1.*sz.x / curr_grid_size);
        bs.y=std::ceil(1.*sz.y / curr_grid_size);
        bs.z=std::ceil(1.*sz.z / curr_grid_size);
    }

    const dim3 gridSize(curr_grid_size,curr_grid_size,curr_grid_size);
    const dim3 blockSize(std::ceil(1.*sz.x / gridSize.x), std::ceil(1.*sz.y / gridSize.y), std::ceil(1.*sz.z / gridSize.z) );

    int Nblocks= gridSize.x*gridSize.y*gridSize.z;

    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, Nblocks * Nbins * Nbins * sizeof(unsigned int));
    cudaMemset(d_part_hist, 0, Nblocks*Nbins * Nbins * sizeof(unsigned int));

    ComputeJointPartialHistogram_kernel <<< gridSize, blockSize,Nbins*Nbins*sizeof(unsigned int) >>> (img1, low_lim1, high_lim1, img2, low_lim2, high_lim2, sz, Nbins, d_part_hist);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    float *d_hist;
    cudaMalloc(&d_hist,  Nbins*Nbins * sizeof(float));
    cudaMemset(d_hist, 0, Nbins*Nbins * sizeof(float));
    AccumulateJointPartialHistogram_kernel<<< Nbins, Nbins >>> (d_part_hist,Nblocks, Nbins,d_hist);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(d_part_hist);


    {
        float *d_hist_img2;
        cudaMalloc(&d_hist_img2,  Nbins * sizeof(float));
        cudaMemset(d_hist_img2, 0, Nbins * sizeof(float));
        ConvertJointHistogramToMovingHistogram_kernel<<< 1 ,Nbins >>>(d_hist,Nbins,d_hist_img2);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        float* hist_sum;
        cudaMalloc((void**)&hist_sum, sizeof(float));
        ScalarFindSum2<<<1, bSize>>>(d_hist_img2, Nbins, hist_sum);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        ConvertHistToBinwiseEntropy_kernel<<<1, Nbins>>> (d_hist_img2, Nbins, hist_sum);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(hist_sum);

        float* d_entropy_img2;
        cudaMalloc((void**)&d_entropy_img2, sizeof(float));
        ScalarFindSum2<<<1, bSize>>>(d_hist_img2, Nbins, d_entropy_img2);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(d_hist_img2);

        cudaMemcpy(&value2, d_entropy_img2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_entropy_img2);
    }

/*
    float okan_hist[40*40];
    cudaMemcpy(okan_hist,d_hist,sizeof(float)*40*40,cudaMemcpyDeviceToHost);
    for(int r=0;r<40;r++)
    {
        for(int c=0;c<40;c++)
        {
            std::cout<<okan_hist[r*40+c]<< " ";
        }
        std::cout<<std::endl;
    }
*/



    float* hist_sum;
    cudaMalloc((void**)&hist_sum, sizeof(float));
    ScalarFindSum2<<<1, bSize>>>(d_hist, Nbins*Nbins, hist_sum);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    ConvertJointHistToBinwiseEntropy_kernel<<<Nbins, Nbins>>> (d_hist, Nbins, hist_sum);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(hist_sum);

    float* d_entropy;
    cudaMalloc((void**)&d_entropy, sizeof(float));
    ScalarFindSum2<<<1, bSize>>>(d_hist, Nbins*Nbins, d_entropy);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(d_hist);

    cudaMemcpy(&value1, d_entropy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_entropy);

}



#endif
