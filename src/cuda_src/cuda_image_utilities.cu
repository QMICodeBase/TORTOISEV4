#ifndef _CUDAIMAGEUTILITIES_CU
#define _CUDAIMAGEUTILITIES_CU

#include <stdio.h>
#include <iostream>

#include "cuda_utils.h"



#define BLOCKSIZE 32
#define PER_SLICE 1

static const int bSize = 1024;
static const int gSize = 24;

extern __constant__ float d_dir[9];
extern __constant__ int d_sz[3];
extern __constant__ float d_orig[3];
extern __constant__ float d_spc[3];





__global__ void
FieldFindMaxLocalNorm(const float *gArr, int arraySize, const float3 spc, float *gOut)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*bSize;
    const int gridSize = bSize*gridDim.x;
    float mx = -1;

    for (int i = gthIdx; i < arraySize; i += gridSize)
    {
        float sm = (gArr[3*i  ]/spc.x)*(gArr[3*i  ]/spc.x) +
                   (gArr[3*i+1]/spc.y)*(gArr[3*i+1]/spc.y) +
                   (gArr[3*i+2]/spc.z)*(gArr[3*i+2]/spc.z) ;
        sm=sqrt(sm);
        if(sm>mx)
            mx=sm;
    }


    __shared__ float shArr[bSize];
    shArr[thIdx] = mx;
    __syncthreads();
    for (int size = bSize/2; size>0; size/=2)
    { //uniform
        if (thIdx<size)
        {
            if(shArr[thIdx+size]> shArr[thIdx])
                shArr[thIdx] = shArr[thIdx+size];
        }
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
}



__global__ void
FieldFindSumLocalNorm(const float *gArr, int arraySize, const float3 spc, float *gOut)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*bSize;
    const int gridSize = bSize*gridDim.x;
    float sum = 0;

    for (int i = gthIdx; i < arraySize; i += gridSize)
    {
        float nrm= (gArr[3*i  ]/spc.x)*(gArr[3*i  ]/spc.x) +
                   (gArr[3*i+1]/spc.y)*(gArr[3*i+1]/spc.y) +
                   (gArr[3*i+2]/spc.z)*(gArr[3*i+2]/spc.z) ;
        nrm=sqrt(nrm);
        sum+=nrm;
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
ScalarFindMax(const float *gArr, int arraySize,  float *gOut)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*bSize;
    const int gridSize = bSize*gridDim.x;
    float mx = -1;

    for (int i = gthIdx; i < arraySize; i += gridSize)
    {
        float sm = gArr[i ];
        if(sm>mx)
            mx=sm;
    }

    __shared__ float shArr[bSize];
    shArr[thIdx] = mx;
    __syncthreads();

    for (int size = bSize/2; size>0; size/=2)
    { //uniform
        if (thIdx<size)
        {
            if(shArr[thIdx+size]> shArr[thIdx])
                shArr[thIdx] = shArr[thIdx+size];
        }
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
}



__global__ void
ScalarFindMin(const float *gArr, int arraySize,  float *gOut)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*bSize;
    const int gridSize = bSize*gridDim.x;
    float mn = 1E100;

    for (int i = gthIdx; i < arraySize; i += gridSize)
    {
        float sm = gArr[i ];
        if(sm<mn)
            mn=sm;
    }

    __shared__ float shArr[bSize];
    shArr[thIdx] = mn;
    __syncthreads();

    for (int size = bSize/2; size>0; size/=2)
    { //uniform
        if (thIdx<size)
        {
            if(shArr[thIdx+size]< shArr[thIdx])
                shArr[thIdx] = shArr[thIdx+size];
        }
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
}





__global__ void
ScalarFindSum(const double *gArr, int arraySize,  double *gOut)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*bSize;
    const int gridSize = bSize*gridDim.x;
    double sum = 0;

    for (int i = gthIdx; i < arraySize; i += gridSize)
    {
        sum+= gArr[i ];
    }

    __shared__ double shArr[bSize];
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
ScalarFindSum(const float *gArr, int arraySize,  float *gOut)
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
ScalarFindSumSq(const float *gArr, int arraySize,  float *gOut)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*bSize;
    const int gridSize = bSize*gridDim.x;
    float sum = 0;

    for (int i = gthIdx; i < arraySize; i += gridSize)
    {
        sum+= gArr[i ]*gArr[i];
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
AddToUpdateField_kernel(cudaPitchedPtr total_data, cudaPitchedPtr to_add_data , float weight, int3 d_sz, int Ncomponents)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
        {
            size_t opitch= total_data.pitch;
            size_t oslicePitch= opitch*d_sz.y*k;
            size_t ocolPitch= j*opitch;

            char *o_ptr= (char *)(total_data.ptr);
            char * slice_o= o_ptr+  oslicePitch;
            float * row_out= (float *)(slice_o+ ocolPitch);


            char *a_ptr= (char *)(to_add_data.ptr);
            char * slice_a= a_ptr+  oslicePitch;
            float * row_add= (float *)(slice_a+ ocolPitch);                     

            for(int mm=0;mm<Ncomponents;mm++)
                row_out[Ncomponents*i + mm]=   row_out[Ncomponents*i + mm] + row_add[Ncomponents*i + mm]*weight;
        }
    }

}


void AddToUpdateField_cuda(cudaPitchedPtr total_data, cudaPitchedPtr to_add_data,float weight, const int3 data_sz,int Ncomponents , bool normalize )
{

    float magnitude=1;

    if(normalize)
    {
        float* dev_out;
        cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

        ScalarFindSumSq<<<gSize, bSize>>>((float *)to_add_data.ptr, to_add_data.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
        ScalarFindSum<<<1, bSize>>>(dev_out, gSize, dev_out);
        cudaDeviceSynchronize();

        cudaMemcpy(&magnitude, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_out);
        magnitude=sqrt(magnitude);

    }

    if(magnitude!=0)
    {
        dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
        dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
        while(gridSize.x *gridSize.y *gridSize.z >1024)
        {
            blockSize.x*=2;
            blockSize.y*=2;
            blockSize.z*=2;
            gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
        }

        AddToUpdateField_kernel<<< blockSize,gridSize>>>(  total_data,to_add_data,weight/magnitude,data_sz,Ncomponents );
    }


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}




__global__ void
ScaleUpdateField_kernel(cudaPitchedPtr field, const int3 d_sz, float scale)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t opitch= field.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;

                char *o_ptr= (char *)(field.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_data= (float *)(slice_o+ ocolPitch);

                row_data[3*i]= row_data[3*i]*scale;
                row_data[3*i+1]= row_data[3*i+1]*scale;
                row_data[3*i+2]= row_data[3*i+2]*scale;
            }
    }
}


float TotalNorm_cuda(cudaPitchedPtr field, const int3 data_sz)
{
    float magnitude=1;
    float* dev_out;
    cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

    ScalarFindSumSq<<<gSize, bSize>>>((float *)field.ptr, field.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
    ScalarFindSum<<<1, bSize>>>(dev_out, gSize, dev_out);
    cudaDeviceSynchronize();

    cudaMemcpy(&magnitude, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_out);
    magnitude=sqrt(magnitude);

    return magnitude;
}



void ScaleUpdateField_cuda(cudaPitchedPtr field, const int3 data_sz,float3 spc, float scale_factor )
{
    float magnitude=0;

    {               
        float* dev_out;
        cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

        FieldFindMaxLocalNorm<<<gSize, bSize>>>((float *)field.ptr, field.pitch/sizeof(float)/3*data_sz.y*data_sz.z,spc,dev_out);
        ScalarFindMax<<<1, bSize>>>(dev_out, gSize, dev_out);
        cudaDeviceSynchronize();


        cudaMemcpy(&magnitude, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_out);                
    }

    {
        dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
        dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
        while(gridSize.x *gridSize.y *gridSize.z >1024)
        {
            blockSize.x*=2;
            blockSize.y*=2;
            blockSize.z*=2;
            gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
        }

        if(magnitude>1E-20)
        {
            ScaleUpdateField_kernel<<< blockSize,gridSize>>>(  field, data_sz, scale_factor/magnitude );
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
    }


}


float ComputeFieldScale_cuda(cudaPitchedPtr field, const int3 data_sz,const float3 spc)
{
    float magnitude=0;

        float* dev_out;
        cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

        FieldFindMaxLocalNorm<<<gSize, bSize>>>((float *)field.ptr, field.pitch/sizeof(float)/3*data_sz.y*data_sz.z,spc,dev_out);
        ScalarFindMax<<<1, bSize>>>(dev_out, gSize, dev_out);
        cudaDeviceSynchronize();


        cudaMemcpy(&magnitude, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_out);

        return  magnitude;

}



__global__ void
RestrictPhase_kernel(cudaPitchedPtr field, const int3 d_sz, float3 phase)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t opitch= field.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;

                char *o_ptr= (char *)(field.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_data= (float *)(slice_o+ ocolPitch);
                
                float nrm = row_data[3*i]*row_data[3*i] +row_data[3*i+1]*row_data[3*i+1] + row_data[3*i+2]*row_data[3*i+2] ; 
                nrm=sqrt(nrm);
                if(nrm!=0)
                {
                    float vec_x = row_data[3*i]/nrm;
                    float vec_y = row_data[3*i+1]/nrm;
                    float vec_z = row_data[3*i+2]/nrm;                                        
                    
                    float dot = vec_x*phase.x + vec_y*phase.y + vec_z*phase.z ;
                    row_data[3*i] =  phase.x*nrm*dot;
                    row_data[3*i+1] = phase.y*nrm*dot;
                    row_data[3*i+2] = phase.z*nrm*dot;
                }

          }          
    }
}



void RestrictPhase_cuda(cudaPitchedPtr field, const int3 data_sz,float3 phase )
{
        dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
        dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
        while(gridSize.x *gridSize.y *gridSize.z >1024)
        {
            blockSize.x*=2;
            blockSize.y*=2;
            blockSize.z*=2;
            gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
        }

        RestrictPhase_kernel<<< blockSize,gridSize>>>(  field, data_sz, phase );


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


__global__ void
ContrainDefFields_kernel(cudaPitchedPtr ufield, cudaPitchedPtr dfield, const int3 d_sz)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t opitch= ufield.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;

                char *u_ptr= (char *)(ufield.ptr);
                char * slice_u= u_ptr+  oslicePitch;
                float * row_data_u= (float *)(slice_u+ ocolPitch);
                
                char *d_ptr= (char *)(dfield.ptr);
                char * slice_d= d_ptr+  oslicePitch;
                float * row_data_d= (float *)(slice_d+ ocolPitch);
                
                for(int v=0;v<3;v++)
                {
                    float val=(row_data_u[3*i+v] - row_data_d[3*i+v])*0.5;
                    row_data_u[3*i+v]=val;
                    row_data_d[3*i+v]=-val;                
                }                             
           }          
    }
}



void ContrainDefFields_cuda(cudaPitchedPtr ufield, cudaPitchedPtr dfield, const int3 data_sz)
{
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    }

    ContrainDefFields_kernel<<< blockSize,gridSize>>>(  ufield, dfield, data_sz );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}


__global__ void
ComposeFields_kernel(cudaPitchedPtr main_field,cudaPitchedPtr update_field,  cudaPitchedPtr output )
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
            {
                size_t mpitch= main_field.pitch;
                size_t mslicePitch= mpitch*d_sz[1]*k;
                size_t mcolPitch= j*mpitch;

                char *m_ptr= (char *)(main_field.ptr);
                char * slice_m= m_ptr+  mslicePitch;
                float * row_m= (float *)(slice_m+ mcolPitch);

                char *o_ptr= (char *)(output.ptr);
                char * slice_o= o_ptr+  mslicePitch;
                float * row_o= (float *)(slice_o+ mcolPitch);

                float x[3];
                x[0]= (d_dir[0]*i  + d_dir[1]*j + d_dir[2]*k)* d_spc[0] ;
                x[1]= (d_dir[3]*i  + d_dir[4]*j + d_dir[5]*k)* d_spc[1] ;
                x[2]= (d_dir[6]*i  + d_dir[7]*j + d_dir[8]*k)* d_spc[2] ;


                float xp[3];
                xp[0]= x[0] + row_m[3*i];
                xp[1]= x[1] + row_m[3*i+1];
                xp[2]= x[2] + row_m[3*i+2];

                float iw = (d_dir[0]*xp[0]  + d_dir[3]*xp[1]  + d_dir[6]*xp[2])/ d_spc[0] ;
                float jw = (d_dir[1]*xp[0]  + d_dir[4]*xp[1]  + d_dir[7]*xp[2])/ d_spc[1] ;
                float kw = (d_dir[2]*xp[0]  + d_dir[5]*xp[1]  + d_dir[8]*xp[2])/ d_spc[2] ;


                if(iw<0 || iw> d_sz[0]-1 || jw<0 || jw> d_sz[1]-1 || kw<0 || kw> d_sz[2]-1)
                {
                    for(int mm=0;mm<3;mm++)
                       row_o[3*i +mm]=row_m[3*i +mm];
                }
                else
                {
                    int floor_x = __float2int_rd(iw);
                    int floor_y = __float2int_rd(jw);
                    int floor_z = __float2int_rd(kw);

                    int ceil_x = __float2int_ru(iw);
                    int ceil_y = __float2int_ru(jw);
                    int ceil_z = __float2int_ru(kw);

                    float xd= iw - floor_x;
                    float yd= jw - floor_y;
                    float zd= kw - floor_z;

                    float ia1[3],ib1[3],ia2[3],ib2[3], ja1[3],jb1[3],ja2[3],jb2[3];


                    size_t dpitch= update_field.pitch;
                    char *d_ptr= (char *)(update_field.ptr);
                    {
                        size_t dslicePitch= dpitch*d_sz[1]*floor_z;
                        char * slice_d= d_ptr+  dslicePitch;
                        {
                            size_t dcolPitch= floor_y*dpitch;
                            float * row_data= (float *)(slice_d+ dcolPitch);
                            for(int mm=0;mm<3;mm++)
                            {
                                ia1[mm] = row_data[3*floor_x +mm];
                                ja1[mm] = row_data[3*ceil_x +mm];
                            }

                        }
                        {
                            size_t dcolPitch= ceil_y*dpitch;
                            float * row_data= (float *)(slice_d+ dcolPitch);
                            for(int mm=0;mm<3;mm++)
                            {
                                ia2[mm] = row_data[3*floor_x +mm];
                                ja2[mm] = row_data[3*ceil_x +mm];
                            }
                        }
                    }
                    {
                        size_t dslicePitch= dpitch*d_sz[1]*ceil_z;
                        char * slice_d= d_ptr+  dslicePitch;
                        {
                            size_t dcolPitch= floor_y*dpitch;
                            float * row_data= (float *)(slice_d+ dcolPitch);
                            for(int mm=0;mm<3;mm++)
                            {
                                ib1[mm] = row_data[3*floor_x +mm];
                                jb1[mm] = row_data[3*ceil_x +mm];
                            }
                        }
                        {
                            size_t dcolPitch= ceil_y*dpitch;
                            float * row_data= (float *)(slice_d+ dcolPitch);
                            for(int mm=0;mm<3;mm++)
                            {
                                ib2[mm] = row_data[3*floor_x +mm];
                                jb2[mm] = row_data[3*ceil_x +mm];
                            }
                        }
                    }

                    for(int mm=0;mm<3;mm++)
                    {
                        float i1= ia1[mm]*(1-zd) + ib1[mm]*zd;
                        float i2= ia2[mm]*(1-zd) + ib2[mm]*zd;
                        float j1= ja1[mm]*(1-zd) + jb1[mm]*zd;
                        float j2= ja2[mm]*(1-zd) + jb2[mm]*zd;

                        float w1= i1*(1-yd) + i2*yd;
                        float w2= j1*(1-yd) + j2*yd;

                        float update= w1*(1-xd) + w2*xd;
                        row_o[3*i+mm]= xp[mm] +update -  x[mm];

                    }
                }
            }
    }

}




void ComposeFields_cuda(cudaPitchedPtr main_field,cudaPitchedPtr update_field,
             int3 data_sz,float3 data_spc,
             float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
             float3 data_orig,
             cudaPitchedPtr output )
{


    float h_d_dir[]= {data_d00,data_d01,data_d02,data_d10,data_d11,data_d12,data_d20,data_d21,data_d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));

    float h_d_orig[]= {data_orig.x,data_orig.y,data_orig.z};
    gpuErrchk(cudaMemcpyToSymbol(d_orig, &h_d_orig, 3 * sizeof(float)));

    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));

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

    ComposeFields_kernel<<< blockSize,gridSize>>>( main_field,update_field,output );


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}




__global__ void
NegateImage_kernel(cudaPitchedPtr image,  const int3 d_sz, const int Ncomponents)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t opitch= image.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;

                char *u_ptr= (char *)(image.ptr);
                char * slice_u= u_ptr+  oslicePitch;
                float * row_data= (float *)(slice_u+ ocolPitch);


                for(int m=0;m<Ncomponents;m++)
                    row_data[Ncomponents*i+m]=-row_data[Ncomponents*i+m];
           }
    }
}

__global__ void
ComputeFieldLocalNormImage(cudaPitchedPtr field,const int3 d_sz,const float3 data_spc,cudaPitchedPtr scale_image)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t opitch= scale_image.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;

                char *u_ptr= (char *)(scale_image.ptr);
                char * slice_u= u_ptr+  oslicePitch;
                float * row_out= (float *)(slice_u+ ocolPitch);


                size_t fpitch= field.pitch;
                size_t fslicePitch= fpitch*d_sz.y*k;
                size_t fcolPitch= j*fpitch;

                char *f_ptr= (char *)(field.ptr);
                char * slice_f= f_ptr+  fslicePitch;
                float * row_data= (float *)(slice_f+ fcolPitch);

                float nrm = row_data[3*i]*row_data[3*i]/data_spc.x/data_spc.x +
                            row_data[3*i+1]*row_data[3*i+1]/data_spc.y/data_spc.y +
                            row_data[3*i+2]*row_data[3*i+2]/data_spc.z/data_spc.z ;
                nrm=sqrt(nrm);

                row_out[i]=nrm;
           }
    }

}

__global__ void
UpdateInvertField_kernel( cudaPitchedPtr composed_field, cudaPitchedPtr scale_image, cudaPitchedPtr output, int3 d_sz, float3 data_spc , float m_Epsilon, float m_MaxErrorNorm)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t cpitch= composed_field.pitch;
                size_t cslicePitch= cpitch*d_sz.y*k;
                size_t ccolPitch= j*cpitch;
                char *c_ptr= (char *)(composed_field.ptr);
                char * slice_c= c_ptr+  cslicePitch;
                float * row_composed_field= (float *)(slice_c+ ccolPitch);


                size_t spitch= scale_image.pitch;
                size_t sslicePitch= spitch*d_sz.y*k;
                size_t scolPitch= j*spitch;
                char *s_ptr= (char *)(scale_image.ptr);
                char * slice_s= s_ptr+  sslicePitch;
                float * row_scale_image= (float *)(slice_s+ scolPitch);

                size_t opitch= output.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;
                char *o_ptr= (char *)(output.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_output= (float *)(slice_o+ ocolPitch);

                float3 update;
                update.x = row_composed_field[3*i];
                update.y = row_composed_field[3*i+1];
                update.z = row_composed_field[3*i+2];


                float scaledNorm= row_scale_image[i];
                if (scaledNorm > m_Epsilon * m_MaxErrorNorm)
                {
                       update.x = update.x * (m_Epsilon * m_MaxErrorNorm / scaledNorm);
                       update.y = update.y * (m_Epsilon * m_MaxErrorNorm / scaledNorm);
                       update.z = update.z * (m_Epsilon * m_MaxErrorNorm / scaledNorm);
                }

                update.x = row_output[3*i]  +  update.x *m_Epsilon;
                update.y = row_output[3*i+1]  + update.y *m_Epsilon;
                update.z = row_output[3*i+2]  + update.z *m_Epsilon;


                row_output[3*i]=update.x;
                row_output[3*i+1]=update.y;
                row_output[3*i+2]=update.z;


                if(i==0 || i==d_sz.x-1 || j==0 || j==d_sz.y-1 || k==0 || k==d_sz.z-1 )
                {
                    row_output[3*i]=0;
                    row_output[3*i+1]=0;
                    row_output[3*i+2]=0;
                }
            }
    }
}

void  NegateField_cuda(cudaPitchedPtr field, const int3 data_sz)
{
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    }
    
    NegateImage_kernel<<< blockSize,gridSize>>>(field,data_sz,3);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}




__global__ void
AddImages_kernel(cudaPitchedPtr im1, cudaPitchedPtr im2,cudaPitchedPtr output,  const int3 d_sz, const int Ncomponents)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t im1pitch= im1.pitch;
                size_t im1slicePitch= im1pitch*d_sz.y*k;
                size_t im1colPitch= j*im1pitch;
                char *im1_ptr= (char *)(im1.ptr);
                char * slice_im1= im1_ptr+  im1slicePitch;
                float * row_im1= (float *)(slice_im1+ im1colPitch);

                size_t im2pitch= im2.pitch;
                size_t im2slicePitch= im2pitch*d_sz.y*k;
                size_t im2colPitch= j*im2pitch;
                char *im2_ptr= (char *)(im2.ptr);
                char * slice_im2= im2_ptr+  im2slicePitch;
                float * row_im2= (float *)(slice_im2+ im2colPitch);

                size_t opitch= output.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;
                char *o_ptr= (char *)(output.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_o= (float *)(slice_o+ ocolPitch);


                for(int m=0;m<Ncomponents;m++)
                    row_o[Ncomponents*i+m]= row_im1[Ncomponents*i+m] + row_im2[Ncomponents*i+m];
           }
    }
}

void  AddImages_cuda(cudaPitchedPtr im1, cudaPitchedPtr im2, cudaPitchedPtr output, const int3 data_sz,const int ncomp)
{
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    }

    AddImages_kernel<<< blockSize,gridSize>>>(im1,im2,output,data_sz,ncomp);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void
MultiplyImage_kernel(cudaPitchedPtr im1, float factor,cudaPitchedPtr output,  const int3 d_sz, const int Ncomponents)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t im1pitch= im1.pitch;
                size_t im1slicePitch= im1pitch*d_sz.y*k;
                size_t im1colPitch= j*im1pitch;
                char *im1_ptr= (char *)(im1.ptr);
                char * slice_im1= im1_ptr+  im1slicePitch;
                float * row_im1= (float *)(slice_im1+ im1colPitch);

                size_t opitch= output.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;
                char *o_ptr= (char *)(output.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_o= (float *)(slice_o+ ocolPitch);


                for(int m=0;m<Ncomponents;m++)
                    row_o[Ncomponents*i+m]= row_im1[Ncomponents*i+m] *factor;
           }
    }
}


void  MultiplyImage_cuda(cudaPitchedPtr im1, float factor, cudaPitchedPtr d_output, const int3 data_sz,const int ncomp)
{
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    }

    MultiplyImage_kernel<<< blockSize,gridSize>>>(im1,factor,d_output,data_sz,ncomp);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}



void InvertField_cuda(cudaPitchedPtr field, const int3 data_sz,const float3 data_spc,
                      float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                      float3 data_orig,
                      cudaPitchedPtr output)
{
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    }


    cudaPitchedPtr scale_image={0};
    cudaExtent extent =  make_cudaExtent(1*sizeof(float)*data_sz.x,data_sz.y,data_sz.z);
    cudaMalloc3D(&scale_image, extent);
    cudaMemset3D(scale_image,0,extent);

    cudaPitchedPtr composed_field={0};
    cudaExtent extent2 =  make_cudaExtent(3*sizeof(float)*data_sz.x,data_sz.y,data_sz.z);
    cudaMalloc3D(&composed_field, extent2);
    cudaMemset3D(composed_field,0,extent2);



    int numberOfPixelsInRegion= data_sz.x  * data_sz.y * data_sz.z;


    //const float m_MaxErrorToleranceThreshold=0.08;
    //const float m_MeanErrorToleranceThreshold=0.0008;
    //const int Niter=20;

    const float m_MaxErrorToleranceThreshold=0.0005;
    const float m_MeanErrorToleranceThreshold=0.00005;
    const int Niter=200;


//    const float m_MaxErrorToleranceThreshold=0.03;
//    const float m_MeanErrorToleranceThreshold=0.0003;
//    const int Niter=200;

    float m_MaxErrorNorm = 1E10;
    float m_MeanErrorNorm = 1E10;

    int iteration=0;

    while (iteration++ < Niter && m_MaxErrorNorm > m_MaxErrorToleranceThreshold &&m_MeanErrorNorm > m_MeanErrorToleranceThreshold)
    {       
        ComposeFields_cuda(output,field,
                     data_sz, data_spc,
                     data_d00,  data_d01, data_d02, data_d10, data_d11, data_d12, data_d20, data_d21, data_d22,
                     data_orig,
                     composed_field );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        ComputeFieldLocalNormImage<<< blockSize,gridSize>>>(composed_field,data_sz,data_spc,scale_image);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        NegateImage_kernel<<< blockSize,gridSize>>>(composed_field,data_sz,3);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        {
            float* dev_out;
            float out;
            cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

            ScalarFindMax<<<gSize, bSize>>>((float *)scale_image.ptr, scale_image.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
            cudaDeviceSynchronize();
            ScalarFindMax<<<1, bSize>>>(dev_out, gSize, dev_out);
            cudaDeviceSynchronize();

            cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(dev_out);
            m_MaxErrorNorm=out;
        }

        {
            float* dev_out;
            float out;
            cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

            ScalarFindSum<<<gSize, bSize>>>((float *)scale_image.ptr, scale_image.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
            cudaDeviceSynchronize();
            ScalarFindSum<<<1, bSize>>>(dev_out, gSize, dev_out);
            cudaDeviceSynchronize();

            cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(dev_out);
            m_MeanErrorNorm=out/numberOfPixelsInRegion;
        }


        float m_Epsilon = 0.5;
        if (iteration == 1)
        {
          m_Epsilon = 0.75;
        }

        UpdateInvertField_kernel<<< blockSize,gridSize>>>(  composed_field, scale_image, output, data_sz, data_spc , m_Epsilon,m_MaxErrorNorm);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


    }
    cudaFree(scale_image.ptr);
    cudaFree(composed_field.ptr);

}



__global__ void
PreprocessImage_kernel(cudaPitchedPtr img,  const int3 d_sz,
                       float img_min, float img_max,
                       float low_val, float up_val,
                       cudaPitchedPtr output)

{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t pitch= img.pitch;
                size_t slicePitch= pitch*d_sz.y*k;
                size_t colPitch= j*pitch;

                char *img_ptr= (char *)(img.ptr);
                char * slice_img= img_ptr+  slicePitch;
                float * row_data= (float *)(slice_img+ colPitch);

                char *output_ptr= (char *)(output.ptr);
                char * slice_o= output_ptr+  slicePitch;
                float * row_output= (float *)(slice_o+ colPitch);

                row_output[i]= (up_val-low_val)/(img_max-img_min)*row_data[i]
                                - img_min*(up_val-low_val)/(img_max-img_min)
                               +low_val;

           }
    }
}


void PreprocessImage_cuda(cudaPitchedPtr img,
                      int3 data_sz,
                      float low_val, float up_val,
                      cudaPitchedPtr output)
{

    float max, min;
    {
        float* dev_out;
        float out;
        cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

        ScalarFindMax<<<gSize, bSize>>>((float *)img.ptr, img.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
        cudaDeviceSynchronize();
        ScalarFindMax<<<1, bSize>>>(dev_out, gSize, dev_out);
        cudaDeviceSynchronize();

        cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_out);
        max=out;
    }
    {
        float* dev_out;
        float out;
        cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

        ScalarFindMin<<<gSize, bSize>>>((float *)img.ptr, img.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
        cudaDeviceSynchronize();
        ScalarFindMin<<<1, bSize>>>(dev_out, gSize, dev_out);
        cudaDeviceSynchronize();

        cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_out);
        min=out;
    }

    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize.x=std::ceil(1.*data_sz.x / blockSize.x);
        gridSize.y=std::ceil(1.*data_sz.y / blockSize.y);
        gridSize.z=std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) ;
    }

    PreprocessImage_kernel<<< blockSize,gridSize>>>( img,data_sz,min,max , low_val,up_val,output );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

__device__ float3 ComputeImageGradientf(cudaPitchedPtr img,int i, int j, int k)
{
    float3 grad;
    grad.x=0;
    grad.y=0;
    grad.z=0;    

    if(i==0 || i== d_sz[0]-1 || j==0 || j== d_sz[1]-1 || k==0 || k== d_sz[2]-1 )
        return grad;

    size_t pitch= img.pitch;
    char *ptr= (char *)(img.ptr);

    {
        size_t slicePitch= pitch*d_sz[1]*k;
        char * slice= ptr+  slicePitch;
        {
            size_t colPitch= j*pitch;
            float * row= (float *)(slice+ colPitch);
            grad.x= 0.5*(row[i+1]-row[i-1])/d_spc[0];
        }
        {
            size_t colPitch= (j+1)*pitch;
            float * row= (float *)(slice+ colPitch);
            grad.y= row[i];

            colPitch= (j-1)*pitch;
            row= (float *)(slice+ colPitch);
            grad.y= 0.5*(grad.y- row[i])/d_spc[1];
        }
    }


    {
        size_t slicePitch= pitch*d_sz[1]*(k+1);
        char * slice= ptr+  slicePitch;
        size_t colPitch= j*pitch;
        float * row= (float *)(slice+ colPitch);
        grad.z= row[i];

        slicePitch= pitch*d_sz[1]*(k-1);
        slice= ptr+  slicePitch;
        row= (float *)(slice+ colPitch);
        grad.z= 0.5*(grad.z -row[i])/d_spc[2];
    }

    float3 grad2;

    grad2.x= d_dir[0]*grad.x + d_dir[1]*grad.y + d_dir[2]*grad.z;
    grad2.y= d_dir[3]*grad.x + d_dir[4]*grad.y + d_dir[5]*grad.z;
    grad2.z= d_dir[6]*grad.x + d_dir[7]*grad.y + d_dir[8]*grad.z;

    return grad2;
}




__global__ void
ComputeImageGradient_kernel(cudaPitchedPtr img,  const int3 d_sz,
                           cudaPitchedPtr outputx,cudaPitchedPtr outputy,cudaPitchedPtr outputz)

{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t xpitch= outputx.pitch;
                size_t xslicePitch= xpitch*d_sz.y*k;
                size_t xcolPitch= j*xpitch;
                char *xoutput_ptr= (char *)(outputx.ptr);
                char * slice_x= xoutput_ptr+  xslicePitch;
                float * row_outputx= (float *)(slice_x+ xcolPitch);

                size_t ypitch= outputy.pitch;
                size_t yslicePitch= ypitch*d_sz.y*k;
                size_t ycolPitch= j*ypitch;
                char *youtput_ptr= (char *)(outputy.ptr);
                char * slice_y= youtput_ptr+  yslicePitch;
                float * row_outputy= (float *)(slice_y+ ycolPitch);

                size_t zpitch= outputz.pitch;
                size_t zslicePitch= zpitch*d_sz.y*k;
                size_t zcolPitch= j*zpitch;
                char *zoutput_ptr= (char *)(outputz.ptr);
                char * slice_z= zoutput_ptr+  zslicePitch;
                float * row_outputz= (float *)(slice_z+zcolPitch);


                 
                float3 grad = ComputeImageGradientf(img, i,  j, k);

                row_outputx[i]=  grad.x;
                row_outputy[i]=  grad.y;
                row_outputz[i]=  grad.z;

           }
    }
}



void ComputeImageGradient_cuda(cudaPitchedPtr img,
                      int3 data_sz, float3 data_spc,
                               float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                      cudaPitchedPtr outputx,cudaPitchedPtr outputy, cudaPitchedPtr outputz)
{
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));

    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));

    float h_d_dir[]= {data_d00,data_d01,data_d02,data_d10,data_d11,data_d12,data_d20,data_d21,data_d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));
    
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    }

    ComputeImageGradient_kernel<<< blockSize,gridSize>>>( img,data_sz,outputx,outputy,outputz );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


__device__ float3 InterpolateVF(cudaPitchedPtr *vf,float *xp,int NT)
{
    float3 disp={0,0,0};


    float iw = (d_dir[0]*xp[0]  + d_dir[3]*xp[1]  + d_dir[6]*xp[2])/ d_spc[0] ;
    float jw = (d_dir[1]*xp[0]  + d_dir[4]*xp[1]  + d_dir[7]*xp[2])/ d_spc[1] ;
    float kw = (d_dir[2]*xp[0]  + d_dir[5]*xp[1]  + d_dir[8]*xp[2])/ d_spc[2] ;
    float tw = xp[3];


    if(  !(iw<0 || iw> d_sz[0]-1 || jw<0 || jw> d_sz[1]-1 || kw<0 || kw> d_sz[2]-1 || tw<0  || tw>NT-1))
    {
        int floor_x = __float2int_rd(iw);
        int floor_y = __float2int_rd(jw);
        int floor_z = __float2int_rd(kw);
        int floor_t = __float2int_rd(tw);



        int ceil_x = __float2int_ru(iw);
        int ceil_y = __float2int_ru(jw);
        int ceil_z = __float2int_ru(kw);
        int ceil_t = __float2int_ru(tw);



        float xd= iw - floor_x;
        float yd= jw - floor_y;
        float zd= kw - floor_z;
        float td= tw - floor_t;



        float ia1[3],ib1[3],ia2[3],ib2[3], ja1[3],jb1[3],ja2[3],jb2[3];
        float Tia1[3],Tib1[3],Tia2[3],Tib2[3], Tja1[3],Tjb1[3],Tja2[3],Tjb2[3];

        {
            cudaPitchedPtr update_field =vf[floor_t];



            size_t dpitch= update_field.pitch;
            char *d_ptr= (char *)(update_field.ptr);
            {
                size_t dslicePitch= dpitch*d_sz[1]*floor_z;
                char * slice_d= d_ptr+  dslicePitch;
                {
                    size_t dcolPitch= floor_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        ia1[mm] = row_data[3*floor_x +mm];
                        ja1[mm] = row_data[3*ceil_x +mm];
                    }

                }
                {
                    size_t dcolPitch= ceil_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        ia2[mm] = row_data[3*floor_x +mm];
                        ja2[mm] = row_data[3*ceil_x +mm];
                    }
                }
            }
            {
                size_t dslicePitch= dpitch*d_sz[1]*ceil_z;
                char * slice_d= d_ptr+  dslicePitch;
                {
                    size_t dcolPitch= floor_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        ib1[mm] = row_data[3*floor_x +mm];
                        jb1[mm] = row_data[3*ceil_x +mm];
                    }
                }
                {
                    size_t dcolPitch= ceil_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        ib2[mm] = row_data[3*floor_x +mm];
                        jb2[mm] = row_data[3*ceil_x +mm];
                    }
                }
            }
        }
        {
            cudaPitchedPtr update_field =vf[ceil_t];
            size_t dpitch= update_field.pitch;
            char *d_ptr= (char *)(update_field.ptr);
            {
                size_t dslicePitch= dpitch*d_sz[1]*floor_z;
                char * slice_d= d_ptr+  dslicePitch;
                {
                    size_t dcolPitch= floor_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        Tia1[mm] = row_data[3*floor_x +mm];
                        Tja1[mm] = row_data[3*ceil_x +mm];
                    }

                }
                {
                    size_t dcolPitch= ceil_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        Tia2[mm] = row_data[3*floor_x +mm];
                        Tja2[mm] = row_data[3*ceil_x +mm];
                    }
                }
            }
            {
                size_t dslicePitch= dpitch*d_sz[1]*ceil_z;
                char * slice_d= d_ptr+  dslicePitch;
                {
                    size_t dcolPitch= floor_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        Tib1[mm] = row_data[3*floor_x +mm];
                        Tjb1[mm] = row_data[3*ceil_x +mm];
                    }
                }
                {
                    size_t dcolPitch= ceil_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        Tib2[mm] = row_data[3*floor_x +mm];
                        Tjb2[mm] = row_data[3*ceil_x +mm];
                    }
                }
            }
        }

        float update[3];
        for(int mm=0;mm<3;mm++)
        {
            float T1= ia1[mm]*(1-td) + Tia1[mm]*td;
            float T2= ia2[mm]*(1-td) + Tia2[mm]*td;
            float T3= ib1[mm]*(1-td) + Tib1[mm]*td;
            float T4= ib2[mm]*(1-td) + Tib2[mm]*td;
            float T5= ja1[mm]*(1-td) + Tja1[mm]*td;
            float T6= ja2[mm]*(1-td) + Tja2[mm]*td;
            float T7= jb1[mm]*(1-td) + Tjb1[mm]*td;
            float T8= jb2[mm]*(1-td) + Tjb2[mm]*td;


            float i1= T1*(1-zd) + T3*zd;
            float i2= T2*(1-zd) + T4*zd;
            float j1= T5*(1-zd) + T7*zd;
            float j2= T6*(1-zd) + T8*zd;

            float w1= i1*(1-yd) + i2*yd;
            float w2= j1*(1-yd) + j2*yd;

            update[mm]= w1*(1-xd) + w2*xd;
        }
        disp.x=update[0];
        disp.y=update[1];
        disp.z=update[2];
    }

    return disp;
}


__device__ float3 InterpolateVFAtT(cudaPitchedPtr *vf,float *xp, int T, int NT)
{
    float3 disp={0,0,0};


    float iw = (d_dir[0]*xp[0]  + d_dir[3]*xp[1]  + d_dir[6]*xp[2])/ d_spc[0] ;
    float jw = (d_dir[1]*xp[0]  + d_dir[4]*xp[1]  + d_dir[7]*xp[2])/ d_spc[1] ;
    float kw = (d_dir[2]*xp[0]  + d_dir[5]*xp[1]  + d_dir[8]*xp[2])/ d_spc[2] ;


    if(  !(iw<0 || iw> d_sz[0]-1 || jw<0 || jw> d_sz[1]-1 || kw<0 || kw> d_sz[2]-1 ))
    {
        int floor_x = __float2int_rd(iw);
        int floor_y = __float2int_rd(jw);
        int floor_z = __float2int_rd(kw);

        int ceil_x = __float2int_ru(iw);
        int ceil_y = __float2int_ru(jw);
        int ceil_z = __float2int_ru(kw);


        float xd= iw - floor_x;
        float yd= jw - floor_y;
        float zd= kw - floor_z;



        float ia1[3],ib1[3],ia2[3],ib2[3], ja1[3],jb1[3],ja2[3],jb2[3];


        {
            cudaPitchedPtr update_field =vf[T];


            size_t dpitch= update_field.pitch;
            char *d_ptr= (char *)(update_field.ptr);
            {
                size_t dslicePitch= dpitch*d_sz[1]*floor_z;
                char * slice_d= d_ptr+  dslicePitch;
                {
                    size_t dcolPitch= floor_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        ia1[mm] = row_data[3*floor_x +mm];
                        ja1[mm] = row_data[3*ceil_x +mm];
                    }

                }
                {
                    size_t dcolPitch= ceil_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        ia2[mm] = row_data[3*floor_x +mm];
                        ja2[mm] = row_data[3*ceil_x +mm];
                    }
                }
            }
            {
                size_t dslicePitch= dpitch*d_sz[1]*ceil_z;
                char * slice_d= d_ptr+  dslicePitch;
                {
                    size_t dcolPitch= floor_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        ib1[mm] = row_data[3*floor_x +mm];
                        jb1[mm] = row_data[3*ceil_x +mm];
                    }
                }
                {
                    size_t dcolPitch= ceil_y*dpitch;
                    float * row_data= (float *)(slice_d+ dcolPitch);
                    for(int mm=0;mm<3;mm++)
                    {
                        ib2[mm] = row_data[3*floor_x +mm];
                        jb2[mm] = row_data[3*ceil_x +mm];
                    }
                }
            }
        }



        float update[3];
        for(int mm=0;mm<3;mm++)
        {
            float i1= ia1[mm]*(1-zd) + ib1[mm]*zd;
            float i2= ia2[mm]*(1-zd) + ib2[mm]*zd;
            float j1= ja1[mm]*(1-zd) + jb1[mm]*zd;
            float j2= ja2[mm]*(1-zd) + jb2[mm]*zd;

            float w1= i1*(1-yd) + i2*yd;
            float w2= j1*(1-yd) + j2*yd;

            update[mm]= w1*(1-xd) + w2*xd;
        }
        disp.x=update[0];
        disp.y=update[1];
        disp.z=update[2];
    }

    return disp;
}


__global__ void
IntegrateVelocityField_kernel(cudaPitchedPtr *vf, cudaPitchedPtr output_field, float lowt,float hight, int NT)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            float x[3],disp[3];
            x[0]= (d_dir[0]*i  + d_dir[1]*j + d_dir[2]*k)* d_spc[0] ;
            x[1]= (d_dir[3]*i  + d_dir[4]*j + d_dir[5]*k)* d_spc[1] ;
            x[2]= (d_dir[6]*i  + d_dir[7]*j + d_dir[8]*k)* d_spc[2] ;
            disp[0]=0;
            disp[1]=0;
            disp[2]=0;


            float timeScale= NT-1;
            float timePointInImage = lowt*timeScale;
            const float deltaTime = (hight - lowt)/(NT+2.);
            float deltaTimeInImage = timeScale*deltaTime;



            for (unsigned int n = 0; n < NT+2; ++n)
            {
                float x1[4],x2[4],x3[4],x4[4];
                for(int d=0;d<3;d++)
                {
                    x1[d]=x[d]+disp[d];
                    x2[d]=x1[d];
                    x3[d]=x1[d];
                    x4[d]=x1[d];
                }
                x1[3]= timePointInImage;
                x2[3]= timePointInImage + 0.5*deltaTimeInImage;
                x3[3]= timePointInImage + 0.5*deltaTimeInImage;
                x4[3]= timePointInImage + deltaTimeInImage;



                float3 f1= InterpolateVF(vf,x1,NT);
                x2[0]+=f1.x * deltaTime*0.5;
                x2[1]+=f1.y * deltaTime*0.5;
                x2[2]+=f1.z * deltaTime*0.5;


                float3 f2= InterpolateVF(vf,x2,NT);
                x3[0]+=f2.x * deltaTime*0.5;
                x3[1]+=f2.y * deltaTime*0.5;
                x3[2]+=f2.z * deltaTime*0.5;

                float3 f3= InterpolateVF(vf,x3,NT);
                x4[0]+=f3.x * deltaTime;
                x4[1]+=f3.y * deltaTime;
                x4[2]+=f3.z * deltaTime;

                float3 f4= InterpolateVF(vf,x4,NT);

                x1[0]+= deltaTime / 6. * (f1.x + 2.0 * f2.x + 2.0 * f3.x + f4.x);
                x1[1]+= deltaTime / 6. * (f1.y + 2.0 * f2.y + 2.0 * f3.y + f4.y);
                x1[2]+= deltaTime / 6. * (f1.z + 2.0 * f2.z + 2.0 * f3.z + f4.z);
                disp[0]= x1[0]- x[0];
                disp[1]= x1[1]- x[1];
                disp[2]= x1[2]- x[2];


                timePointInImage += deltaTimeInImage;

            }

            size_t opitch= output_field.pitch;
            size_t oslicePitch= opitch*d_sz[1]*k;
            size_t ocolPitch= j*opitch;
            char *o_ptr= (char *)(output_field.ptr);
            char * slice_o= o_ptr+  oslicePitch;
            float * row_o= (float *)(slice_o+ ocolPitch);

            row_o[3*i]= disp[0];
            row_o[3*i+1]= disp[1];
            row_o[3*i+2]= disp[2];

        }
    }
}


void IntegrateVelocityField_cuda(cudaPitchedPtr *velocity_field,
                                 cudaPitchedPtr output_field,
                                 float lowt, float hight,
                                 int NT,
                                 int3 data_sz,float3 data_spc,
                                 float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                                 float3 data_orig)
{


    float h_d_dir[]= {data_d00,data_d01,data_d02,data_d10,data_d11,data_d12,data_d20,data_d21,data_d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));

    float h_d_orig[]= {data_orig.x,data_orig.y,data_orig.z};
    gpuErrchk(cudaMemcpyToSymbol(d_orig, &h_d_orig, 3 * sizeof(float)));

    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));

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

    IntegrateVelocityField_kernel<<< blockSize,gridSize>>>( velocity_field,output_field, lowt,hight, NT );


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}











__global__ void
ConstrainVelocityFields_kernel(cudaPitchedPtr *vfield, cudaPitchedPtr *new_vfield, cudaPitchedPtr ufield,  int NT)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;


    int halfT= NT/2;
    float deltaT= 1./(NT-1);

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {

            float x[3], x0[3];
            x[0]= (d_dir[0]*i  + d_dir[1]*j + d_dir[2]*k)* d_spc[0] ;
            x[1]= (d_dir[3]*i  + d_dir[4]*j + d_dir[5]*k)* d_spc[1] ;
            x[2]= (d_dir[6]*i  + d_dir[7]*j + d_dir[8]*k)* d_spc[2] ;

            size_t upitch= ufield.pitch;
            size_t uslicePitch= upitch*d_sz[1]*k;
            size_t ucolPitch= j*upitch;
            char *u_ptr= (char *)(ufield.ptr);
            char * slice_u= u_ptr+  uslicePitch;
            float * row_u= (float *)(slice_u+ ucolPitch);


            size_t vpitch= vfield[halfT].pitch;
            size_t vslicePitch= vpitch*d_sz[1]*k;
            size_t vcolPitch= j*vpitch;
            char *v_ptr= (char *)(vfield[halfT].ptr);
            char * slice_v= v_ptr+  vslicePitch;
            float * row_v= (float *)(slice_v+ vcolPitch);


            x0[0]= x[0] + row_u[3*i+0];
            x0[1]= x[1] + row_u[3*i+1];
            x0[2]= x[2] + row_u[3*i+2];

            for(int T=0;T<halfT;T++)
            {

                float curru[3];
                curru[0]=x0[0];
                curru[1]=x0[1];
                curru[2]=x0[2];
                for(int T2=0;T2<T;T2++)
                {
                    float3 udisp= InterpolateVFAtT(vfield,curru,T2,NT);
                    curru[0]+= udisp.x * deltaT;
                    curru[1]+= udisp.y * deltaT;
                    curru[2]+= udisp.z * deltaT;
                }
                float3 udisp= InterpolateVFAtT(vfield,curru,T,NT);



                float currd[3];
                currd[0]=x[0] + deltaT * row_v[3*i+0];
                currd[1]=x[1] + deltaT * row_v[3*i+1];
                currd[2]=x[2] + deltaT * row_v[3*i+2];
                for(int T2=0;T2<T;T2++)
                {
                    float3 ddisp= InterpolateVFAtT(vfield,currd,T2+halfT+1,NT);
                    currd[0]+= ddisp.x * deltaT;
                    currd[1]+= ddisp.y * deltaT;
                    currd[2]+= ddisp.z * deltaT;
                }
                float3 ddisp= InterpolateVFAtT(vfield,currd,T+halfT+1,NT);

                float3 disp;
                disp.x= (udisp.x + ddisp.x)/2.;
                disp.y= (udisp.y + ddisp.y)/2.;
                disp.z= (udisp.z + ddisp.z)/2.;


                int iuw = round((d_dir[0]*curru[0]  + d_dir[3]*curru[1]  + d_dir[6]*curru[2])/ d_spc[0]) ;
                int juw = round((d_dir[1]*curru[0]  + d_dir[4]*curru[1]  + d_dir[7]*curru[2])/ d_spc[1]) ;
                int kuw = round((d_dir[2]*curru[0]  + d_dir[5]*curru[1]  + d_dir[8]*curru[2])/ d_spc[2]) ;

                int idw = round((d_dir[0]*currd[0]  + d_dir[3]*currd[1]  + d_dir[6]*currd[2])/ d_spc[0]) ;
                int jdw = round((d_dir[1]*currd[0]  + d_dir[4]*currd[1]  + d_dir[7]*currd[2])/ d_spc[1]) ;
                int kdw = round((d_dir[2]*currd[0]  + d_dir[5]*currd[1]  + d_dir[8]*currd[2])/ d_spc[2]) ;


                size_t vupitch= new_vfield[T].pitch;
                size_t vuslicePitch= vupitch*d_sz[1]*kuw;
                size_t vucolPitch= juw*vupitch;
                char *vu_ptr= (char *)(new_vfield[T].ptr);
                char * slice_vu= vu_ptr+  vuslicePitch;
                float * row_vu= (float *)(slice_vu+ vucolPitch);

                size_t vdpitch= new_vfield[halfT+T+1].pitch;
                size_t vdslicePitch= vdpitch*d_sz[1]*kdw;
                size_t vdcolPitch= jdw*vdpitch;
                char *vd_ptr= (char *)(new_vfield[halfT+T+1].ptr);
                char * slice_vd= vd_ptr+  vdslicePitch;
                float * row_vd= (float *)(slice_vd+ vdcolPitch);

                row_vu[3*iuw+0] = disp.x;
                row_vu[3*iuw+1] = disp.y;
                row_vu[3*iuw+2] = disp.z;

                row_vd[3*idw+0] = disp.x;
                row_vd[3*idw+1] = disp.y;
                row_vd[3*idw+2] = disp.z;
            }
        }
    }
}





void ContrainVelocityFields_cuda(cudaPitchedPtr *vfield,
                                 cudaPitchedPtr *new_vfield,
                                 cudaPitchedPtr ufield,
                                 int NT,
                                 int3 data_sz,float3 data_spc,
                                 float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                                 float3 data_orig)
{


    float h_d_dir[]= {data_d00,data_d01,data_d02,data_d10,data_d11,data_d12,data_d20,data_d21,data_d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));

    float h_d_orig[]= {data_orig.x,data_orig.y,data_orig.z};
    gpuErrchk(cudaMemcpyToSymbol(d_orig, &h_d_orig, 3 * sizeof(float)));

    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));

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

    ConstrainVelocityFields_kernel<<< blockSize,gridSize>>>( vfield,new_vfield, ufield,  NT );


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}





__global__ void
DivideImages_kernel(cudaPitchedPtr im1, cudaPitchedPtr im2,cudaPitchedPtr output,  const int3 d_sz, const int Ncomponents)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t im1pitch= im1.pitch;
                size_t im1slicePitch= im1pitch*d_sz.y*k;
                size_t im1colPitch= j*im1pitch;
                char *im1_ptr= (char *)(im1.ptr);
                char * slice_im1= im1_ptr+  im1slicePitch;
                float * row_im1= (float *)(slice_im1+ im1colPitch);

                size_t im2pitch= im2.pitch;
                size_t im2slicePitch= im2pitch*d_sz.y*k;
                size_t im2colPitch= j*im2pitch;
                char *im2_ptr= (char *)(im2.ptr);
                char * slice_im2= im2_ptr+  im2slicePitch;
                float * row_im2= (float *)(slice_im2+ im2colPitch);

                size_t opitch= output.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;
                char *o_ptr= (char *)(output.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_o= (float *)(slice_o+ ocolPitch);


                for(int m=0;m<Ncomponents;m++)
                {
                    if(row_im2[Ncomponents*i+m]!=0)
                        row_o[Ncomponents*i+m]= row_im1[Ncomponents*i+m] / row_im2[Ncomponents*i+m];
                }
           }
    }
}


void  DivideImages_cuda(cudaPitchedPtr im1,cudaPitchedPtr im2, cudaPitchedPtr d_output, const int3 data_sz,const int ncomp)
{
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    }

    DivideImages_kernel<<< blockSize,gridSize>>>(im1,im2,d_output,data_sz,ncomp);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}


__global__ void
MultiplyImages_kernel(cudaPitchedPtr im1, cudaPitchedPtr im2,cudaPitchedPtr output,  const int3 d_sz, const int Ncomponents)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t im1pitch= im1.pitch;
                size_t im1slicePitch= im1pitch*d_sz.y*k;
                size_t im1colPitch= j*im1pitch;
                char *im1_ptr= (char *)(im1.ptr);
                char * slice_im1= im1_ptr+  im1slicePitch;
                float * row_im1= (float *)(slice_im1+ im1colPitch);

                size_t im2pitch= im2.pitch;
                size_t im2slicePitch= im2pitch*d_sz.y*k;
                size_t im2colPitch= j*im2pitch;
                char *im2_ptr= (char *)(im2.ptr);
                char * slice_im2= im2_ptr+  im2slicePitch;
                float * row_im2= (float *)(slice_im2+ im2colPitch);

                size_t opitch= output.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;
                char *o_ptr= (char *)(output.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_o= (float *)(slice_o+ ocolPitch);


                for(int m=0;m<Ncomponents;m++)
                {
                        row_o[Ncomponents*i+m]= row_im1[Ncomponents*i+m] * row_im2[Ncomponents*i+m];
                }
           }
    }
}




void  MultiplyImages_cuda(cudaPitchedPtr im1,cudaPitchedPtr im2, cudaPitchedPtr d_output, const int3 data_sz,const int ncomp)
{
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    }

    MultiplyImages_kernel<<< blockSize,gridSize>>>(im1,im2,d_output,data_sz,ncomp);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}



__global__ void
ScaleImageForAnisotropicSmoothing_kernel(cudaPitchedPtr img, float scale, float add )
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            size_t im1pitch= img.pitch;
            size_t im1slicePitch= im1pitch*d_sz[1]*k;
            size_t im1colPitch= j*im1pitch;
            char *im1_ptr= (char *)(img.ptr);
            char * slice_im1= im1_ptr+  im1slicePitch;
            float * row_img= (float *)(slice_im1+ im1colPitch);

            row_img[i]= row_img[i] *scale + add;
        }
    }
}


__global__ void
ConcatImagesForAnisotropicSmoothing_kernel(cudaPitchedPtr field, cudaPitchedPtr TR_img, cudaPitchedPtr FA_img, cudaPitchedPtr concat_img )
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            size_t fieldpitch= field.pitch;
            size_t fieldslicePitch= fieldpitch*d_sz[1]*k;
            size_t fieldcolPitch= j*fieldpitch;
            char * field_ptr= (char *)(field.ptr);
            char * slice_field= field_ptr+  fieldslicePitch;
            float * row_field= (float *)(slice_field+ fieldcolPitch);

            size_t TR_imgpitch= TR_img.pitch;
            size_t TR_imgslicePitch= TR_imgpitch*d_sz[1]*k;
            size_t TR_imgcolPitch= j*TR_imgpitch;
            char * TR_img_ptr= (char *)(TR_img.ptr);
            char * slice_TR_img= TR_img_ptr+  TR_imgslicePitch;
            float * row_TR_img= (float *)(slice_TR_img+ TR_imgcolPitch);

            size_t FA_imgpitch= FA_img.pitch;
            size_t FA_imgslicePitch= FA_imgpitch*d_sz[1]*k;
            size_t FA_imgcolPitch= j*FA_imgpitch;
            char * FA_img_ptr= (char *)(FA_img.ptr);
            char * slice_FA_img= FA_img_ptr+  FA_imgslicePitch;
            float * row_FA_img= (float *)(slice_FA_img+ FA_imgcolPitch);

            size_t concat_imgpitch= concat_img.pitch;
            size_t concat_imgslicePitch= concat_imgpitch*d_sz[1]*k;
            size_t concat_imgcolPitch= j*concat_imgpitch;
            char * concat_img_ptr= (char *)(concat_img.ptr);
            char * slice_concat_img= concat_img_ptr+  concat_imgslicePitch;
            float * row_concat_img= (float *)(slice_concat_img+ concat_imgcolPitch);

            row_concat_img[i*5+0] = row_field[i*3+0];
            row_concat_img[i*5+1] = row_field[i*3+1];
            row_concat_img[i*5+2] = row_field[i*3+2];
            row_concat_img[i*5+3] = row_TR_img[i];
            row_concat_img[i*5+4] = row_FA_img[i];
        }
    }
}


__device__ double ComputeVectorImageGradientMagnitudeSquare(cudaPitchedPtr img,int i, int j, int k, int N)
{
    double grad_sq=0;


    //if(i==0 || i== d_sz[0]-1 || j==0 || j== d_sz[1]-1 || k==0 || k== d_sz[2]-1 )
      //  return 0;


    size_t pitch= img.pitch;
    char *ptr= (char *)(img.ptr);

    for(int V=0;V<N;V++)
    {
        {
            size_t slicePitch= pitch*d_sz[1]*k;
            char * slice= ptr+  slicePitch;
            size_t colPitch= j*pitch;
            float * row= (float *)(slice+ colPitch);

            float val=0;
            if(i!=0 && i!=d_sz[0]-1)
                val= 0.5*(row[(i+1)*N+V]-row[(i-1)*N+V]);
            else
            {
                if(i==0)
                    val= 0.5*(row[(i+1)*N+V]-row[(i)*N+V]);
                else
                    val= 0.5*(row[(i)*N+V]-row[(i-1)*N+V]);
            }
            grad_sq+= val*val;



            {
                size_t colPitch;
                if(j!=d_sz[1]-1)
                    colPitch= (j+1)*pitch;
                else
                    colPitch= (j)*pitch;
                float * row= (float *)(slice+ colPitch);
                float val= row[i*N+V];

                if(j!=0)
                    colPitch= (j-1)*pitch;
                else
                    colPitch= (j)*pitch;
                row= (float *)(slice+ colPitch);
                val= 0.5*(val- row[i*N+V]);
                grad_sq+= val*val;
            }
        }

        {
            size_t slicePitch;
            if(k!=d_sz[2]-1)
                slicePitch= pitch*d_sz[1]*(k+1);
            else
                slicePitch= pitch*d_sz[1]*(k);
            char * slice= ptr+  slicePitch;
            size_t colPitch= j*pitch;
            float * row= (float *)(slice+ colPitch);
            float val= row[i*N+V];

            if(k!=0)
                slicePitch= pitch*d_sz[1]*(k-1);
            else
                slicePitch= pitch*d_sz[1]*(k);
            slice= ptr+  slicePitch;
            row= (float *)(slice+ colPitch);
            val= 0.5*(val -row[i*N+V]);

            grad_sq+= val*val;
        }
    }
    return grad_sq;
}




__global__ void
ComputeAverageSquaredGradientImage_kernel(cudaPitchedPtr concat_img, cudaPitchedPtr grad_sq_img)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2] )
        {
            size_t grad_sq_imgpitch= grad_sq_img.pitch;
            size_t grad_sq_imgslicePitch= grad_sq_imgpitch*d_sz[1]*k;
            size_t grad_sq_imgcolPitch= j*grad_sq_imgpitch;
            char * grad_sq_img_ptr= (char *)(grad_sq_img.ptr);
            char * slice_grad_sq_img= grad_sq_img_ptr+  grad_sq_imgslicePitch;
            double * row_grad_sq_img= (double *)(slice_grad_sq_img+ grad_sq_imgcolPitch);

            double grad_sq= ComputeVectorImageGradientMagnitudeSquare(concat_img, i,  j,  k, 5);
            row_grad_sq_img[i]=grad_sq;
        }
    }
}







__global__ void
ComputeAnisotropicFilteringUpdate_kernel(cudaPitchedPtr concat_img, double *dev_mK, float *dev_deltaT, cudaPitchedPtr update_img)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        //if(i<d_sz[0]-1 && j <d_sz[1]-1 && k<d_sz[2]-1 && i>1 && j>1 && k>1)
        {
            size_t update_imgpitch= update_img.pitch;
            size_t update_imgslicePitch= update_imgpitch*d_sz[1]*k;
            size_t update_imgcolPitch= j*update_imgpitch;
            char * update_img_ptr= (char *)(update_img.ptr);
            char * slice_update_img= update_img_ptr+  update_imgslicePitch;
            float * row_update_img= (float *)(slice_update_img+ update_imgcolPitch);

            double m_K=dev_mK[0];
            float deltaT=dev_deltaT[0];


            const int N=5;
            float dx_forward[3][N];
            float dx_backward[3][N];
            float dx[3][N];


            size_t pitch= concat_img.pitch;
            char *ptr= (char *)(concat_img.ptr);

            int ip=i+1;
            int im=i-1;
            int jp=j+1;
            int jm=j-1;
            int kp=k+1;
            int km=k-1;

            if(ip>d_sz[0]-1)
                ip=d_sz[0]-1;
            if(im<0)
                im=0;
            if(jp>d_sz[1]-1)
                jp=d_sz[1]-1;
            if(jm<0)
                jm=0;
            if(kp>d_sz[2]-1)
                kp=d_sz[2]-1;
            if(km<0)
                km=0;


            for(int V=0;V<N;V++)
            {
                float val_c;
                {
                    size_t slicePitch= pitch*d_sz[1]*k;
                    char * slice= ptr+  slicePitch;
                    {
                        size_t colPitch= j*pitch;
                        float * row= (float *)(slice+ colPitch);
                        val_c= row[(i)*N+V] ;
                        float val_p= row[(ip)*N+V] ;
                        float val_m= row[(im)*N+V] ;

                        dx_forward[0][V]=  (val_p - val_c)/d_spc[0];
                        dx_backward[0][V]= (val_c - val_m)/d_spc[0];
                        dx[0][V]= 0.5*(val_p - val_m)/d_spc[0];

                    }
                    {
                        size_t colPitch= (jp)*pitch;
                        float * row= (float *)(slice+ colPitch);
                        float val_p= row[i*N+V];

                        colPitch= (jm)*pitch;
                        row= (float *)(slice+ colPitch);
                        float val_m= row[i*N+V];

                        dx_forward[1][V]=  (val_p - val_c)/d_spc[1];
                        dx_backward[1][V]= (val_c - val_m)/d_spc[1];
                        dx[1][V]= 0.5*(val_p - val_m)/d_spc[1];

                    }
                }

                {
                    size_t slicePitch= pitch*d_sz[1]*(kp);
                    char * slice= ptr+  slicePitch;
                    size_t colPitch= j*pitch;
                    float * row= (float *)(slice+ colPitch);
                    float val_p= row[i*N+V];

                    slicePitch= pitch*d_sz[1]*(km);
                    slice= ptr+  slicePitch;
                    row= (float *)(slice+ colPitch);
                    float val_m= row[i*N+V];

                    dx_forward[2][V]=  (val_p - val_c)/d_spc[2];
                    dx_backward[2][V]= (val_c - val_m)/d_spc[2];
                    dx[2][V]= 0.5*(val_p - val_m)/d_spc[2];
                }
            } //for V



            double shifted_derivs_aug[3][3][N]={0};
            double shifted_derivs_dim[3][3][N]={0};

            for(int V=0;V<N;V++)
            {
                {
                    size_t slicePitch= pitch*d_sz[1]*k;
                    char * slice= ptr+  slicePitch;

                    size_t colPitch= (jp)*pitch;
                    float * row= (float *)(slice+ colPitch);
                    float val_p= row[(ip)*N+V] ;
                    float val_m= row[(im)*N+V] ;

                    shifted_derivs_aug[0][1][V]= 0.5*(val_p - val_m)/d_spc[0];

                    colPitch= (jm)*pitch;
                    row= (float *)(slice+ colPitch);
                    val_p= row[(ip)*N+V] ;
                    val_m= row[(im)*N+V] ;

                    shifted_derivs_dim[0][1][V]= 0.5*(val_p - val_m)/d_spc[0];
                }

                {
                    size_t slicePitch= pitch*d_sz[1]*(kp);
                    char * slice= ptr+  slicePitch;
                    size_t colPitch= j*pitch;
                    float * row= (float *)(slice+ colPitch);
                    float val_p= row[(ip)*N+V] ;
                    float val_m= row[(im)*N+V] ;

                    shifted_derivs_aug[0][2][V]= 0.5*(val_p - val_m)/d_spc[0];

                    slicePitch= pitch*d_sz[1]*(km);
                    slice= ptr+  slicePitch;
                    colPitch= j*pitch;
                    row= (float *)(slice+ colPitch);
                    val_p= row[(ip)*N+V] ;
                    val_m= row[(im)*N+V] ;

                    shifted_derivs_dim[0][2][V]= 0.5*(val_p - val_m)/d_spc[0];
                }


                {
                    size_t slicePitch= pitch*d_sz[1]*(k);
                    char *slice= ptr+  slicePitch;
                    size_t colPitch= (jp)*pitch;
                    float * row= (float *)(slice+ colPitch);
                    float val_pp= row[(ip)*N+V];
                    float val_pm= row[(im)*N+V];

                    colPitch= (jm)*pitch;
                    row= (float *)(slice+ colPitch);
                    float val_mp= row[(ip)*N+V];
                    float val_mm= row[(im)*N+V];

                    shifted_derivs_aug[1][0][V]= 0.5*(val_pp - val_mp)/d_spc[1];
                    shifted_derivs_dim[1][0][V]= 0.5*(val_pm - val_mm)/d_spc[1];
                }


                {
                    size_t slicePitch= pitch*d_sz[1]*(kp);
                    char * slice= ptr+  slicePitch;
                    size_t colPitch= j*pitch;
                    float * row= (float *)(slice+ colPitch);
                    float val_pp= row[(ip)*N+V] ;
                    float val_pm= row[(im)*N+V] ;

                    slicePitch= pitch*d_sz[1]*(km);
                    slice= ptr+  slicePitch;
                    colPitch= j*pitch;
                    row= (float *)(slice+ colPitch);
                    float val_mp= row[(ip)*N+V] ;
                    float val_mm= row[(im)*N+V] ;

                    shifted_derivs_aug[2][0][V]= 0.5*(val_pp - val_mp)/d_spc[2];
                    shifted_derivs_dim[2][0][V]= 0.5*(val_pm - val_mm)/d_spc[2];
                }



                {
                    size_t slicePitch= pitch*d_sz[1]*(kp);
                    char * slice= ptr+  slicePitch;
                    size_t colPitch= (jp)*pitch;
                    float * row= (float *)(slice+ colPitch);
                    float val_pp= row[(i)*N+V] ;
                    colPitch= (jm)*pitch;
                    row= (float *)(slice+ colPitch);
                    float val_pm= row[(i)*N+V] ;

                    slicePitch= pitch*d_sz[1]*(km);
                    slice= ptr+  slicePitch;
                    colPitch= (jp)*pitch;
                    row= (float *)(slice+ colPitch);
                    float val_mp= row[(i)*N+V] ;
                    colPitch= (jm)*pitch;
                    row= (float *)(slice+ colPitch);
                    float val_mm= row[(i)*N+V] ;

                    shifted_derivs_aug[2][1][V]= 0.5*(val_pp - val_mp)/d_spc[2];
                    shifted_derivs_dim[2][1][V]= 0.5*(val_pm - val_mm)/d_spc[2];
                }

                {
                    size_t slicePitch= pitch*d_sz[1]*(kp);
                    char * slice= ptr+  slicePitch;
                    size_t colPitch= (jp)*pitch;
                    float * row= (float *)(slice+ colPitch);
                    float val_pp= row[(i)*N+V] ;
                    colPitch= (jm)*pitch;
                    row= (float *)(slice+ colPitch);
                    float val_mp= row[(i)*N+V] ;

                    slicePitch= pitch*d_sz[1]*(km);
                    slice= ptr+  slicePitch;
                    colPitch= (jp)*pitch;
                    row= (float *)(slice+ colPitch);
                    float val_pm= row[(i)*N+V] ;
                    colPitch= (jm)*pitch;
                    row= (float *)(slice+ colPitch);
                    float val_mm= row[(i)*N+V] ;

                    shifted_derivs_aug[1][2][V]= 0.5*(val_pp - val_mp)/d_spc[1];
                    shifted_derivs_dim[1][2][V]= 0.5*(val_pm - val_mm)/d_spc[1];
                }
            } //for V





            double Cx[3];
            double Cxd[3];

            for (unsigned int d = 0; d < 3; ++d) //derivative direction
            {
                double GradMag = 0.0;
                double GradMag_d = 0.0;
                for (unsigned int V = 3; V < N; ++V)
                {
                    GradMag += dx_forward[d][V]*dx_forward[d][V];
                    GradMag_d += dx_backward[d][V]*dx_backward[d][V];


                    for (unsigned int d2 = 0; d2 < 3; ++d2) //offset direction
                    {
                        if (d2 != d)
                        {
                            double dx_aug= shifted_derivs_aug[d2][d][V];
                            double dx_dim= shifted_derivs_dim[d2][d][V];

                            GradMag   += 0.25 * (dx[d2][V] + dx_aug) * (dx[d2][V] + dx_aug);
                            GradMag_d += 0.25 * (dx[d2][V] + dx_dim) * (dx[d2][V] + dx_dim);
                        }
                    }
                }
                if (m_K == 0.0)
                {
                    Cx[d] = 0.0;
                    Cxd[d] = 0.0;
                }
                else
                {
                    Cx[d] = exp(GradMag / m_K);
                    Cxd[d] = exp(GradMag_d / m_K);
                }
            }

            double delta;
            for (unsigned int V = 0; V < 5; ++V)
            {
                delta = 0;

                for (unsigned int d = 0; d < 3; ++d)
                {
                    dx_forward[d][V] *= Cx[d];
                    dx_backward[d][V] *= Cxd[d];
                    delta += dx_forward[d][V] - dx_backward[d][V];
                }

                row_update_img[i*N+V]= deltaT* delta;
            }
        }
    }
}



__global__ void
CopyNecessary_kernel(cudaPitchedPtr input_img, cudaPitchedPtr output_img)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2] )
        {
            size_t input_imgpitch= input_img.pitch;
            size_t input_imgslicePitch= input_imgpitch*d_sz[1]*k;
            size_t input_imgcolPitch= j*input_imgpitch;
            char * input_img_ptr= (char *)(input_img.ptr);
            char * slice_input_img= input_img_ptr+  input_imgslicePitch;
            float * row_input_img= (float *)(slice_input_img+ input_imgcolPitch);

            size_t output_imgpitch= output_img.pitch;
            size_t output_imgslicePitch= output_imgpitch*d_sz[1]*k;
            size_t output_imgcolPitch= j*output_imgpitch;
            char * output_img_ptr= (char *)(output_img.ptr);
            char * slice_output_img= output_img_ptr+  output_imgslicePitch;
            float * row_output_img= (float *)(slice_output_img+ output_imgcolPitch);


            row_output_img[3*i+0] =row_input_img[5*i+0];
            row_output_img[3*i+1] =row_input_img[5*i+1];
            row_output_img[3*i+2] =row_input_img[5*i+2];
        }
    }
}

__global__ void
ZeroOut_kernel(cudaPitchedPtr img)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if( j <d_sz[1] && k<d_sz[2] )
        {
            size_t imgpitch= img.pitch;
            size_t imgslicePitch= imgpitch*d_sz[1]*k;
            size_t imgcolPitch= j*imgpitch;
            char * img_ptr= (char *)(img.ptr);
            char * slice_img= img_ptr+  imgslicePitch;
            char * row_img= (char *)(slice_img+ imgcolPitch);

            if(i==0)
            {
                for(int ma=0;ma<imgpitch;ma++)
                    row_img[ma]=0;
            }


        }
    }
}

__global__ void sum3DArray(double* data, double* result, int size) {
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-striding loop for arrays larger than the grid size
   // for (int i = idx; i < size; i += blockDim.x * gridDim.x)
    //{
      //  atomicAdd(result, data[i]);
    //}

    if(blockIdx.x==0 &&threadIdx.x ==0)
    {
        result[0]=0;
        for(int i=0;i<size;i++)
            result[0]+=data[i];
    }
}

void AnisotropicSmoothField_cuda(cudaPitchedPtr field,
                                 cudaPitchedPtr TR_img,cudaPitchedPtr FA_img,
                                 int3 data_sz,float3 data_spc,
                                 float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                                 float3 data_orig,
                                 cudaPitchedPtr output )
{
    float h_d_dir[]= {data_d00,data_d01,data_d02,data_d10,data_d11,data_d12,data_d20,data_d21,data_d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));

    float h_d_orig[]= {data_orig.x,data_orig.y,data_orig.z};
    gpuErrchk(cudaMemcpyToSymbol(d_orig, &h_d_orig, 3 * sizeof(float)));

    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));

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


    float max_val_field , min_val_field;
    {
        float* dev_out;
        float out;
        cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

        ScalarFindMax<<<gSize, bSize>>>((float *)field.ptr, field.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
        cudaDeviceSynchronize();
        ScalarFindMax<<<1, bSize>>>(dev_out, gSize, dev_out);
        cudaDeviceSynchronize();

        cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_out);
        max_val_field=out;
    }
    {
        float* dev_out;
        float out;
        cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

        ScalarFindMin<<<gSize, bSize>>>((float *)field.ptr, field.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
        cudaDeviceSynchronize();
        ScalarFindMin<<<1, bSize>>>(dev_out, gSize, dev_out);
        cudaDeviceSynchronize();

        cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_out);
        min_val_field=out;
    }

    ScaleImageForAnisotropicSmoothing_kernel<<< blockSize,gridSize>>>( TR_img, (max_val_field -min_val_field)/9000., min_val_field );
    ScaleImageForAnisotropicSmoothing_kernel<<< blockSize,gridSize>>>( FA_img, (max_val_field -min_val_field), min_val_field );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize())

    cudaPitchedPtr concat_img={0};
    cudaExtent extent =  make_cudaExtent(5*sizeof(float)*data_sz.x,data_sz.y,data_sz.z);
    cudaMalloc3D(&concat_img, extent);
    //cudaMemset3D(concat_img, 0,extent);
    ZeroOut_kernel<<< blockSize,gridSize>>>(concat_img);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    ConcatImagesForAnisotropicSmoothing_kernel<<< blockSize,gridSize>>>( field,TR_img,FA_img, concat_img);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int Niter=30;
    double Kond=(max_val_field-min_val_field)*0.7;
    float timestep=0.0625;

    float *dev_deltaT;
    cudaMalloc((void**)&dev_deltaT, sizeof(float));
    cudaMemcpy(dev_deltaT, &timestep, sizeof(float), cudaMemcpyHostToDevice);


    cudaExtent extent2 =  make_cudaExtent(sizeof(double)*data_sz.x,data_sz.y,data_sz.z);
    cudaPitchedPtr grad_sq_img={0};
    cudaMalloc3D(&grad_sq_img, extent2);

    cudaPitchedPtr update_img={0};
    cudaMalloc3D(&update_img, extent);


    for(int iter=0;iter<Niter;iter++)
    {
       // cudaMemset3D(grad_sq_img,0,extent2);
        ZeroOut_kernel<<< blockSize,gridSize>>>(grad_sq_img);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

       // cudaMemset3D(update_img,0,extent);
        ZeroOut_kernel<<< blockSize,gridSize>>>(update_img);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        ComputeAverageSquaredGradientImage_kernel<<< blockSize,gridSize>>>(concat_img, grad_sq_img );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        double out;
        {
            double* dev_out;

            cudaMalloc((void**)&dev_out, sizeof(double)*gSize);

            ScalarFindSum<<<gSize, bSize>>>((double *)grad_sq_img.ptr, grad_sq_img.pitch/sizeof(double)*data_sz.y*data_sz.z,dev_out);
            cudaDeviceSynchronize();
            ScalarFindSum<<<1, bSize>>>(dev_out, gSize, dev_out);
            cudaDeviceSynchronize();

            cudaMemcpy(&out, dev_out, sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(dev_out);
        }

        double avg_grad_sq = out/data_sz.x/data_sz.y/data_sz.z;
        double mK = avg_grad_sq * Kond * Kond * -2.;


        //std::cout<<out<< " " << avg_grad_sq<< " " << mK << std::endl;

        double *dev_mK;
        cudaMalloc((void**)&dev_mK, sizeof(double));
        cudaMemcpy(dev_mK, &mK, sizeof(double), cudaMemcpyHostToDevice);

        ComputeAnisotropicFilteringUpdate_kernel<<< blockSize,gridSize>>>(concat_img, dev_mK,dev_deltaT,update_img  );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        AddImages_cuda(concat_img, update_img , concat_img,data_sz,5);
        cudaFree(dev_mK);
    }


    CopyNecessary_kernel<<< blockSize,gridSize>>>(concat_img, output );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaFree(update_img.ptr);
    cudaFree(grad_sq_img.ptr);
    cudaFree(dev_deltaT);
    cudaFree(concat_img.ptr);


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

#endif







/*
    {

        const int Ncomp=3;
        using ConcatenatedImagePixelType= itk::Vector<float,Ncomp>;
        using ConcatenatedImageType = itk::Image<ConcatenatedImagePixelType,3>;
        using ImageType3D= itk::Image<float,3>;
        using ImageType4D= itk::Image<float,4>;


        ImageType3D::SizeType sz2;
        sz2[0]= data_sz.x;
        sz2[1]= data_sz.y;
        sz2[2]= data_sz.z;
        ImageType3D::IndexType start;
        start.Fill(0);
        ImageType3D::RegionType reg(start,sz2);

        ImageType3D::PointType orig;
        orig[0]= data_orig.x;
        orig[1]= data_orig.y;
        orig[2]= data_orig.z;

        ImageType3D::SpacingType spc;
        spc[0]= data_spc.x;
        spc[1]= data_spc.y;
        spc[2]= data_spc.z;

        float * itk_image_data2 = new float[(long)sz2[0]*sz2[1]*sz2[2]*Ncomp];
        copy3DPitchedPtrToHost(output,itk_image_data2,Ncomp*sz2[0],sz2[1],sz2[2]);


        ConcatenatedImageType::PixelType* itk_image_data = (ConcatenatedImageType::PixelType*)itk_image_data2 ;

        typedef itk::ImportImageFilter< ConcatenatedImageType::PixelType , 3 >   ImportFilterType;
        ImportFilterType::Pointer importFilter = ImportFilterType::New();
        importFilter->SetRegion( reg );
        importFilter->SetOrigin( orig );
        importFilter->SetSpacing( spc );


        const bool importImageFilterWillOwnTheBuffer = true;
        importFilter->SetImportPointer( itk_image_data, (long)sz2[0]*sz2[1]*sz2[2]*Ncomp,    importImageFilterWillOwnTheBuffer );
        importFilter->Update();
        ConcatenatedImageType::Pointer itk_image_float= importFilter->GetOutput();

        ConcatenatedImageType::IndexType ind3;
        ind3[0]=9;ind3[1]=15;ind3[2]=10;
        std::cout<<itk_image_float->GetPixel(ind3)<<std::endl;
        using WrType = itk::ImageFileWriter<ConcatenatedImageType>;
        WrType::Pointer aa =WrType::New();
        aa->SetFileName("/qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/src/tools/Test/aaabbbb_gpu.nii");
        aa->SetInput(itk_image_float);
        aa->Update();

        exit(0);
    }
*/



