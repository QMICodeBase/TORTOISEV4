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

    const float m_MaxErrorToleranceThreshold=0.1;
    const float m_MeanErrorToleranceThreshold=0.001;
    const int Niter=20;

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





#endif
