#ifndef _RESAMPLEIMAGE_CUDA_CU
#define _RESAMPLEIMAGE_CUDA_CU

#include <stdio.h>
#include <iostream>

#include "cuda_utils.h"



#define BLOCKSIZE 32
#define PER_SLICE 1

__constant__ float d_dir[9];
__constant__ float v_dir[9];

__constant__ int d_sz[3];
__constant__ int v_sz[3];

__constant__ float d_orig[3];
__constant__ float v_orig[3];

__constant__ float d_spc[3];
__constant__ float v_spc[3];



__global__ void
ResampleImage_kernel(cudaPitchedPtr data, cudaPitchedPtr output ,int Ncomponents)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
	    if(i<v_sz[0] && j <v_sz[1] && k<v_sz[2])
	    {
		size_t opitch= output.pitch;
		size_t oslicePitch= opitch*v_sz[1]*k;
		size_t ocolPitch= j*opitch;

		char *o_ptr= (char *)(output.ptr);
		char * slice_o= o_ptr+  oslicePitch;
		float * row_out= (float *)(slice_o+ ocolPitch);
		
                float x= (v_dir[0]*i  + v_dir[1]*j + v_dir[2]*k)* v_spc[0] + v_orig[0] -d_orig[0];
                float y= (v_dir[3]*i  + v_dir[4]*j + v_dir[5]*k)* v_spc[1] + v_orig[1] -d_orig[1];
                float z= (v_dir[6]*i  + v_dir[7]*j + v_dir[8]*k)* v_spc[2] + v_orig[2] -d_orig[2];
	
                float iw = (d_dir[0]*x  + d_dir[3]*y  + d_dir[6]*z)/ d_spc[0] ;
                float jw = (d_dir[1]*x  + d_dir[4]*y  + d_dir[7]*z)/ d_spc[1] ;
                float kw = (d_dir[2]*x  + d_dir[5]*y  + d_dir[8]*z)/ d_spc[2] ;


                if(iw<0 || iw> d_sz[0]-1 || jw<0 || jw> d_sz[1]-1 || kw<0 || kw> d_sz[2]-1)
                {
                    for(int mm=0;mm<Ncomponents;mm++)
                        row_out[Ncomponents*i +mm]=0;
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


                    size_t dpitch= data.pitch;
                    char *d_ptr= (char *)(data.ptr);
                    {
                        size_t dslicePitch= dpitch*d_sz[1]*floor_z;
                        char * slice_d= d_ptr+  dslicePitch;
                        {
                            size_t dcolPitch= floor_y*dpitch;
                            float * row_data= (float *)(slice_d+ dcolPitch);                            
                            for(int mm=0;mm<Ncomponents;mm++)
                            {
                                ia1[mm] = row_data[Ncomponents*floor_x +mm];
                                ja1[mm] = row_data[Ncomponents*ceil_x +mm];
                            }

                        }
                        {
                            size_t dcolPitch= ceil_y*dpitch;
                            float * row_data= (float *)(slice_d+ dcolPitch);
                            for(int mm=0;mm<Ncomponents;mm++)
                            {
                                ia2[mm] = row_data[Ncomponents*floor_x +mm];
                                ja2[mm] = row_data[Ncomponents*ceil_x +mm];
                            }
                        }
                    }
                    {
                        size_t dslicePitch= dpitch*d_sz[1]*ceil_z;
                        char * slice_d= d_ptr+  dslicePitch;
                        {
                            size_t dcolPitch= floor_y*dpitch;
                            float * row_data= (float *)(slice_d+ dcolPitch);
                            for(int mm=0;mm<Ncomponents;mm++)
                            {
                                ib1[mm] = row_data[Ncomponents*floor_x +mm];
                                jb1[mm] = row_data[Ncomponents*ceil_x +mm];
                            }
                        }
                        {
                            size_t dcolPitch= ceil_y*dpitch;
                            float * row_data= (float *)(slice_d+ dcolPitch);
                            for(int mm=0;mm<Ncomponents;mm++)
                            {
                                ib2[mm] = row_data[Ncomponents*floor_x +mm];
                                jb2[mm] = row_data[Ncomponents*ceil_x +mm];
                            }
                        }
                    }

                    for(int mm=0;mm<Ncomponents;mm++)
                    {
                        float i1= ia1[mm]*(1-zd) + ib1[mm]*zd;
                        float i2= ia2[mm]*(1-zd) + ib2[mm]*zd;
                        float j1= ja1[mm]*(1-zd) + jb1[mm]*zd;
                        float j2= ja2[mm]*(1-zd) + jb2[mm]*zd;

                        float w1= i1*(1-yd) + i2*yd;
                        float w2= j1*(1-yd) + j2*yd;

                        row_out[Ncomponents*i + mm]=   w1*(1-xd) + w2*xd;
                    }
                }
	    }
    }

}



void ResampleImage_cuda(cudaPitchedPtr data,
                     int3 data_sz,float3 data_spc,
                     float data_d00,  float data_d01,float data_d02,float data_d10,float data_d11,float data_d12,float data_d20,float data_d21,float data_d22,
                     float3 data_orig,
                     int3 virtual_sz,float3 virtual_spc,
                     float virtual_d00,  float virtual_d01,float virtual_d02,float virtual_d10,float virtual_d11,float virtual_d12,float virtual_d20,float virtual_d21,float virtual_d22,
                     float3 virtual_orig,
                     int Ncomponents,
                     cudaPitchedPtr output )
{
    

    float h_d_dir[]= {data_d00,data_d01,data_d02,data_d10,data_d11,data_d12,data_d20,data_d21,data_d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));
    float h_v_dir[]= {virtual_d00,virtual_d01,virtual_d02,virtual_d10,virtual_d11,virtual_d12,virtual_d20,virtual_d21,virtual_d22};
    gpuErrchk(cudaMemcpyToSymbol(v_dir, &h_v_dir, 9 * sizeof(float)));
    
    float h_d_orig[]= {data_orig.x,data_orig.y,data_orig.z};
    gpuErrchk(cudaMemcpyToSymbol(d_orig, &h_d_orig, 3 * sizeof(float)));
    float h_v_orig[]= {virtual_orig.x,virtual_orig.y,virtual_orig.z};
    gpuErrchk(cudaMemcpyToSymbol(v_orig, &h_v_orig, 3 * sizeof(float)));
     
    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));
    float h_v_spc[]= {virtual_spc.x,virtual_spc.y,virtual_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(v_spc, &h_v_spc, 3 * sizeof(float)));
    
       
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));
    int h_v_sz[]= {virtual_sz.x,virtual_sz.y,virtual_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(v_sz, &h_v_sz, 3 * sizeof(int)));


    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(std::ceil(1.*virtual_sz.x / blockSize.x), std::ceil(1.*virtual_sz.y / blockSize.y), std::ceil(1.*virtual_sz.z / blockSize.z/PER_SLICE) );
    while(gridSize.x *gridSize.y *gridSize.z >1024)
    {
        blockSize.x*=2;
        blockSize.y*=2;
        blockSize.z*=2;
        gridSize=dim3(std::ceil(1.*virtual_sz.x / blockSize.x), std::ceil(1.*virtual_sz.y / blockSize.y), std::ceil(1.*virtual_sz.z / blockSize.z/PER_SLICE) );
    }


    ResampleImage_kernel<<< blockSize,gridSize>>>( data,output,Ncomponents );


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}








#endif
