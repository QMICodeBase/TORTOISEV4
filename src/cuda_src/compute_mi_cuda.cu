#ifndef _COMPUTEMICUDA_CU
#define _COMPUTEMICUDA_CU

#include <stdio.h>
#include <iostream>

#include "cuda_utils.h"



uint NUM_BINS;
uint NUM_PARTS;


__global__ void single_histogram_smem_atomics(cudaPitchedPtr data, int3 sz, 
                                              float low_lim,float high_lim,
                                              unsigned int *out)
{
  // pixel coordinates
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if(i<sz.x && j<sz.y && k< sz.z)
    {
   
    
	  // grid dimensions
	    int nx = blockDim.x * gridDim.x; 
	    int ny = blockDim.y * gridDim.y;
	    int nz = blockDim.z * gridDim.z;
	  

	  // linear block index within 3D grid 
	    int g = blockIdx.x + blockIdx.y * gridDim.x  + gridDim.x * gridDim.y * blockIdx.z;

	  
	  // linear thread index within 3D block
	    int t = threadIdx.x + threadIdx.y * blockDim.x +  threadIdx.z *blockDim.x*blockDim.y ; 
	  

	  // total threads in 3D block
	    int nt = blockDim.x * blockDim.y * blockDim.z ; 
	  

	    // initialize temporary accumulation array in shared memory
	    __shared__ unsigned int smem[NUM_BINS + 1];
	    for (int i = t; i <  NUM_BINS + 1; i += nt) 
		smem[i] = 0;
	    __syncthreads();

	  // process pixels
	  // updates our block's partial histogram in shared memory
	    for (int col = i; col < sz.x; col += nx) 
		for (int row = j; row < sz.y; row += ny)
		    for (int depth = k; depth < sz.z; depth += nz)  
		    { 
		        size_t dpitch= data.pitch;
                        size_t dslicePitch= dpitch*sz.y*depth;
                        size_t dcolPitch= row*fpitch;
                        char *d_ptr= (char *)(data.ptr);
                        char * slice_d= d_ptr+  dslicePitch;
                        float * row_data= (float *)(slice_d+ dcolPitch);
		    
		        float val = row_data[col];
		        if(val>= low_lim && val <= high_lim)
		        {
		            int bin_id=   floor((val - low_lim) * NUM_BINS / (high_lim-low_lim));
		            if(bin_id>=NUM_BINS)
  		                bin_id--;
  		            if(bin_id<0)
  		                bin_id=0;

       		            atomicAdd(&smem[bin_id], 1);
  		                
		        }
		    }
	    __syncthreads();

	  // write partial histogram into the global memory
	  out += g * NUM_PARTS;
	  for (int hi = t; hi < NUM_BINS; hi += nt) 
	  {
	    out[hi ] = smem[hi ];
	  }
    }
}








__global__ void
QuadraticTransformImage_kernel(cudaTextureObject_t tex, int3 sz, float3 res, float3 orig,
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
	    if(i<sz.x && j <sz.y && k<sz.z)
	    {
		size_t opitch= output.pitch;
		size_t oslicePitch= opitch*sz.y*k;
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
                                                

		row_out[i] =tex3D<float>(tex, iw+0.5, jw +0.5, kw+0.5);
	    }

    }

}



void QuadraticTransformImage_cuda(cudaTextureObject_t tex,int3 sz,float3 res, float3 orig,
                     float d00,  float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                     float rotmat_arr[],
                     float *params_arr;
                     cudaPitchedPtr output )
{
    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*sz.x / blockSize.x), std::ceil(1.*sz.y / blockSize.y), std::ceil(1.*sz.z / blockSize.z/PER_SLICE) );


    float smat[16]={0},smat_inv[16]={0};
    smat[0]= d00*res.x;
    smat[1]= d01*res.y;
    smat[2]= d02*res.z;    
    smat[4]= d10*res.x;
    smat[5]= d11*res.y;
    smat[6]= d12*res.z;    
    smat[7]= d20*res.x;
    smat[9]= d21*res.y;
    smat[10]=d22*res.z;    
    smat[3]= orig.x;
    smat[7]= orig.y;
    smat[11]= orig.z;
    smat[15]=1;    
    
    smat_inv[0]= d00/res.x;
    smat_inv[1]= d10/res.x;
    smat_inv[2]= d20/res.x;
    smat_inv[4]= d01/res.y;
    smat_inv[5]= d11/res.y;
    smat_inv[6]= d21/res.y;
    smat_inv[7]= d02/res.z;
    smat_inv[8]= d12/res.z;
    smat_inv[10]=d22/res.z;
    smat_inv[15]=1;
    smat_inv[3]= -(smat_inv[0] *orig.x +smat_inv[1] *orig.y + smat_inv[2] *orig.z );
    smat_inv[7]= -(smat_inv[4] *orig.x +smat_inv[5] *orig.y + smat_inv[6] *orig.z );
    smat_inv[11]=-(smat_inv[8] *orig.x +smat_inv[9] *orig.y + smat_inv[10] *orig.z );
                        

    float* dsmat,dsmat_inv;
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
    if( (fabs(params_arr[7]) > fabs(params_arr[6]) && (fabs(params_arr[7]) > fabs(params_arr[8]) )
        phase=1;
    if( (fabs(params_arr[8]) > fabs(params_arr[6]) && (fabs(params_arr[8]) > fabs(params_arr[7]) )
        phase=2;
        
    bool do_cubic=false;
    for(int p=14;p<=20;p++)
    {
        if(params_arr[p] !=0)
        {
            do_cubic=true;
            break;
        }
    }
        
      
    
    QuadraticTransformImage_kernel<<< blockSize,gridSize>>>( tex, sz, res, orig,
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
