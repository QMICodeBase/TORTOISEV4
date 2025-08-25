#ifndef _DRTAMAS_UTILITIES_CU
#define _DRTAMAS_UTILITIES_CU



#include "cuda_utils.h"



#define BLOCKSIZE 32
#define PER_SLICE 1
#define CUDART_PI_F 3.141592654f
#define LOGNEG -25

extern __constant__ float d_dir[9];
extern __constant__ int d_sz[3];
extern __constant__ float d_orig[3];
extern __constant__ float d_spc[3];


extern __device__ void MatrixMultiply(float A[3][3], float B[3][3], float C[3][3]);

extern __device__ void MatrixTranspose(float A[3][3], float AT[3][3]);



__device__ void ComputeEigenVals(float x[3][3], float *vals)
{
    typedef float scalar_t;
    const scalar_t x11 = x[0][0];
    const scalar_t x12 = x[0][1];
    const scalar_t x13 = x[0][2];
    const scalar_t x21 = x[1][0];
    const scalar_t x22 = x[1][1];
    const scalar_t x23 = x[1][2];
    const scalar_t x31 = x[2][0];
    const scalar_t x32 = x[2][1];
    const scalar_t x33 = x[2][2];

    const scalar_t p1 = x12 * x12 + x13 * x13 + x23 * x23;

    if (p1 == 0)
    {
        vals[ 0] = x11;
        vals[ 1] = x22;
        vals[ 2] = x33;
    }
    else
    {
        const scalar_t q = (x11 + x22 + x33) / 3.0;
        const scalar_t p2 = (x11 - q) * (x11 - q) + (x22 - q) * (x22 - q) +
                            (x33 - q) * (x33 - q) + 2 * p1;
        const scalar_t p = sqrt(p2 / 6.0);

        const scalar_t b11 = (1.0 / p) * (x11 - q);
        const scalar_t b12 = (1.0 / p) * x12;
        const scalar_t b13 = (1.0 / p) * x13;
        const scalar_t b21 = (1.0 / p) * x21;
        const scalar_t b22 = (1.0 / p) * (x22 - q);
        const scalar_t b23 = (1.0 / p) * x23;
        const scalar_t b31 = (1.0 / p) * x31;
        const scalar_t b32 = (1.0 / p) * x32;
        const scalar_t b33 = (1.0 / p) * (x33 - q);

        scalar_t r = b11 * b22 * b33 + b12 * b23 * b31 + b13 * b21 * b32 -
                     b13 * b22 * b31 - b12 * b21 * b33 - b11 * b23 * b32;
        r = r / 2.0;

        scalar_t phi;
        if (r <= -1)
        {
            phi = M_PI / 3.0;
        }
        else if (r >= 1)
        {
            phi = 0;
        }
        else
        {
            phi = acos(r) / 3.0;
        }

        vals[ 0] = q + 2 * p * cos(phi);
        vals[ 2] = q + 2 * p * cos(phi + (2 * CUDART_PI_F / 3));
        vals[ 1] = 3 * q - vals[ 0] - vals[ 2];
    }

}


__device__ void ComputeEigVecs(float x[3][3], float *eig_val, float eig_vec[3][3])
{
    typedef float scalar_t;
    for(int e=0;e<3;e++)
    {
        const scalar_t x11 = x[0][0] -eig_val[e];
        const scalar_t x12 = x[0][1];
        const scalar_t x13 = x[0][2];
        const scalar_t x21 = x[1][0];
        const scalar_t x22 = x[1][1] -eig_val[e];
        const scalar_t x23 = x[1][2];
        const scalar_t x31 = x[2][0];
        const scalar_t x32 = x[2][1];
        const scalar_t x33 = x[2][2] -eig_val[e];

        const scalar_t r12_1 = x12 * x23 - x13 * x22;
        const scalar_t r12_2 = x13 * x21 - x11 * x23;
        const scalar_t r12_3 = x11 * x22 - x12 * x21;
        const scalar_t r13_1 = x12 * x33 - x13 * x32;
        const scalar_t r13_2 = x13 * x31 - x11 * x33;
        const scalar_t r13_3 = x11 * x32 - x12 * x31;
        const scalar_t r23_1 = x22 * x33 - x23 * x32;
        const scalar_t r23_2 = x23 * x31 - x21 * x33;
        const scalar_t r23_3 = x21 * x32 - x22 * x31;

        const scalar_t d1 = r12_1 * r12_1 + r12_2 * r12_2 + r12_3 * r12_3;
        const scalar_t d2 = r13_1 * r13_1 + r13_2 * r13_2 + r13_3 * r13_3;
        const scalar_t d3 = r23_1 * r23_1 + r23_2 * r23_2 + r23_3 * r23_3;

        scalar_t d_max = d1;
        int i_max = 0;

        if (d2 > d_max)
        {
            d_max = d2;
            i_max = 1;
        }

        if (d3 > d_max)
        {
            i_max = 2;
        }

        if(d1==0 && d2==0 && d3==0)
        {
            eig_vec[0][e]=0;
            eig_vec[1][e]=0;
            eig_vec[2][e]=0;
            eig_vec[e][e]=1;
        }
        else
        {
            if (i_max == 0)
            {
                eig_vec[0][e]  = r12_1 / sqrt(d1);
                eig_vec[1][e]  = r12_2 / sqrt(d1);
                eig_vec[2][e]  = r12_3 / sqrt(d1);
            } else if (i_max == 1) {
                eig_vec[0][e] = r13_1 / sqrt(d2);
                eig_vec[1][e] = r13_2 / sqrt(d2);
                eig_vec[2][e] = r13_3 / sqrt(d2);
            } else {
                eig_vec[0][e] = r23_1 / sqrt(d3);
                eig_vec[1][e] = r23_2 / sqrt(d3);
                eig_vec[2][e] = r23_3 / sqrt(d3);
            }

        }

    }

}


__device__ void ComputeEigens(float mat[3][3], float *vals, float vecs[3][3])
{
    ComputeEigenVals(mat, vals);
    ComputeEigVecs(mat, vals, vecs);

}



__global__ void
ComputeTRMapC_kernel(cudaPitchedPtr tensor, cudaPitchedPtr output,  const int3 d_sz)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t tpitch= tensor.pitch;
                size_t tslicePitch= tpitch*d_sz.y*k;
                size_t tcolPitch= j*tpitch;
                char *t_ptr= (char *)(tensor.ptr);
                char * slice_t= t_ptr+  tslicePitch;
                float * row_t= (float *)(slice_t+ tcolPitch);

                size_t opitch= output.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;
                char *o_ptr= (char *)(output.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_o= (float *)(slice_o+ ocolPitch);
                
                row_o[i] = row_t[6*i+0] +  row_t[6*i+3] + row_t[6*i+5];

           }
    }
}

void  ComputeTRMapC_cuda(cudaPitchedPtr tensor_img, cudaPitchedPtr output, const int3 data_sz)
{
    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );
    
    ComputeTRMapC_kernel<<< blockSize,gridSize>>>(tensor_img,output,data_sz);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}





__global__ void
ComputeFAMapC_kernel(cudaPitchedPtr tensor, cudaPitchedPtr output,  const int3 d_sz)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
        {
            size_t tpitch= tensor.pitch;
            size_t tslicePitch= tpitch*d_sz.y*k;
            size_t tcolPitch= j*tpitch;
            char *t_ptr= (char *)(tensor.ptr);
            char * slice_t= t_ptr+  tslicePitch;
            float * row_t= (float *)(slice_t+ tcolPitch);

            size_t opitch= output.pitch;
            size_t oslicePitch= opitch*d_sz.y*k;
            size_t ocolPitch= j*opitch;
            char *o_ptr= (char *)(output.ptr);
            char * slice_o= o_ptr+  oslicePitch;
            float * row_o= (float *)(slice_o+ ocolPitch);


            float A[3][3];
            A[0][0]=row_t[6*i+0];
            A[0][1]=row_t[6*i+1];
            A[0][2]=row_t[6*i+2];
            A[1][0]=row_t[6*i+1];
            A[1][1]=row_t[6*i+3];
            A[1][2]=row_t[6*i+4];
            A[2][0]=row_t[6*i+2];
            A[2][1]=row_t[6*i+4];
            A[2][2]=row_t[6*i+5];

            float vals[3]={0};
            float vecs[3][3]={0};
            ComputeEigens(A, vals, vecs);



            float mn = (vals[0]+vals[1]+vals[2])/3.;
            float nom = (vals[0]-mn)*(vals[0]-mn)+(vals[1]-mn)*(vals[1]-mn)+(vals[2]-mn)*(vals[2]-mn);
            float denom = vals[0]*vals[0]+vals[1]*vals[1]+vals[2]*vals[2];
            float FA= sqrt(1.5*nom/denom);

            row_o[i] = FA;
        }
    }
}

void  ComputeFAMapC_cuda(cudaPitchedPtr tensor_img, cudaPitchedPtr output, const int3 data_sz)
{
    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );

    ComputeFAMapC_kernel<<< blockSize,gridSize>>>(tensor_img,output,data_sz);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}













__global__ void
SplitImageComponents_kernel(cudaPitchedPtr img, cudaPitchedPtr *output,  const int3 d_sz , const int Ncomps)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t tpitch= img.pitch;
                size_t tslicePitch= tpitch*d_sz.y*k;
                size_t tcolPitch= j*tpitch;
                char *t_ptr= (char *)(img.ptr);
                char * slice_t= t_ptr+  tslicePitch;
                float * row_t= (float *)(slice_t+ tcolPitch);


                for(int v=0;v<Ncomps;v++)
                {
                    size_t opitch= output[v].pitch;
                    size_t oslicePitch= opitch*d_sz.y*k;
                    size_t ocolPitch= j*opitch;
                    char *o_ptr= (char *)(output[v].ptr);
                    char * slice_o= o_ptr+  oslicePitch;
                    float * row_o= (float *)(slice_o+ ocolPitch);

                    row_o[i] = row_t[6*i+v] ;

                }
           }
    }
}



void SplitImageComponents_cuda(cudaPitchedPtr img,
                          cudaPitchedPtr *output,
                          int3 data_sz,
                          int Ncomp)
{

    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );

    SplitImageComponents_kernel<<< blockSize,gridSize>>>(img,output,data_sz,Ncomp);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}




__global__ void
CombineImageComponents_kernel(cudaPitchedPtr *img, cudaPitchedPtr output,  const int3 d_sz , const int Ncomps)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t opitch= output.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;
                char *o_ptr= (char *)(output.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_o= (float *)(slice_o+ ocolPitch);


                for(int v=0;v<Ncomps;v++)
                {
                    size_t tpitch= img[v].pitch;
                    size_t tslicePitch= tpitch*d_sz.y*k;
                    size_t tcolPitch= j*tpitch;
                    char *t_ptr= (char *)(img[v].ptr);
                    char * slice_t= t_ptr+  tslicePitch;
                    float * row_t= (float *)(slice_t+ tcolPitch);

                    row_o[6*i+v] = row_t[i];
                }
           }
    }
}



void CombineImageComponents_cuda(cudaPitchedPtr *img,
                          cudaPitchedPtr output,
                          int3 data_sz,
                          int Ncomp)
{
    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );

    CombineImageComponents_kernel<<< blockSize,gridSize>>>(img,output,data_sz,Ncomp);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}


__global__ void
LogTensor_kernel(cudaPitchedPtr tens, cudaPitchedPtr output,  const int3 d_sz )
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;


    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
        if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
        {
            size_t opitch= output.pitch;
            size_t oslicePitch= opitch*d_sz.y*k;
            size_t ocolPitch= j*opitch;
            char *o_ptr= (char *)(output.ptr);
            char * slice_o= o_ptr+  oslicePitch;
            float * row_o= (float *)(slice_o+ ocolPitch);

            size_t tpitch= tens.pitch;
            size_t tslicePitch= tpitch*d_sz.y*k;
            size_t tcolPitch= j*tpitch;
            char *t_ptr= (char *)(tens.ptr);
            char * slice_t= t_ptr+  tslicePitch;
            float * row_t= (float *)(slice_t+ tcolPitch);

            float vals[3]={0};
            float vecs[3][3]={0};
            float mat[3][3];
            mat[0][0]=row_t[6*i+0];mat[0][1]=row_t[6*i+1];mat[0][2]=row_t[6*i+2];
            mat[1][0]=row_t[6*i+1];mat[1][1]=row_t[6*i+3];mat[1][2]=row_t[6*i+4];
            mat[2][0]=row_t[6*i+2];mat[2][1]=row_t[6*i+4];mat[2][2]=row_t[6*i+5];

            ComputeEigens(mat, vals, vecs);

            if(vals[0]<=0)
                vals[0]=LOGNEG;
            else
                vals[0]=log(vals[0]);
            if(vals[1]<=0)
                vals[1]=LOGNEG;
            else
                vals[1]=log(vals[1]);
            if(vals[2]<=0)
                vals[2]=LOGNEG;
            else
                vals[2]=log(vals[2]);

            float valsm[3][3]={0};
            valsm[0][0]= (vals[0]);
            valsm[1][1]= (vals[1]);
            valsm[2][2]= (vals[2]);

            float UT[3][3]={0};
            MatrixTranspose(vecs,UT);

            float temp[3][3]={0};
            float final[3][3]={0};
            MatrixMultiply(vecs,valsm,temp);
            MatrixMultiply(temp,UT,final);

            row_o[6*i+0] = final[0][0];
            row_o[6*i+1] = final[0][1];
            row_o[6*i+2] = final[0][2];
            row_o[6*i+3] = final[1][1];
            row_o[6*i+4] = final[1][2];
            row_o[6*i+5] = final[2][2];

        }
    }

}



void  LogTensor_cuda(cudaPitchedPtr tens, cudaPitchedPtr output, const int3 data_sz)
{
    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );

    LogTensor_kernel<<< blockSize,gridSize>>>(tens,output,data_sz);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}



__global__ void
ExpTensor_kernel(cudaPitchedPtr tens, cudaPitchedPtr output,  const int3 d_sz )
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t opitch= output.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;
                char *o_ptr= (char *)(output.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_o= (float *)(slice_o+ ocolPitch);

                size_t tpitch= tens.pitch;
                size_t tslicePitch= tpitch*d_sz.y*k;
                size_t tcolPitch= j*tpitch;
                char *t_ptr= (char *)(tens.ptr);
                char * slice_t= t_ptr+  tslicePitch;
                float * row_t= (float *)(slice_t+ tcolPitch);

                float vals[3]={0};
                float vecs[3][3]={0};
                float mat[3][3];
                mat[0][0]=row_t[6*i+0];mat[0][1]=row_t[6*i+1];mat[0][2]=row_t[6*i+2];
                mat[1][0]=row_t[6*i+1];mat[1][1]=row_t[6*i+3];mat[1][2]=row_t[6*i+4];
                mat[2][0]=row_t[6*i+2];mat[2][1]=row_t[6*i+4];mat[2][2]=row_t[6*i+5];

                ComputeEigens(mat, vals, vecs);


                float valsm[3][3]={0};
                valsm[0][0]= exp(vals[0]);
                valsm[1][1]= exp(vals[1]);
                valsm[2][2]= exp(vals[2]);

                float UT[3][3]={0};
                MatrixTranspose(vecs,UT);

                float temp[3][3]={0};
                float final[3][3]={0};
                MatrixMultiply(vecs,valsm,temp);
                MatrixMultiply(temp,UT,final);

                row_o[6*i+0] = final[0][0];
                row_o[6*i+1] = final[0][1];
                row_o[6*i+2] = final[0][2];
                row_o[6*i+3] = final[1][1];
                row_o[6*i+4] = final[1][2];
                row_o[6*i+5] = final[2][2];
           }
    }
}



void  ExpTensor_cuda(cudaPitchedPtr tens, cudaPitchedPtr output, const int3 data_sz)
{
    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );

    ExpTensor_kernel<<< blockSize,gridSize>>>(tens,output,data_sz);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}



__global__ void
RotateTensors_kernel(cudaPitchedPtr tens, cudaPitchedPtr output,  const int3 d_sz , float *drotmat)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int k=PER_SLICE*kk;k<PER_SLICE*kk+PER_SLICE;k++)
    {
            if(i<d_sz.x && j <d_sz.y && k<d_sz.z)
            {
                size_t opitch= output.pitch;
                size_t oslicePitch= opitch*d_sz.y*k;
                size_t ocolPitch= j*opitch;
                char *o_ptr= (char *)(output.ptr);
                char * slice_o= o_ptr+  oslicePitch;
                float * row_o= (float *)(slice_o+ ocolPitch);

                size_t tpitch= tens.pitch;
                size_t tslicePitch= tpitch*d_sz.y*k;
                size_t tcolPitch= j*tpitch;
                char *t_ptr= (char *)(tens.ptr);
                char * slice_t= t_ptr+  tslicePitch;
                float * row_t= (float *)(slice_t+ tcolPitch);


                float mat[3][3];
                mat[0][0]=row_t[6*i+0];mat[0][1]=row_t[6*i+1];mat[0][2]=row_t[6*i+2];
                mat[1][0]=row_t[6*i+1];mat[1][1]=row_t[6*i+3];mat[1][2]=row_t[6*i+4];
                mat[2][0]=row_t[6*i+2];mat[2][1]=row_t[6*i+4];mat[2][2]=row_t[6*i+5];
                
                
                float R[3][3]={0};
                R[0][0]= drotmat[0]; R[0][1]= drotmat[1]; R[0][2]= drotmat[2];
                R[1][0]= drotmat[3]; R[1][1]= drotmat[4]; R[1][2]= drotmat[5];
                R[2][0]= drotmat[6]; R[2][1]= drotmat[7]; R[2][2]= drotmat[8];  
                
                float RT[3][3]={0};
                MatrixTranspose(R,RT);              
                
                float temp[3][3]={0};
                float final[3][3]={0};
                MatrixMultiply(RT,mat,temp);
                MatrixMultiply(temp,R,final);
                
                row_o[6*i+0] = final[0][0];
                row_o[6*i+1] = final[0][1];
                row_o[6*i+2] = final[0][2];
                row_o[6*i+3] = final[1][1];
                row_o[6*i+4] = final[1][2];
                row_o[6*i+5] = final[2][2];
           }
    }
}



void RotateTensors_cuda(cudaPitchedPtr tens, cudaPitchedPtr output, const int3 data_sz, float rotmat_arr[] )
{
    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z/PER_SLICE) );

    ExpTensor_kernel<<< blockSize,gridSize>>>(tens,output,data_sz);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

        
    float *d_rotmat;
    cudaMalloc((void**)&d_rotmat, sizeof(float)*9);
    cudaMemcpy(d_rotmat, rotmat_arr, sizeof(float)*9, cudaMemcpyHostToDevice);
    

    RotateTensors_kernel<<< blockSize,gridSize>>>(tens,output,data_sz, d_rotmat);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    cudaFree(d_rotmat);

}











#endif
