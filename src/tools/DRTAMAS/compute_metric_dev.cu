#ifndef _COMPUTEMETRICDEV_CU
#define _COMPUTEMETRICDEV_CU


#include "cuda_utils.h"

#undef __CUDACC_VER__

#include <iostream>
#include <Eigen/Dense>
#include <cuda_runtime.h>


#define BLOCKSIZE 256
#define PER_SLICE 1
#define PER_GROUP 1

extern __constant__ int d_sz[3];
extern __constant__ float d_dir[9];
extern __constant__ float d_spc[3];

const int bSize2=1024 ;
const int gSize2=24 ;


#define DPHI 0.1

__device__ bool tensonly;

using namespace Eigen;


extern __global__ void
ScalarFindSum(const float *gArr, int arraySize,  float *gOut);

extern __device__ void ComputeEigens(float mat[3][3], float *vals, float vecs[3][3]);


__device__ float3 ComputeImageGradient(cudaPitchedPtr img,int i, int j, int k, int v)
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
            grad.x= 0.5*(row[6*(i+1)+v]-row[6*(i-1)+v])/d_spc[0];
        }
        {
            size_t colPitch= (j+1)*pitch;
            float * row= (float *)(slice+ colPitch);
            grad.y= row[6*i+v];

            colPitch= (j-1)*pitch;
            row= (float *)(slice+ colPitch);
            grad.y= 0.5*(grad.y- row[6*i+v])/d_spc[1];
        }
    }


    {
        size_t slicePitch= pitch*d_sz[1]*(k+1);
        char * slice= ptr+  slicePitch;
        size_t colPitch= j*pitch;
        float * row= (float *)(slice+ colPitch);
        grad.z= row[6*i+v];

        slicePitch= pitch*d_sz[1]*(k-1);
        slice= ptr+  slicePitch;
        row= (float *)(slice+ colPitch);
        grad.z= 0.5*(grad.z -row[6*i+v])/d_spc[2];
    }

    float3 grad2;

    grad2.x= d_dir[0]*grad.x + d_dir[1]*grad.y + d_dir[2]*grad.z;
    grad2.y= d_dir[3]*grad.x + d_dir[4]*grad.y + d_dir[5]*grad.z;
    grad2.z= d_dir[6]*grad.x + d_dir[7]*grad.y + d_dir[8]*grad.z;

    return grad2;
}



__device__ void MatrixMultiply(float A[3][3], float B[3][3], float C[3][3])
{
    C[0][0]= A[0][0]*B[0][0] +  A[0][1]*B[1][0] + A[0][2]*B[2][0];  C[0][1]= A[0][0]*B[0][1] +  A[0][1]*B[1][1] + A[0][2]*B[2][1]; C[0][2]= A[0][0]*B[0][2] +  A[0][1]*B[1][2] + A[0][2]*B[2][2];
    C[1][0]= A[1][0]*B[0][0] +  A[1][1]*B[1][0] + A[1][2]*B[2][0];  C[1][1]= A[1][0]*B[0][1] +  A[1][1]*B[1][1] + A[1][2]*B[2][1]; C[1][2]= A[1][0]*B[0][2] +  A[1][1]*B[1][2] + A[1][2]*B[2][2];
    C[2][0]= A[2][0]*B[0][0] +  A[2][1]*B[1][0] + A[2][2]*B[2][0];  C[2][1]= A[2][0]*B[0][1] +  A[2][1]*B[1][1] + A[2][2]*B[2][1]; C[2][2]= A[2][0]*B[0][2] +  A[2][1]*B[1][2] + A[2][2]*B[2][2];
}

__device__ void MatrixTranspose(float A[3][3], float AT[3][3])
{
    AT[0][0]=A[0][0];AT[0][1]=A[1][0];AT[0][2]=A[2][0];
    AT[1][0]=A[0][1];AT[1][1]=A[1][1];AT[1][2]=A[2][1];
    AT[2][0]=A[0][2];AT[2][1]=A[1][2];AT[2][2]=A[2][2];
}



__device__ Matrix3f ComputeSingleJacobianMatrixAtIndex(cudaPitchedPtr field ,int i, int j,int k)
{
    Matrix3f B= Matrix3f::Identity();

    if(i<1 || i> d_sz[0]-2 || j<1 || j> d_sz[1]-2 || k<1 || k> d_sz[2]-2)
        return B;

    Matrix3f A;
    Matrix3f SD;
    SD(0,0)=d_dir[0]/d_spc[0];   SD(0,1)=d_dir[3]/d_spc[0];   SD(0,2)=d_dir[6]/d_spc[0];
    SD(1,0)=d_dir[1]/d_spc[1];   SD(1,1)=d_dir[4]/d_spc[1];   SD(1,2)=d_dir[7]/d_spc[1];
    SD(2,0)=d_dir[2]/d_spc[2];   SD(2,1)=d_dir[5]/d_spc[2];   SD(2,2)=d_dir[8]/d_spc[2];

    float grad;
    {
        size_t pitch= field.pitch;
        char *ptr= (char *)(field.ptr);
        size_t slicePitch= pitch*d_sz[1]*k;
        size_t colPitch= j*pitch;
        char * slice= ptr+  slicePitch;
        float * row= (float *)(slice+ colPitch);

        grad=0.5*(row[3*(i+1)]- row[3*(i-1)]);
        A(0,0)=grad;
        grad=0.5*(row[3*(i+1)+1]- row[3*(i-1)+1]);
        A(1,0)=grad;
        grad=0.5*(row[3*(i+1)+2]- row[3*(i-1)+2]);
        A(2,0)=grad;
    }

    {
        size_t pitch= field.pitch;
        char *ptr= (char *)(field.ptr);
        size_t slicePitch= pitch*d_sz[1]*k;
        char * slice= ptr+  slicePitch;

        size_t colPitch1= (j+1)*pitch;
        float * row1= (float *)(slice+ colPitch1);
        size_t colPitch2= (j-1)*pitch;
        float * row2= (float *)(slice+ colPitch2);

        grad=0.5*(row1[3*i]-row2[3*i]);
        A(0,1)=grad;
        grad=0.5*(row1[3*i+1]-row2[3*i+1]);
        A(1,1)=grad;
        grad=0.5*(row1[3*i+2]-row2[3*i+2]);
        A(2,1)=grad;
    }

    {
        size_t pitch= field.pitch;
        char *ptr= (char *)(field.ptr);
        size_t slicePitch1= pitch*d_sz[1]*(k+1);
        char * slice1= ptr+  slicePitch1;
        size_t slicePitch2= pitch*d_sz[1]*(k-1);
        char * slice2= ptr+  slicePitch2;

        size_t colPitch= j*pitch;
        float * row1= (float *)(slice1+ colPitch);
        float * row2= (float *)(slice2+ colPitch);

        grad=0.5*(row1[3*i]-row2[3*i]);
        A(0,2)=grad;
        grad=0.5*(row1[3*i+1]-row2[3*i+1]);
        A(1,2)=grad;
        grad=0.5*(row1[3*i+2]-row2[3*i+2]);
        A(2,2)=grad;
    }


    B= A * SD;
    B(0,0)+=1;
    B(1,1)+=1;
    B(2,2)+=1;

    return B;
}





__global__ void
ComputeDeviatoric_kernel( cudaPitchedPtr img)
{
    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {

            if(i>=1 && j>=1 && k>=1 && i<=d_sz[0]-2  && j<=d_sz[1]-2  && k<=d_sz[2]-2)
            {
                size_t upitch= img.pitch;
                size_t uslicePitch= upitch*d_sz[1]*k;
                size_t ucolPitch= j*upitch;
                char *u_ptr= (char *)(img.ptr);
                char * slice_u= u_ptr+  uslicePitch;
                float * row_img= (float *)(slice_u+ ucolPitch);

                float MD = (row_img[6*i+0] + row_img[6*i+3] + row_img[6*i+5])/3.;
                row_img[6*i+0]= row_img[6*i+0] -MD;
                row_img[6*i+3]= row_img[6*i+3] -MD;
                row_img[6*i+5]= row_img[6*i+5] -MD;
            }
        }
    }
}


__device__ Matrix3f ComputeRotationMatrix(Matrix3f &A)
{
    Matrix3f R=Matrix3f::Identity();

    Matrix3d Ab = A.cast <double> ();
    Matrix3d AAT= Ab * Ab.transpose();

    float AATd[3][3];
    for(int r=0;r<3;r++)
        for(int c=0;c<3;c++)
            AATd[r][c]=AAT(r,c);

    float vals[3]={0};
    float vecs[3][3]={0};
    float valsm[3][3]={0};

    ComputeEigens(AATd, vals, vecs);
    for(int d=0;d<3;d++)
    {
        if(fabs(vals[d])>1E-2)
            valsm[d][d]= pow(vals[d],-0.5);
        else
        {
            return R;
        }
    }

    float UT[3][3];
    MatrixTranspose(vecs,UT);

    float temp[3][3]={0}, temp2[3][3]={0};
    MatrixMultiply(vecs,valsm,temp);
    MatrixMultiply(temp,UT,temp2);

    Matrix3f isq;

    for(int r=0;r<3;r++)
        for(int c=0;c<3;c++)
            isq(r,c)= temp2[r][c];

    R= isq * A;

    return R;


}


__device__ Matrix3f ComputeDelRDelA(Matrix3f &A,int r, int c)
{
    Matrix3f At= A;
    At(r,c)+=DPHI;
    Matrix3f Rp = ComputeRotationMatrix(At);
    At(r,c)-=2*DPHI;
    Matrix3f Rm = ComputeRotationMatrix(At);

    Matrix3f res= 0.5*(Rp-Rm)/DPHI;
    return res;
}


__device__ Matrix3f ComputeDelEQDelR(Matrix3f &A, Matrix3f &R,int r, int c)
{
    Matrix3f res=Matrix3f::Zero();

    if(r==0 && c==0)
    {
        res(0,0)=2*A(0,0)*R(0,0) + 2*A(0,1)*R(1,0) + 2*A(0,2)*R(2,0); res(0,1) = A(0,0)*R(0,1) + A(0,1)*R(1,1) + A(0,2)*R(2,1); res(0,2)= A(0,0)*R(0,2) + A(0,1)*R(1,2) + A(0,2)*R(2,2);
        res(1,0)= A(0,0)*R(0,1) + A(0,1)*R(1,1) + A(0,2)*R(2,1);
        res(2,0)= A(0,0)*R(0,2) + A(0,1)*R(1,2) + A(0,2)*R(2,2);
    }
    if(r==0 && c==1)
    {
                                                                  res(0,1)= A(0,0)*R(0,0) + A(0,1)*R(1,0) + A(0,2)*R(2,0);
        res(1,0)=A(0,0)*R(0,0) + A(0,1)*R(1,0) + A(0,2)*R(2,0); res(1,1)= 2*A(0,0)*R(0,1) + 2*A(0,1)*R(1,1) + 2*A(0,2)*R(2,1); res(1,2) = A(0,0)*R(0,2) + A(0,1)*R(1,2) + A(0,2)*R(2,2);
                                                             res(2,1)= A(0,0)*R(0,2) + A(0,1)*R(1,2) + A(0,2)*R(2,2);
    }
    if(r==0 && c==2)
    {
        res(0,2)=A(0,0)*R(0,0) + A(0,1)*R(1,0) + A(0,2)*R(2,0);
        res(1,2)=A(0,0)*R(0,1) + A(0,1)*R(1,1) + A(0,2)*R(2,1);
        res(2,0)=A(0,0)*R(0,0) + A(0,1)*R(1,0) + A(0,2)*R(2,0); res(2,1)=A(0,0)*R(0,1)+A(0,1)*R(1,1)+ A(0,2)*R(2,1); 2*A(0,0)*R(0,2)+ 2*A(0,1)*R(1,2)+ 2* A(0,2)*R(2,2);
    }

    if(r==1 && c==0)
    {
        res(0,0)=2*A(0,1)*R(0,0) + 2*A(1,1)*R(1,0) + 2*A(1,2)*R(2,0);  res(0,1) = A(0,1)*R(0,1) + A(1,1)*R(1,1) + A(1,2)*R(2,1); res(0,2)= A(0,1)*R(0,2) + A(1,1)*R(1,2) + A(1,2)*R(2,2);
        res(1,0)= A(0,1)*R(0,1) + A(1,1)*R(1,1) + A(1,2)*R(2,1);
        res(2,0)= A(0,1)*R(0,2) + A(1,1)*R(1,2) + A(1,2)*R(2,2);
    }
    if(r==1 && c==1)
    {
        res(0,1)= A(0,1)*R(0,0) + A(1,1)*R(1,0) + A(1,2)*R(2,0);
        res(1,0)=A(0,1)*R(0,0) + A(1,1)*R(1,0) + A(1,2)*R(2,0); res(1,1)= 2*A(0,1)*R(0,1) + 2*A(1,1)*R(1,1) + 2*A(1,2)*R(2,1); res(1,2) = A(0,1)*R(0,2) + A(1,1)*R(1,2) + A(1,2)*R(2,2);
        res(2,1)= A(0,1)*R(0,2) + A(1,1)*R(1,2) + A(1,2)*R(2,2);
    }
    if(r==1 && c==2)
    {
        res(0,2)=A(0,1)*R(0,0) + A(1,1)*R(1,0) + A(1,2)*R(2,0);
        res(1,2)=A(0,1)*R(0,1) + A(1,1)*R(1,1) + A(1,2)*R(2,1);
        res(2,0)=A(0,1)*R(0,0) + A(1,1)*R(1,0) + A(1,2)*R(2,0); res(2,1)=A(0,1)*R(0,1)+A(1,1)*R(1,1)+ A(1,2)*R(2,1); 2*A(0,1)*R(0,2)+ 2*A(1,1)*R(1,2)+ 2* A(1,2)*R(2,2);
    }
    if(r==2 && c==0)
    {
        res(0,0)=2*A(0,2)*R(0,0) + 2*A(1,2)*R(1,0) + 2*A(2,2)*R(2,0); res(0,1) = A(0,2)*R(0,1) + A(1,2)*R(1,1) + A(2,2)*R(2,1); res(0,2)= A(0,2)*R(0,2) + A(1,2)*R(1,2) + A(2,2)*R(2,2);
        res(1,0)= A(0,2)*R(0,1) + A(1,2)*R(1,1) + A(2,2)*R(2,1);
        res(2,0)= A(0,2)*R(0,2) + A(1,2)*R(1,2) + A(2,2)*R(2,2);
    }
    if(r==2 && c==1)
    {
        res(0,1)= A(0,2)*R(0,0) + A(1,2)*R(1,0) + A(2,2)*R(2,0);
        res(1,0)=A(0,2)*R(0,0) + A(1,2)*R(1,0) + A(2,2)*R(2,0); res(1,1)= 2*A(0,2)*R(0,1) + 2*A(1,2)*R(1,1) + 2*A(2,2)*R(2,1); res(1,2) = A(0,2)*R(0,2) + A(1,2)*R(1,2) + A(2,2)*R(2,2);
        res(2,1)= A(0,2)*R(0,2) + A(1,2)*R(1,2) + A(2,2)*R(2,2);
    }
    if(r==2 && c==2)
    {
        res(0,2)=A(0,2)*R(0,0) + A(1,2)*R(1,0) + A(2,2)*R(2,0);
        res(1,2)=A(0,2)*R(0,1) + A(1,2)*R(1,1) + A(2,2)*R(2,1);
        res(2,0)=A(0,2)*R(0,0) + A(1,2)*R(1,0) + A(2,2)*R(2,0); res(2,1)=A(0,2)*R(0,1)+A(1,2)*R(1,1)+ A(2,2)*R(2,1); 2*A(0,2)*R(0,2)+ 2*A(1,2)*R(1,2)+ 2* A(2,2)*R(2,2);
    }

    return res;
}

__global__ void
ComputeMetric_DEV_kernel( cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                            cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV, 
                            cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                            cudaPitchedPtr metric_image,
                         cudaPitchedPtr Rf_img, cudaPitchedPtr Rm_img,
                         cudaPitchedPtr derf_img, cudaPitchedPtr derm_img)

{

  //  if(se==1)
    //    asm("exit;");

    uint ii2 = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    Matrix3f SD;
    SD(0,0)=d_dir[0]/d_spc[0];   SD(0,1)=d_dir[3]/d_spc[0];   SD(0,2)=d_dir[6]/d_spc[0];
    SD(1,0)=d_dir[1]/d_spc[1];   SD(1,1)=d_dir[4]/d_spc[1];   SD(1,2)=d_dir[7]/d_spc[1];
    SD(2,0)=d_dir[2]/d_spc[2];   SD(2,1)=d_dir[5]/d_spc[2];   SD(2,2)=d_dir[8]/d_spc[2];    

    for(int i=PER_GROUP*ii2;i<PER_GROUP*ii2+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            float updateF[3]={0,0,0};
            float updateM[3]={0,0,0};

            if(i>=1 && j>=1 && k>=1 && i<=d_sz[0]-2  && j<=d_sz[1]-2  && k<=d_sz[2]-2)
            {                


                size_t fpitch= up_img.pitch;
                size_t fslicePitch= fpitch*d_sz[1]*k;
                size_t fcolPitch= j*fpitch;
                char *f_ptr= (char *)(up_img.ptr);
                char * slice_f= f_ptr+  fslicePitch;
                float * row_f= (float *)(slice_f+ fcolPitch);

                size_t mpitch= down_img.pitch;
                size_t mslicePitch= mpitch*d_sz[1]*k;
                size_t mcolPitch= j*mpitch;
                char *m_ptr= (char *)(down_img.ptr);
                char * slice_m= m_ptr+  mslicePitch;
                float * row_m= (float *)(slice_m+ mcolPitch);


                size_t metpitch= metric_image.pitch;
                size_t metslicePitch= metpitch*d_sz[1]*k;
                size_t metcolPitch= j*metpitch;
                char *met_ptr= (char *)(metric_image.ptr);
                char * slice_met= met_ptr+  metslicePitch;
                float * row_metric= (float *)(slice_met+ metcolPitch);




                //////////////////////////at x/////////////////////////////////
                {

                    size_t Rfpitch= Rf_img.pitch;
                    size_t RfslicePitch= Rfpitch*d_sz[1]*k;
                    size_t RfcolPitch= j*Rfpitch;
                    char *Rf_ptr= (char *)(Rf_img.ptr);
                    char * slice_Rf= Rf_ptr+  RfslicePitch;
                    float * row_Rf= (float *)(slice_Rf+ RfcolPitch);

                    size_t Rmpitch= Rm_img.pitch;
                    size_t RmslicePitch= Rmpitch*d_sz[1]*k;
                    size_t RmcolPitch= j*Rmpitch;
                    char *Rm_ptr= (char *)(Rm_img.ptr);
                    char * slice_Rm= Rm_ptr+  RmslicePitch;
                    float * row_Rm= (float *)(slice_Rm+ RmcolPitch);


                    /////////////////// Metric computation////////////////

                    Matrix3f F,Fi,M,Mi;


                    F(0,0)=row_f[6*i+0]; F(0,1)=row_f[6*i+1]; F(0,2)=row_f[6*i+2];
                    F(1,0)=row_f[6*i+1]; F(1,1)=row_f[6*i+3]; F(1,2)=row_f[6*i+4];
                    F(2,0)=row_f[6*i+2]; F(2,1)=row_f[6*i+4]; F(2,2)=row_f[6*i+5];

                    M(0,0)=row_m[6*i+0]; M(0,1)=row_m[6*i+1]; M(0,2)=row_m[6*i+2];
                    M(1,0)=row_m[6*i+1]; M(1,1)=row_m[6*i+3]; M(1,2)=row_m[6*i+4];
                    M(2,0)=row_m[6*i+2]; M(2,1)=row_m[6*i+4]; M(2,2)=row_m[6*i+5];

                    Matrix3f Rf,Rm, Af,Am;

                    float metric_val=0;
                    if(!tensonly)
                    {
                        Rf(0,0)=row_Rf[9*i+0];Rf(0,1)=row_Rf[9*i+3];Rf(0,2)=row_Rf[9*i+6];
                        Rf(1,0)=row_Rf[9*i+1];Rf(1,1)=row_Rf[9*i+4];Rf(1,2)=row_Rf[9*i+7];
                        Rf(2,0)=row_Rf[9*i+2];Rf(2,1)=row_Rf[9*i+5];Rf(2,2)=row_Rf[9*i+8];

                        Rm(0,0)=row_Rm[9*i+0];Rm(0,1)=row_Rm[9*i+3];Rm(0,2)=row_Rm[9*i+6];
                        Rm(1,0)=row_Rm[9*i+1];Rm(1,1)=row_Rm[9*i+4];Rm(1,2)=row_Rm[9*i+7];
                        Rm(2,0)=row_Rm[9*i+2];Rm(2,1)=row_Rm[9*i+5];Rm(2,2)=row_Rm[9*i+8];

                        Fi = Rf.transpose()* F *  Rf ;
                        Mi = Rm.transpose()* M *  Rm ;
                        Matrix3f diff = Fi- Mi;
                        metric_val= diff.squaredNorm();
                    }
                    else
                    {
                        Matrix3f diff = F - M;
                        metric_val= diff.squaredNorm();
                    }

                    metric_val=sqrt(metric_val);
                    row_metric[i]=metric_val;

                    /////////////////Gradient computation///////////////////////////

                    float gradI[3][6]={0}, gradJ[3][6]={0};
                    for(int v=0;v<6;v++)
                    {
                        float3 gradIt= ComputeImageGradient(up_img,i,j,k,v);
                        float3 gradJt= ComputeImageGradient(down_img,i,j,k,v);

                        gradI[0][v]=gradIt.x;
                        gradI[1][v]=gradIt.y;
                        gradI[2][v]=gradIt.z;

                        gradJ[0][v]=gradJt.x;
                        gradJ[1][v]=gradJt.y;
                        gradJ[2][v]=gradJt.z;
                    }

                    for(int gdim=0;gdim<3;gdim++)
                    {
                        Matrix3f F2, M2,F2i, M2i;

                        /////for i
                        F2(0,0)=gradI[gdim][0]; F2(0,1)=gradI[gdim][1]; F2(0,2)=gradI[gdim][2];
                        F2(1,0)=gradI[gdim][1]; F2(1,1)=gradI[gdim][3]; F2(1,2)=gradI[gdim][4];
                        F2(2,0)=gradI[gdim][2]; F2(2,1)=gradI[gdim][4]; F2(2,2)=gradI[gdim][5];

                        M2(0,0)=gradJ[gdim][0]; M2(0,1)=gradJ[gdim][1]; M2(0,2)=gradJ[gdim][2];
                        M2(1,0)=gradJ[gdim][1]; M2(1,1)=gradJ[gdim][3]; M2(1,2)=gradJ[gdim][4];
                        M2(2,0)=gradJ[gdim][2]; M2(2,1)=gradJ[gdim][4]; M2(2,2)=gradJ[gdim][5];


                        float smf,smm;

                        if(!tensonly)
                        {
                            F2i= Rf.transpose() * F2 * Rf;
                            M2i= Rm.transpose() * M2 * Rm;

                            Matrix3f diff= Fi- Mi;

                            Matrix3f resf= diff.array() * F2i.array() ;
                            Matrix3f resm= diff.array() * M2i.array() ;

                            smf= 2* resf.sum();
                            smm= 2* resm.sum();
                        }
                        else
                        {
                            smf= 2 * ( (F(0,0)-M(0,0)) * gradI[gdim][0] +
                                       (F(1,1)-M(1,1)) * gradI[gdim][3] +
                                       (F(2,2)-M(2,2)) * gradI[gdim][5] +
                                       2*(F(0,1)-M(0,1)) * gradI[gdim][1] +
                                       2*(F(0,2)-M(0,2)) * gradI[gdim][2] +
                                       2*(F(1,2)-M(1,2)) * gradI[gdim][4]
                                       ) ;

                            smm= 2 * ( (F(0,0)-M(0,0)) * gradJ[gdim][0] +
                                       (F(1,1)-M(1,1)) * gradJ[gdim][3] +
                                       (F(2,2)-M(2,2)) * gradJ[gdim][5] +
                                       2*(F(0,1)-M(0,1)) * gradJ[gdim][1] +
                                       2*(F(0,2)-M(0,2)) * gradJ[gdim][2] +
                                       2*(F(1,2)-M(1,2)) * gradJ[gdim][4]
                                       ) ;

                        }

                        updateF[gdim]= smf;
                        updateM[gdim]= -smm;
                    } //for gdim
                } // at x


                if(!tensonly)
                {
                    ///////////////////////////for neighbors of x //////////////////////////
                    for(int dim=0;dim<3;dim++)
                    {
                        for(int pn=-1;pn<=1;pn+=2)
                        {

                            int ii=i;
                            int jj=j;
                            int kk=k;
                            if(dim==0)
                                ii+=pn;
                            if(dim==1)
                                jj+=pn;
                            if(dim==2)
                                kk+=pn;

                            size_t derfpitch= derf_img.pitch;
                            size_t derfslicePitch= derfpitch*d_sz[1]*kk;
                            size_t derfcolPitch= jj*derfpitch;
                            char *derf_ptr= (char *)(derf_img.ptr);
                            char * slice_derf= derf_ptr+  derfslicePitch;
                            float * row_derf= (float *)(slice_derf+ derfcolPitch);

                            size_t dermpitch= derm_img.pitch;
                            size_t dermslicePitch= dermpitch*d_sz[1]*kk;
                            size_t dermcolPitch= jj*dermpitch;
                            char *derm_ptr= (char *)(derm_img.ptr);
                            char * slice_derm= derm_ptr+  dermslicePitch;
                            float * row_derm= (float *)(slice_derm+ dermcolPitch);


                            for(int px=0;px<3;px++)
                            {
                                int id= dim*3+px;

                                float delf= row_derf[9*ii+id]* -1*pn* 0.5/d_spc[dim];
                                float delm= row_derm[9*ii+id]* -1*pn* 0.5/d_spc[dim];

                                updateF[px]+=delf;
                                updateM[px]-=delm;
                            }
                        } //for pn
                    } //for dim

                } //if tensoronly




                size_t upitch= updateFieldF.pitch;
                size_t uslicePitch= upitch*d_sz[1]*k;
                size_t ucolPitch= j*upitch;
                char *u_ptr= (char *)(updateFieldF.ptr);
                char * slice_u= u_ptr+  uslicePitch;
                float * row_uf= (float *)(slice_u+ ucolPitch);

                size_t dpitch= updateFieldM.pitch;
                size_t dslicePitch= dpitch*d_sz[1]*k;
                size_t dcolPitch= j*dpitch;
                char *d_ptr= (char *)(updateFieldM.ptr);
                char * slice_d= d_ptr+  dslicePitch;
                float * row_df= (float *)(slice_d+ dcolPitch);

                row_uf[3*i]= updateF[0];
                row_uf[3*i+1]= updateF[1];
                row_uf[3*i+2]= updateF[2];

                row_df[3*i]= updateM[0];
                row_df[3*i+1]= updateM[1];
                row_df[3*i+2]= updateM[2];
            }
        }
     }
}


__global__ void
FillImgs_kernel ( cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                  cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV,
                  cudaPitchedPtr Rf_img, cudaPitchedPtr Rm_img,
                  cudaPitchedPtr derf_img, cudaPitchedPtr derm_img)
{
    uint ii2 = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii2;i<PER_GROUP*ii2+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            if(i>=1 && j>=1 && k>=1 && i<=d_sz[0]-2  && j<=d_sz[1]-2  && k<=d_sz[2]-2)
            {


                size_t fpitch= up_img.pitch;
                size_t fslicePitch= fpitch*d_sz[1]*k;
                size_t fcolPitch= j*fpitch;
                char *f_ptr= (char *)(up_img.ptr);
                char * slice_f= f_ptr+  fslicePitch;
                float * row_f= (float *)(slice_f+ fcolPitch);

                size_t mpitch= down_img.pitch;
                size_t mslicePitch= mpitch*d_sz[1]*k;
                size_t mcolPitch= j*mpitch;
                char *m_ptr= (char *)(down_img.ptr);
                char * slice_m= m_ptr+  mslicePitch;
                float * row_m= (float *)(slice_m+ mcolPitch);

                size_t Rfpitch= Rf_img.pitch;
                size_t RfslicePitch= Rfpitch*d_sz[1]*k;
                size_t RfcolPitch= j*Rfpitch;
                char *Rf_ptr= (char *)(Rf_img.ptr);
                char * slice_Rf= Rf_ptr+  RfslicePitch;
                float * row_Rf= (float *)(slice_Rf+ RfcolPitch);

                size_t Rmpitch= Rm_img.pitch;
                size_t RmslicePitch= Rmpitch*d_sz[1]*k;
                size_t RmcolPitch= j*Rmpitch;
                char *Rm_ptr= (char *)(Rm_img.ptr);
                char * slice_Rm= Rm_ptr+  RmslicePitch;
                float * row_Rm= (float *)(slice_Rm+ RmcolPitch);

                size_t derfpitch= derf_img.pitch;
                size_t derfslicePitch= derfpitch*d_sz[1]*k;
                size_t derfcolPitch= j*derfpitch;
                char *derf_ptr= (char *)(derf_img.ptr);
                char * slice_derf= derf_ptr+  derfslicePitch;
                float * row_derf= (float *)(slice_derf+ derfcolPitch);

                size_t dermpitch= derm_img.pitch;
                size_t dermslicePitch= dermpitch*d_sz[1]*k;
                size_t dermcolPitch= j*dermpitch;
                char *derm_ptr= (char *)(derm_img.ptr);
                char * slice_derm= derm_ptr+  dermslicePitch;
                float * row_derm= (float *)(slice_derm+ dermcolPitch);

                Matrix3f Af= ComputeSingleJacobianMatrixAtIndex(def_FINV,i,j,k);
                Matrix3f Am= ComputeSingleJacobianMatrixAtIndex(def_MINV,i,j,k);

                Matrix3f Rf= ComputeRotationMatrix(Af);
                Matrix3f Rm= ComputeRotationMatrix(Am);

                Matrix3f F,M;


                F(0,0)=row_f[6*i+0]; F(0,1)=row_f[6*i+1]; F(0,2)=row_f[6*i+2];
                F(1,0)=row_f[6*i+1]; F(1,1)=row_f[6*i+3]; F(1,2)=row_f[6*i+4];
                F(2,0)=row_f[6*i+2]; F(2,1)=row_f[6*i+4]; F(2,2)=row_f[6*i+5];

                M(0,0)=row_m[6*i+0]; M(0,1)=row_m[6*i+1]; M(0,2)=row_m[6*i+2];
                M(1,0)=row_m[6*i+1]; M(1,1)=row_m[6*i+3]; M(1,2)=row_m[6*i+4];
                M(2,0)=row_m[6*i+2]; M(2,1)=row_m[6*i+4]; M(2,2)=row_m[6*i+5];

                Matrix3f Fi= Rf.transpose()* F * Rf;
                Matrix3f Mi= Rm.transpose()* M * Rm;
                Matrix3f diff= Fi -Mi;
                VectorXf delF_delEQ(9);

                int cnt=0;
                for(int c=0;c<3;c++)
                {
                    for(int r=0;r<3;r++)
                    {
                        delF_delEQ(cnt)= 2*diff(r,c);
                        cnt++;
                    }
                }


                MatrixXf delEQ_delRf(9,9),delEQ_delRm(9,9);
                cnt=0;
                for(int c=0;c<3;c++)
                {
                    for(int r=0;r<3;r++)
                    {
                        Matrix3f derf= ComputeDelEQDelR(F,Rf,r,c);
                        Matrix3f derm= ComputeDelEQDelR(M,Rm,r,c);

                        int ma=0;
                        for(int c2=0;c2<3;c2++)
                        {
                            for(int r2=0;r2<3;r2++)
                            {
                                delEQ_delRf(cnt,ma) =  derf(r2,c2);
                                delEQ_delRm(cnt,ma) =  derm(r2,c2);
                                ma++;
                            }
                        }
                        cnt++;
                    }
                }


                MatrixXf delR_delAf(9,9),delR_delAm(9,9);
                cnt=0;
                for(int c=0;c<3;c++)
                {
                    for(int r=0;r<3;r++)
                    {
                        Matrix3f derf= ComputeDelRDelA(Af,r,c);
                        Matrix3f derm= ComputeDelRDelA(Am,r,c);

                        int ma=0;
                        for(int c2=0;c2<3;c2++)
                        {
                            for(int r2=0;r2<3;r2++)
                            {
                                delR_delAf(cnt,ma)=derf(r2,c2);
                                delR_delAm(cnt,ma)=derm(r2,c2);
                                ma++;
                            }
                        }

                        cnt++;
                    }
                }

                VectorXf del_so_far_f= (delF_delEQ.transpose()* delEQ_delRf) * delR_delAf;
                VectorXf del_so_far_m= (delF_delEQ.transpose()* delEQ_delRm) * delR_delAm;


                for(int p=0;p<9;p++)
                {
                    row_Rf[9*i+p]= Rf.data()[p];
                    row_Rm[9*i+p]= Rm.data()[p];

                    row_derf[9*i+p]= del_so_far_f(p);
                    row_derm[9*i+p]= del_so_far_m(p);
                }
            }
        }
    }
}




void ComputeMetric_DEV_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img,
		   int3 data_sz, float3 data_spc, 
		   float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
		   cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV,
                   cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                   float &metric_value, bool to
             )
{
    float h_d_dir[]= {d00,d01,d02,d10,d11,d12,d20,d21,d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));      
    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));      
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));



    bool tensonly_h=to;
    gpuErrchk(cudaMemcpyToSymbol(tensonly, &tensonly_h,  sizeof(bool)));



    cudaPitchedPtr metric_image={0};
    cudaExtent extent =  make_cudaExtent(up_img.pitch,data_sz.y,data_sz.z);
    cudaMalloc3D(&metric_image, extent);
    cudaMemset3D(metric_image,0,extent);


    cudaPitchedPtr Rf_img={0},Rm_img={0}, derf_img={0},derm_img={0};
    if(!tensonly_h)
    {
        cudaExtent extent2 =  make_cudaExtent(9*data_sz.x*sizeof(float),data_sz.y,data_sz.z);
        gpuErrchk(cudaMalloc3D(&Rf_img, extent2));
        cudaMemset3D(Rf_img,0,extent2);
        gpuErrchk(cudaMalloc3D(&Rm_img, extent2));
        cudaMemset3D(Rm_img,0,extent2);

        gpuErrchk(cudaMalloc3D(&derf_img, extent2));
        cudaMemset3D(derf_img,0,extent2);
        gpuErrchk(cudaMalloc3D(&derm_img, extent2));
        cudaMemset3D(derm_img,0,extent2);
    }


    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x/PER_GROUP), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z) );

    std::cout<<blockSize.x<<" " <<blockSize.y<<" " <<blockSize.z<<" "  <<std::endl;
    std::cout<<gridSize.x<<" " <<gridSize.y<<" " <<gridSize.z<<" "  <<std::endl;

    if(!tensonly_h)
    {
        FillImgs_kernel<<< blockSize,gridSize>>> (up_img, down_img, def_FINV,def_MINV,
                                              Rf_img,Rm_img,
                                             derf_img,derm_img);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    ComputeMetric_DEV_kernel<<< blockSize,gridSize>>>( up_img, down_img, def_FINV,def_MINV, updateFieldF, updateFieldM, metric_image,Rf_img,Rm_img,derf_img,derm_img);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());



    float* dev_out;
    float out;
    cudaMalloc((void**)&dev_out, sizeof(float)*gSize2);

    ScalarFindSum<<<gSize2, bSize2>>>((float *)metric_image.ptr, metric_image.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
    cudaDeviceSynchronize();
    ScalarFindSum<<<1, bSize2>>>(dev_out, gSize2, dev_out);
    cudaDeviceSynchronize();

    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_out);
    metric_value=out/data_sz.x/data_sz.y/data_sz.z;

    cudaFree(metric_image.ptr);
    cudaFree(Rf_img.ptr);
    cudaFree(Rm_img.ptr);
    cudaFree(derf_img.ptr);
    cudaFree(derm_img.ptr);


}		   







__global__ void
ComputeMetric_DEV_ONLY_kernel( cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                            cudaPitchedPtr metric_image)
{
    uint ii2 = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii2;i<PER_GROUP*ii2+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            {                


                size_t fpitch= up_img.pitch;
                size_t fslicePitch= fpitch*d_sz[1]*k;
                size_t fcolPitch= j*fpitch;
                char *f_ptr= (char *)(up_img.ptr);
                char * slice_f= f_ptr+  fslicePitch;
                float * row_f= (float *)(slice_f+ fcolPitch);

                size_t mpitch= down_img.pitch;
                size_t mslicePitch= mpitch*d_sz[1]*k;
                size_t mcolPitch= j*mpitch;
                char *m_ptr= (char *)(down_img.ptr);
                char * slice_m= m_ptr+  mslicePitch;
                float * row_m= (float *)(slice_m+ mcolPitch);


                size_t metpitch= metric_image.pitch;
                size_t metslicePitch= metpitch*d_sz[1]*k;
                size_t metcolPitch= j*metpitch;
                char *met_ptr= (char *)(metric_image.ptr);
                char * slice_met= met_ptr+  metslicePitch;
                float * row_metric= (float *)(slice_met+ metcolPitch);


                //////////////////////////at x/////////////////////////////////
                {


                    /////////////////// Metric computation////////////////
                    float F[3][3], M[3][3];
                    F[0][0]=row_f[6*i+0]; F[0][1]=row_f[6*i+1]; F[0][2]=row_f[6*i+2];
                    F[1][0]=row_f[6*i+1]; F[1][1]=row_f[6*i+3]; F[1][2]=row_f[6*i+4];
                    F[2][0]=row_f[6*i+2]; F[2][1]=row_f[6*i+4]; F[2][2]=row_f[6*i+5];

                    M[0][0]=row_m[6*i+0]; M[0][1]=row_m[6*i+1]; M[0][2]=row_m[6*i+2];
                    M[1][0]=row_m[6*i+1]; M[1][1]=row_m[6*i+3]; M[1][2]=row_m[6*i+4];
                    M[2][0]=row_m[6*i+2]; M[2][1]=row_m[6*i+4]; M[2][2]=row_m[6*i+5];


                    float metric_val = (F[0][0]-M[0][0])*(F[0][0]-M[0][0])+
                                     2*(F[0][1]-M[0][1])*(F[0][1]-M[0][1])+
                                     2*(F[0][2]-M[0][2])*(F[0][2]-M[0][2])+
                                       (F[1][1]-M[1][1])*(F[1][1]-M[1][1])+
                                     2*(F[1][2]-M[1][2])*(F[1][2]-M[1][2])+
                                       (F[2][2]-M[2][2])*(F[2][2]-M[2][2]);


                    metric_val=sqrt(metric_val);
                    row_metric[i]=metric_val;
                 }

            }
        }
     }
}






void ComputeMetric_DEV_ONLY_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img,
		   int3 data_sz, 
           float &metric_value)
{

     int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));


    cudaPitchedPtr metric_image={0};
    cudaExtent extent =  make_cudaExtent(up_img.pitch,data_sz.y,data_sz.z);
    cudaMalloc3D(&metric_image, extent);
    cudaMemset3D(metric_image,0,extent);


    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x/PER_GROUP), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z) );


    ComputeMetric_DEV_ONLY_kernel<<< blockSize,gridSize>>>( up_img, down_img, metric_image);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());



    float* dev_out;
    float out;
    cudaMalloc((void**)&dev_out, sizeof(float)*gSize2);

    ScalarFindSum<<<gSize2, bSize2>>>((float *)metric_image.ptr, metric_image.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
    cudaDeviceSynchronize();
    ScalarFindSum<<<1, bSize2>>>(dev_out, gSize2, dev_out);
    cudaDeviceSynchronize();

    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_out);
    metric_value=out/data_sz.x/data_sz.y/data_sz.z;

    cudaFree(metric_image.ptr);
}




void ComputeDeviatoricTensor_cuda(cudaPitchedPtr img,   int3 data_sz)
{
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));


    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x/PER_GROUP), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z) );

    ComputeDeviatoric_kernel<<< blockSize,gridSize>>>( img);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}






#endif
