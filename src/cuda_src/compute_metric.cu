#ifndef _COMPUTEMETRICMSJAC_CU
#define _COMPUTEMETRICMSJAC_CU

#include <stdio.h>
#include <iostream>

#include "cuda_utils.h"


#define LIMCCSK (1E-5)
#define LIMCC (1E-10)
#define LIMCCJAC (1E-5)

#define WIN_RAD 4
#define WIN_RAD_Z 2

#define WIN_RAD_JAC 4
#define WIN_RAD_JAC_Z 2



#define WIN_SZ (2*WIN_RAD+1)
#define WIN_SZ_Z (2*WIN_RAD_Z+1)

#define CENTER_IND (WIN_SZ/2)
#define CENTER_IND_Z (WIN_SZ_Z/2)


#define BLOCKSIZE 32
#define PER_GROUP 1
#define PER_SLICE 1

extern __constant__ int d_sz[3];
extern __constant__ float d_dir[9];
extern __constant__ float d_spc[3];


extern const int bSize=1024 ;
extern const int gSize=24 ;

extern __constant__  float c_Kernel[];
__constant__ float d_new_phase[3];


extern __global__ void
ScalarFindSum(const float *gArr, int arraySize,  float *gOut);


__global__ void
ComputeMetric_CC_kernel( cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                           cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                           cudaPitchedPtr metric_img);

__device__ float mf(float det)
{
    float logd = log(det);
    float ly = logd / (sqrt(1+0.2*logd*logd));
    float y= exp(ly);
    return y;
}


__device__ float dmf(float x)
{
    float y= mf(x);
    float lx= log(x);

    float nom = 1./x * sqrt(1+0.2* lx*lx) - lx *  1./sqrt(1+0.2*lx*lx) *0.2* lx *1./x;
    float denom = 1+0.2* lx*lx;

    return y*nom/denom;


}

__device__ float3 ComputeImageGradient(cudaPitchedPtr img,int i, int j, int k)
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



__device__ float ComputeSingleJacobianMatrixAtIndex(cudaPitchedPtr field ,int i, int j,int k,int h,int phase, int phase_xyz)
{
    int index[3]={i,j,k};

    if(index[phase]<h || index[phase]> d_sz[phase]-h-1)
        return 1.;


    size_t pitch= field.pitch;
    char *ptr= (char *)(field.ptr);

    float grad;

    if(phase==0)
    {
        size_t slicePitch= pitch*d_sz[1]*k;
        size_t colPitch= j*pitch;
        char * slice= ptr+  slicePitch;
        float * row= (float *)(slice+ colPitch);

         grad=0.5*(row[3*(i+h)]- row[3*(i-h)])/d_spc[0]/h;

    }
    if(phase==1)
    {
        size_t slicePitch= pitch*d_sz[1]*k;
        char * slice= ptr+  slicePitch;

        size_t colPitch1= (j+h)*pitch;
        float * row1= (float *)(slice+ colPitch1);
        size_t colPitch2= (j-h)*pitch;
        float * row2= (float *)(slice+ colPitch2);

        grad=0.5*(row1[3*i+1]-row2[3*i+1])/d_spc[1]/h;
    }
    if(phase==2)
    {
        size_t slicePitch1= pitch*d_sz[1]*(k+h);
        char * slice1= ptr+  slicePitch1;
        size_t slicePitch2= pitch*d_sz[1]*(k-h);
        char * slice2= ptr+  slicePitch2;

        size_t colPitch= j*pitch;
        float * row1= (float *)(slice1+ colPitch);
        float * row2= (float *)(slice2+ colPitch);

        grad=0.5*(row1[3*i+2]-row2[3*i+2])/d_spc[2]/h;
    }

    float temp[3]={0,0,0};
    float temp2[3]={0,0,0};

    temp[phase]=grad;
    temp2[0] = d_dir[0]*temp[0] + d_dir[1]*temp[1] + d_dir[2]*temp[2];
    temp2[1] = d_dir[3]*temp[0] + d_dir[4]*temp[1] + d_dir[5]*temp[2];
    temp2[2] = d_dir[6]*temp[0] + d_dir[7]*temp[1] + d_dir[8]*temp[2];

    return  temp2[phase_xyz];
   // return 1+ temp2[phase_xyz];

}


__global__ void
computeDetImg( cudaPitchedPtr img,cudaPitchedPtr field,cudaPitchedPtr detimg, int phase,int phase_xyz)
{
    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            size_t detIpitch= detimg.pitch;
            size_t detIslicePitch= detIpitch*d_sz[1]*k;
            size_t detIcolPitch= j*detIpitch;
            char *detI_ptr= (char *)(detimg.ptr);
            char * slice_detI= detI_ptr+  detIslicePitch;
            float * row_detI= (float *)(slice_detI+ detIcolPitch);

            size_t Ipitch= img.pitch;
            size_t IslicePitch= Ipitch*d_sz[1]*k;
            size_t IcolPitch= j*Ipitch;
            char *I_ptr= (char *)(img.ptr);
            char * slice_I= I_ptr+  IslicePitch;
            float * row_I= (float *)(slice_I+ IcolPitch);


            float det= ComputeSingleJacobianMatrixAtIndex(field,i,j,k,1,phase,phase_xyz);
            if(det <=-1)
                det=-1+1E-5;

            row_detI[i]= (1+det) * row_I[i];
        }
    }
}



__global__ void
computeFiniteDiffStructs( cudaPitchedPtr det_img,cudaPitchedPtr str_img,
                          cudaPitchedPtr sKS, cudaPitchedPtr sSS, cudaPitchedPtr sKK, cudaPitchedPtr valS, cudaPitchedPtr valK)
{
    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            char *str_ptr= (char *)(str_img.ptr);
            size_t dpitch= det_img.pitch;
            char *d_ptr= (char *)(det_img.ptr);


            size_t sSSpitch= sSS.pitch;
            size_t sSSslicePitch= sSSpitch*d_sz[1]*k;
            size_t sSScolPitch= j*sSSpitch;
            char *sSS_ptr= (char *)(sSS.ptr);
            char * slice_sSS= sSS_ptr+  sSSslicePitch;
            float * row_sSS= (float *)(slice_sSS+ sSScolPitch);

            size_t sSKpitch= sKS.pitch;
            size_t sSKslicePitch= sSKpitch*d_sz[1]*k;
            size_t sSKcolPitch= j*sSKpitch;
            char *sSK_ptr= (char *)(sKS.ptr);
            char * slice_sSK= sSK_ptr+  sSKslicePitch;
            float * row_sSK= (float *)(slice_sSK+ sSKcolPitch);

            size_t sKKpitch= sKK.pitch;
            size_t sKKslicePitch= sKKpitch*d_sz[1]*k;
            size_t sKKcolPitch= j*sKKpitch;
            char *sKK_ptr= (char *)(sKK.ptr);
            char * slice_sKK= sKK_ptr+  sKKslicePitch;
            float * row_sKK= (float *)(slice_sKK+ sKKcolPitch);

            size_t valSpitch= valS.pitch;
            size_t valSslicePitch= valSpitch*d_sz[1]*k;
            size_t valScolPitch= j*valSpitch;
            char *valS_ptr= (char *)(valS.ptr);
            char * slice_valS= valS_ptr+  valSslicePitch;
            float * row_valS= (float *)(slice_valS+ valScolPitch);

            size_t valKpitch= valK.pitch;
            size_t valKslicePitch= valKpitch*d_sz[1]*k;
            size_t valKcolPitch= j*valKpitch;
            char *valK_ptr= (char *)(valK.ptr);
            char * slice_valK= valK_ptr+  valKslicePitch;
            float * row_valK= (float *)(slice_valK+ valKcolPitch);

            int start[3],end[3];
            start[2]=k-WIN_RAD_JAC_Z;
            if(start[2]<0)
               start[2]=0;
            start[1]=j-WIN_RAD_JAC;
            if(start[1]<0)
               start[1]=0;
            start[0]=i-WIN_RAD_JAC;
            if(start[0]<0)
               start[0]=0;

            end[2]=k+WIN_RAD_JAC_Z+1;
            if(end[2]>d_sz[2])
               end[2]=d_sz[2];
            end[1]=j+WIN_RAD_JAC+1;
            if(end[1]>d_sz[1])
               end[1]=d_sz[1];
            end[0]=i+WIN_RAD_JAC+1;
            if(end[0]>d_sz[0])
               end[0]=d_sz[0];

            float suma2 = 0.0;
            float suma = 0.0;
            float  sumac=0;
            float sumc2 = 0.0;
            float sumc = 0.0;
            int N=0;

            float vald_center;
            float valS_center;

            for(int z=start[2];z<end[2];z++)
            {
                size_t dslicePitch= dpitch*d_sz[1]*z;
                char * slice_d= d_ptr+  dslicePitch;
                char * slice_str= str_ptr+  dslicePitch;

                for(int y=start[1];y<end[1];y++)
                {
                    size_t dcolPitch= y*dpitch;

                    float * row_d= (float *)(slice_d+ dcolPitch);
                    float * row_str= (float *)(slice_str+ dcolPitch);

                    for(int x=start[0];x<end[0];x++)
                    {

                        float Kim= row_d[x];
                        float c= row_str[x];

                        if(z==k && y==j && x==i)
                        {
                            vald_center=Kim;
                            valS_center=c;
                        }

                        suma2 += Kim * Kim;
                        suma += Kim;
                        sumc2 += c * c;
                        sumc += c;
                        sumac += Kim*c;

                        N++;
                    }
                }
            }
            float Umean = suma/N;
            float Smean = sumc/N;

            float valU = vald_center- Umean;
            float valS = valS_center -Smean;

            float sKK = suma2 - Umean*suma;
            float sSS = sumc2 - Smean*sumc;
            float sKS = sumac - Umean*sumc;

            row_sSS[i]=sSS;
            row_sSK[i]=sKS;
            row_sKK[i]=sKK;
            row_valS[i]=valS;
            row_valK[i]=valU;
        }
    }
}





__global__ void
ComputeMetric_CCJacSSingle_kernel( cudaPitchedPtr b0_img,  cudaPitchedPtr str_img,
                                 cudaTextureObject_t grad_img_x, cudaTextureObject_t grad_img_y, cudaTextureObject_t grad_img_z,
                                 cudaPitchedPtr field,
                            cudaPitchedPtr updateFieldINV,
                            cudaPitchedPtr metric_image,
                            int phase, int phase_xyz,
                            int kernel_sz,
                            cudaPitchedPtr sKS, cudaPitchedPtr sSS, cudaPitchedPtr sKK, cudaPitchedPtr valS, cudaPitchedPtr valK)
{
    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            float update[3]={0,0,0};

            if(i>=1 && j>=1 && k>=1 && i<=d_sz[0]-2  && j<=d_sz[1]-2  && k<=d_sz[2]-2)
            {
                //int mid_ind=(kernel_sz-1)/2;
                //float b= c_Kernel[  mid_ind ];
                //float a=0;
                //if((kernel_sz-1)/2 > 0)
               //     a=c_Kernel[  mid_ind -1];

             //   float a_b= a/b;
                float a_b=0;



                ////////////////////////   at x ///////////////////////////////////////////
                {
                    size_t mpitch= metric_image.pitch;
                    size_t mslicePitch= mpitch*d_sz[1]*k;
                    size_t mcolPitch= j*mpitch;
                    char *m_ptr= (char *)(metric_image.ptr);
                    char * slice_m= m_ptr+  mslicePitch;
                    float * row_M= (float *)(slice_m+ mcolPitch);


                    size_t sKSpitch= sKS.pitch;
                    size_t sKSslicePitch= sKSpitch*d_sz[1]*k;
                    size_t sKScolPitch= j*sKSpitch;
                    char *sKS_ptr= (char *)(sKS.ptr);
                    char * slice_sKS= sKS_ptr+  sKSslicePitch;
                    float * row_sKS= (float *)(slice_sKS+ sKScolPitch);

                    size_t sSSpitch= sSS.pitch;
                    size_t sSSslicePitch= sSSpitch*d_sz[1]*k;
                    size_t sSScolPitch= j*sSSpitch;
                    char *sSS_ptr= (char *)(sSS.ptr);
                    char * slice_sSS= sSS_ptr+  sSSslicePitch;
                    float * row_sSS= (float *)(slice_sSS+ sSScolPitch);

                    size_t sKKpitch= sKK.pitch;
                    size_t sKKslicePitch= sKKpitch*d_sz[1]*k;
                    size_t sKKcolPitch= j*sKKpitch;
                    char *sKK_ptr= (char *)(sKK.ptr);
                    char * slice_sKK= sKK_ptr+  sKKslicePitch;
                    float * row_sKK= (float *)(slice_sKK+ sKKcolPitch);

                    size_t valSpitch= valS.pitch;
                    size_t valSslicePitch= valSpitch*d_sz[1]*k;
                    size_t valScolPitch= j*valSpitch;
                    char *valS_ptr= (char *)(valS.ptr);
                    char * slice_valS= valS_ptr+  valSslicePitch;
                    float * row_valS= (float *)(slice_valS+ valScolPitch);

                    size_t valKpitch= valK.pitch;
                    size_t valKslicePitch= valKpitch*d_sz[1]*k;
                    size_t valKcolPitch= j*valKpitch;
                    char *valK_ptr= (char *)(valK.ptr);
                    char * slice_valK= valK_ptr+  valKslicePitch;
                    float * row_valK= (float *)(slice_valK+ valKcolPitch);

                    size_t fpitch= field.pitch;
                    size_t fslicePitch= fpitch*d_sz[1]*k;
                    size_t fcolPitch= j*fpitch;
                    char *f_ptr= (char *)(field.ptr);
                    char * slice_f= f_ptr+  fslicePitch;
                    float * row_f= (float *)(slice_f+ fcolPitch);



                    float sSS_val=row_sSS[i];
                    float sKS_val=row_sKS[i];
                    float sKK_val=row_sKK[i];
                    float valS_val=row_valS[i];
                    float valK_val=row_valK[i];

                    float sSS_sKK = sSS_val * sKK_val;

                    if(fabs(sSS_sKK) > LIMCCJAC && fabs(sKK_val) > LIMCCJAC )
                    {
                        row_M[i]+= -sKS_val*sKS_val/ sSS_sKK;

                        float x= (d_dir[0]*i  + d_dir[1]*j + d_dir[2]*k)* d_spc[0] ;
                        float y= (d_dir[3]*i  + d_dir[4]*j + d_dir[5]*k)* d_spc[1] ;
                        float z= (d_dir[6]*i  + d_dir[7]*j + d_dir[8]*k)* d_spc[2] ;

                        float xw= x + row_f[3*i];
                        float yw= y + row_f[3*i+1];
                        float zw= z + row_f[3*i+2];

                        float iw = (d_dir[0]*xw  + d_dir[3]*yw  + d_dir[6]*zw)/ d_spc[0] ;
                        float jw = (d_dir[1]*xw  + d_dir[4]*yw  + d_dir[7]*zw)/ d_spc[1] ;
                        float kw = (d_dir[2]*xw  + d_dir[5]*yw  + d_dir[8]*zw)/ d_spc[2] ;

                        float grad_x =tex3D<float>(grad_img_x, iw+0.5, jw +0.5, kw+0.5);
                        float grad_y =tex3D<float>(grad_img_y, iw+0.5, jw +0.5, kw+0.5);
                        float grad_z =tex3D<float>(grad_img_z, iw+0.5, jw +0.5, kw+0.5);


                        float first_term= -2*sKS_val/ sSS_sKK;

                        float det = ComputeSingleJacobianMatrixAtIndex(field,i,j,k,1,phase,phase_xyz);
                        if(det <=-1)
                           det=-1+1E-5;


                        float M1[3]= {grad_x *(1+det),grad_y *(1+det),grad_z*(1+det)};
                        float M2= 0;
                        M1[phase_xyz]+=M2;

                       float second_term= (valS_val - sKS_val/ sKK_val *valK_val);
                       update[0] = first_term * second_term *M1[0] ;
                       update[1] = first_term * second_term *M1[1] ;
                       update[2] = first_term * second_term *M1[2] ;

                   }
                }  // at x


                ////////////////////////   at x+1 ///////////////////////////////////////////

                {
                    int h=1;
                    int nindex[3]={(int)i,(int)j,(int)k};
                    nindex[phase]+=h;


                    if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
                    {
                        size_t sKSpitch= sKS.pitch;
                        size_t sKSslicePitch= sKSpitch*d_sz[1]*nindex[2];
                        size_t sKScolPitch= nindex[1]*sKSpitch;
                        char *sKS_ptr= (char *)(sKS.ptr);
                        char * slice_sKS= sKS_ptr+  sKSslicePitch;
                        float * row_sKS= (float *)(slice_sKS+ sKScolPitch);

                        size_t sSSpitch= sSS.pitch;
                        size_t sSSslicePitch= sSSpitch*d_sz[1]*nindex[2];
                        size_t sSScolPitch= nindex[1]*sSSpitch;
                        char *sSS_ptr= (char *)(sSS.ptr);
                        char * slice_sSS= sSS_ptr+  sSSslicePitch;
                        float * row_sSS= (float *)(slice_sSS+ sSScolPitch);

                        size_t sKKpitch= sKK.pitch;
                        size_t sKKslicePitch= sKKpitch*d_sz[1]*nindex[2];
                        size_t sKKcolPitch= nindex[1]*sKKpitch;
                        char *sKK_ptr= (char *)(sKK.ptr);
                        char * slice_sKK= sKK_ptr+  sKKslicePitch;
                        float * row_sKK= (float *)(slice_sKK+ sKKcolPitch);

                        size_t valSpitch= valS.pitch;
                        size_t valSslicePitch= valSpitch*d_sz[1]*nindex[2];
                        size_t valScolPitch= nindex[1]*valSpitch;
                        char *valS_ptr= (char *)(valS.ptr);
                        char * slice_valS= valS_ptr+  valSslicePitch;
                        float * row_valS= (float *)(slice_valS+ valScolPitch);

                        size_t valKpitch= valK.pitch;
                        size_t valKslicePitch= valKpitch*d_sz[1]*nindex[2];
                        size_t valKcolPitch= nindex[1]*valKpitch;
                        char *valK_ptr= (char *)(valK.ptr);
                        char * slice_valK= valK_ptr+  valKslicePitch;
                        float * row_valK= (float *)(slice_valK+ valKcolPitch);


                        size_t bimgpitch= b0_img.pitch;
                        size_t bimgslicePitch= bimgpitch*d_sz[1]*nindex[2];
                        size_t bimgcolPitch= nindex[1]*bimgpitch;
                        char *bimg_ptr= (char *)(b0_img.ptr);
                        char * slice_bimg= bimg_ptr+  bimgslicePitch;
                        float * row_bimg= (float *)(slice_bimg+ bimgcolPitch);

                        size_t fpitch= field.pitch;
                        size_t fslicePitch= fpitch*d_sz[1]*nindex[2];
                        size_t fcolPitch= nindex[1]*fpitch;
                        char *f_ptr= (char *)(field.ptr);
                        char * slice_f= f_ptr+  fslicePitch;
                        float * row_f= (float *)(slice_f+ fcolPitch);


                        float valb_center=row_bimg[nindex[0]];
                        float sSS_val=row_sSS[nindex[0]];
                        float sKS_val=row_sKS[nindex[0]];
                        float sKK_val=row_sKK[nindex[0]];
                        float valS_val=row_valS[nindex[0]];
                        float valK_val=row_valK[nindex[0]];

                        float sSS_sKK = sSS_val * sKK_val;


                        if(fabs(sSS_sKK) > LIMCCJAC && fabs(sKK_val) > LIMCCJAC )
                        {
                          float first_term= -2*sKS_val/ sSS_sKK;

                          float det = ComputeSingleJacobianMatrixAtIndex(field,nindex[0],nindex[1],nindex[2],1,phase,phase_xyz);
                          if(det <=-1)
                              det=-1+1E-5;


                          float x= (d_dir[0]*nindex[0]  + d_dir[1]*nindex[1] + d_dir[2]*nindex[2])* d_spc[0] ;
                          float y= (d_dir[3]*nindex[0]  + d_dir[4]*nindex[1] + d_dir[5]*nindex[2])* d_spc[1] ;
                          float z= (d_dir[6]*nindex[0]  + d_dir[7]*nindex[1] + d_dir[8]*nindex[2])* d_spc[2] ;

                          float xw= x + row_f[3*nindex[0]];
                          float yw= y + row_f[3*nindex[0]+1];
                          float zw= z + row_f[3*nindex[0]+2];

                          float iw = (d_dir[0]*xw  + d_dir[3]*yw  + d_dir[6]*zw)/ d_spc[0] ;
                          float jw = (d_dir[1]*xw  + d_dir[4]*yw  + d_dir[7]*zw)/ d_spc[1] ;
                          float kw = (d_dir[2]*xw  + d_dir[5]*yw  + d_dir[8]*zw)/ d_spc[2] ;

                          float grad_x =tex3D<float>(grad_img_x, iw+0.5, jw +0.5, kw+0.5);
                          float grad_y =tex3D<float>(grad_img_y, iw+0.5, jw +0.5, kw+0.5);
                          float grad_z =tex3D<float>(grad_img_z, iw+0.5, jw +0.5, kw+0.5);



                          float M1[3]= {grad_x *(1+det)*a_b,grad_y *(1+det)*a_b,grad_z*(1+det)*a_b};
                          float M2= d_new_phase[phase_xyz]* valb_center* -0.5/d_spc[phase]/h ;;
                          M1[phase_xyz]+=M2;

                          float second_term= (valS_val - sKS_val/ sKK_val *valK_val);
                          update[0] += first_term * second_term *M1[0] ;
                          update[1] += first_term * second_term *M1[1] ;
                          update[2] += first_term * second_term *M1[2] ;
                      }
                    }
                } //x+1


                ////////////////////////   at x-1 ///////////////////////////////////////////

                {
                    int h=1;
                    int nindex[3]={(int)i,(int)j,(int)k};
                    nindex[phase]-=h;


                    if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
                    {
                        size_t sKSpitch= sKS.pitch;
                        size_t sKSslicePitch= sKSpitch*d_sz[1]*nindex[2];
                        size_t sKScolPitch= nindex[1]*sKSpitch;
                        char *sKS_ptr= (char *)(sKS.ptr);
                        char * slice_sKS= sKS_ptr+  sKSslicePitch;
                        float * row_sKS= (float *)(slice_sKS+ sKScolPitch);

                        size_t sSSpitch= sSS.pitch;
                        size_t sSSslicePitch= sSSpitch*d_sz[1]*nindex[2];
                        size_t sSScolPitch= nindex[1]*sSSpitch;
                        char *sSS_ptr= (char *)(sSS.ptr);
                        char * slice_sSS= sSS_ptr+  sSSslicePitch;
                        float * row_sSS= (float *)(slice_sSS+ sSScolPitch);

                        size_t sKKpitch= sKK.pitch;
                        size_t sKKslicePitch= sKKpitch*d_sz[1]*nindex[2];
                        size_t sKKcolPitch= nindex[1]*sKKpitch;
                        char *sKK_ptr= (char *)(sKK.ptr);
                        char * slice_sKK= sKK_ptr+  sKKslicePitch;
                        float * row_sKK= (float *)(slice_sKK+ sKKcolPitch);

                        size_t valSpitch= valS.pitch;
                        size_t valSslicePitch= valSpitch*d_sz[1]*nindex[2];
                        size_t valScolPitch= nindex[1]*valSpitch;
                        char *valS_ptr= (char *)(valS.ptr);
                        char * slice_valS= valS_ptr+  valSslicePitch;
                        float * row_valS= (float *)(slice_valS+ valScolPitch);

                        size_t valKpitch= valK.pitch;
                        size_t valKslicePitch= valKpitch*d_sz[1]*nindex[2];
                        size_t valKcolPitch= nindex[1]*valKpitch;
                        char *valK_ptr= (char *)(valK.ptr);
                        char * slice_valK= valK_ptr+  valKslicePitch;
                        float * row_valK= (float *)(slice_valK+ valKcolPitch);

                        size_t bimgpitch= b0_img.pitch;
                        size_t bimgslicePitch= bimgpitch*d_sz[1]*nindex[2];
                        size_t bimgcolPitch= nindex[1]*bimgpitch;
                        char *bimg_ptr= (char *)(b0_img.ptr);
                        char * slice_bimg= bimg_ptr+  bimgslicePitch;
                        float * row_bimg= (float *)(slice_bimg+ bimgcolPitch);

                        size_t fpitch= field.pitch;
                        size_t fslicePitch= fpitch*d_sz[1]*nindex[2];
                        size_t fcolPitch= nindex[1]*fpitch;
                        char *f_ptr= (char *)(field.ptr);
                        char * slice_f= f_ptr+  fslicePitch;
                        float * row_f= (float *)(slice_f+ fcolPitch);


                        float valb_center=row_bimg[nindex[0]];
                        float sSS_val=row_sSS[nindex[0]];
                        float sKS_val=row_sKS[nindex[0]];
                        float sKK_val=row_sKK[nindex[0]];
                        float valS_val=row_valS[nindex[0]];
                        float valK_val=row_valK[nindex[0]];

                       float sSS_sKK = sSS_val * sKK_val;

                       if(fabs(sSS_sKK) > LIMCCJAC && fabs(sKK_val) > LIMCCJAC )
                       {
                         float first_term= -2*sKS_val/ sSS_sKK;

                         float det = ComputeSingleJacobianMatrixAtIndex(field,nindex[0],nindex[1],nindex[2],1,phase,phase_xyz);
                         if(det <=-1)
                             det=-1+1E-5;


                         float x= (d_dir[0]*nindex[0]  + d_dir[1]*nindex[1] + d_dir[2]*nindex[2])* d_spc[0] ;
                         float y= (d_dir[3]*nindex[0]  + d_dir[4]*nindex[1] + d_dir[5]*nindex[2])* d_spc[1] ;
                         float z= (d_dir[6]*nindex[0]  + d_dir[7]*nindex[1] + d_dir[8]*nindex[2])* d_spc[2] ;

                         float xw= x + row_f[3*nindex[0]];
                         float yw= y + row_f[3*nindex[0]+1];
                         float zw= z + row_f[3*nindex[0]+2];

                         float iw = (d_dir[0]*xw  + d_dir[3]*yw  + d_dir[6]*zw)/ d_spc[0] ;
                         float jw = (d_dir[1]*xw  + d_dir[4]*yw  + d_dir[7]*zw)/ d_spc[1] ;
                         float kw = (d_dir[2]*xw  + d_dir[5]*yw  + d_dir[8]*zw)/ d_spc[2] ;

                         float grad_x =tex3D<float>(grad_img_x, iw+0.5, jw +0.5, kw+0.5);
                         float grad_y =tex3D<float>(grad_img_y, iw+0.5, jw +0.5, kw+0.5);
                         float grad_z =tex3D<float>(grad_img_z, iw+0.5, jw +0.5, kw+0.5);



                         float M1[3]= {grad_x *(1+det)*a_b,grad_y *(1+det)*a_b,grad_z*(1+det)*a_b};
                         float M2= d_new_phase[phase_xyz]* valb_center* 0.5/d_spc[phase]/h ;;
                         M1[phase_xyz]+=M2;

                         float second_term= (valS_val - sKS_val/ sKK_val *valK_val);
                         update[0] += first_term * second_term *M1[0] ;
                         update[1] += first_term * second_term *M1[1] ;
                         update[2] += first_term * second_term *M1[2] ;
                     }

                    }
                } //x-1


                size_t ufpitch= updateFieldINV.pitch;
                size_t ufslicePitch= ufpitch*d_sz[1]*k;
                size_t ufcolPitch= j*ufpitch;
                char *uf_ptr= (char *)(updateFieldINV.ptr);
                char * slice_uf= uf_ptr+  ufslicePitch;
                float * row_uf= (float *)(slice_uf+ ufcolPitch);

                row_uf[3*i]= -update[0];
                row_uf[3*i+1]=- update[1];
                row_uf[3*i+2]= -update[2];
            }
        }
     }
}


__global__ void
NegateImage2_kernel(cudaPitchedPtr image,  const int3 d_sz, const int Ncomponents)
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
AddToUpdateField2_kernel(cudaPitchedPtr total_data, cudaPitchedPtr to_add_data , float weight, int3 d_sz, int Ncomponents)
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



void ComputeMetric_CCJacSSingle_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img, cudaPitchedPtr str_img,
                                    cudaTextureObject_t up_grad_img_x, cudaTextureObject_t up_grad_img_y, cudaTextureObject_t up_grad_img_z,
                                    cudaTextureObject_t down_grad_img_x, cudaTextureObject_t down_grad_img_y, cudaTextureObject_t down_grad_img_z,
                                    int3 data_sz, float3 data_spc,
                                    float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                                    cudaPitchedPtr def_FINV,
                                    cudaPitchedPtr updateFieldFINV,
                                    float3 phase_vector,int kernel_sz, float* h_kernel, float &metric_value)
{

    float h_d_dir[]= {d00,d01,d02,d10,d11,d12,d20,d21,d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));
    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));

    gpuErrchk(cudaMemcpyToSymbol(c_Kernel, h_kernel, kernel_sz * sizeof(float)));


    cudaPitchedPtr metric_image={0};
    cudaExtent extent =  make_cudaExtent(up_img.pitch,data_sz.y,data_sz.z);
    cudaMalloc3D(&metric_image, extent);
    cudaMemset3D(metric_image,0,extent);


    float new_phase[3];
    new_phase[0]= d00*phase_vector.x + d01*phase_vector.y +d02*phase_vector.z ;
    new_phase[1]= d10*phase_vector.x + d11*phase_vector.y +d12*phase_vector.z ;
    new_phase[2]= d20*phase_vector.x + d21*phase_vector.y +d22*phase_vector.z ;
    gpuErrchk(cudaMemcpyToSymbol(d_new_phase, &new_phase, 3 * sizeof(float)));

    int phase_xyz,phase;
    if( (fabs(phase_vector.x) > fabs(phase_vector.y))  && (fabs(phase_vector.x) > fabs(phase_vector.z)))
        phase=0;
    else if( (fabs(phase_vector.y) > fabs(phase_vector.x))  && (fabs(phase_vector.y) > fabs(phase_vector.z)))
        phase=1;
    else phase=2;

    if( (fabs(new_phase[0]) > fabs(new_phase[1]))  && (fabs(new_phase[0]) > fabs(new_phase[2])))
        phase_xyz=0;
    else if( (fabs(new_phase[1]) > fabs(new_phase[0]))  && (fabs(new_phase[1]) > fabs(new_phase[2])))
        phase_xyz=1;
    else phase_xyz=2;



    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x/PER_GROUP), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z) );

    cudaPitchedPtr sSS={0};
    cudaMalloc3D(&sSS, extent);    cudaMemset3D(sSS,0,extent);
    cudaPitchedPtr valS={0};
    cudaMalloc3D(&valS, extent);    cudaMemset3D(valS,0,extent);


    {
        cudaPitchedPtr sKS={0};
        cudaMalloc3D(&sKS, extent);    cudaMemset3D(sKS,0,extent);
        cudaPitchedPtr sKK={0};
        cudaMalloc3D(&sKK, extent);    cudaMemset3D(sKK,0,extent);
        cudaPitchedPtr valK={0};
        cudaMalloc3D(&valK, extent);    cudaMemset3D(valK,0,extent);

        cudaPitchedPtr detimg={0};
        cudaMalloc3D(&detimg, extent);    cudaMemset3D(detimg,0,extent);
        computeDetImg<<< blockSize,gridSize>>>( up_img,def_FINV,detimg,phase,phase_xyz);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        computeFiniteDiffStructs<<< blockSize,gridSize>>>( detimg, str_img, sKS,sSS,sKK,valS,valK);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        ComputeMetric_CCJacSSingle_kernel<<< blockSize,gridSize>>>( up_img, str_img, up_grad_img_x,up_grad_img_y,up_grad_img_z,def_FINV, updateFieldFINV, metric_image, phase, phase_xyz, kernel_sz ,sKS,sSS,sKK,valS,valK);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        cudaFree(valK.ptr);
        cudaFree(sKK.ptr);
        cudaFree(sKS.ptr);
        cudaFree(detimg.ptr);
    }

    {
        cudaPitchedPtr sKS={0};
        cudaMalloc3D(&sKS, extent);    cudaMemset3D(sKS,0,extent);
        cudaPitchedPtr sKK={0};
        cudaMalloc3D(&sKK, extent);    cudaMemset3D(sKK,0,extent);
        cudaPitchedPtr valK={0};
        cudaMalloc3D(&valK, extent);    cudaMemset3D(valK,0,extent);
        cudaMemset3D(valS,0,extent);
        cudaMemset3D(sSS,0,extent);


        cudaPitchedPtr def_MINV={0};
        cudaExtent extentF =  make_cudaExtent(3*sizeof(float)*data_sz.x,data_sz.y,data_sz.z);
        cudaMalloc3D(&def_MINV, extentF);

        cudaPitchedPtr updateFieldMINV={0};
        cudaMalloc3D(&updateFieldMINV, extentF);
        cudaMemset3D(updateFieldMINV,0,extentF);


        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = def_FINV;
        copyParams.dstPtr =   def_MINV;
        copyParams.extent   = extentF;
        copyParams.kind     = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParams);


        NegateImage2_kernel<<< blockSize,gridSize>>>(def_MINV,data_sz,3);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        cudaPitchedPtr detimg={0};
        cudaMalloc3D(&detimg, extent);    cudaMemset3D(detimg,0,extent);
        computeDetImg<<< blockSize,gridSize>>>( down_img,def_MINV,detimg,phase,phase_xyz);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        computeFiniteDiffStructs<<< blockSize,gridSize>>>( detimg, str_img,  sKS,sSS,sKK,valS,valK);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        ComputeMetric_CCJacSSingle_kernel<<< blockSize,gridSize>>>( down_img, str_img, down_grad_img_x,down_grad_img_y,down_grad_img_z, def_MINV, updateFieldMINV, metric_image, phase, phase_xyz, kernel_sz ,sKS,sSS,sKK,valS,valK);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        AddToUpdateField2_kernel<<< blockSize,gridSize>>>(  updateFieldFINV,updateFieldMINV,-1,data_sz,3);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        cudaFree(updateFieldMINV.ptr);
        cudaFree(def_MINV.ptr);
        cudaFree(valK.ptr);
        cudaFree(sKK.ptr);
        cudaFree(sKS.ptr);
        cudaFree(detimg.ptr);
    }

    cudaFree(sSS.ptr);
    cudaFree(valS.ptr);



    float* dev_out;
    float out;
    cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

    ScalarFindSum<<<gSize, bSize>>>((float *)metric_image.ptr, metric_image.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
    cudaDeviceSynchronize();
    ScalarFindSum<<<1, bSize>>>(dev_out, gSize, dev_out);
    cudaDeviceSynchronize();

    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_out);
    metric_value=out/data_sz.x/data_sz.y/data_sz.z;

    cudaFree(metric_image.ptr);

}



__global__ void
ComputeMetric_CCJacS_kernel( cudaPitchedPtr b0_img,  cudaPitchedPtr str_img,
                            cudaPitchedPtr field,
                            cudaPitchedPtr updateField,
                            cudaPitchedPtr metric_image,
                            int phase, int phase_xyz,
                            int kernel_sz,
                             cudaPitchedPtr sKS, cudaPitchedPtr sSS, cudaPitchedPtr sKK, cudaPitchedPtr valS, cudaPitchedPtr valK)
{
    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            float update[3]={0,0,0};

            if(i>=1 && j>=1 && k>=1 && i<=d_sz[0]-2  && j<=d_sz[1]-2  && k<=d_sz[2]-2)
            {
                int mid_ind=(kernel_sz-1)/2;
                float b= c_Kernel[  mid_ind ];
                float a=0;
                if((kernel_sz-1)/2 > 0)
                    a=c_Kernel[  mid_ind -1];

                float a_b= a/b;



                ////////////////////////   at x ///////////////////////////////////////////
                {
                    size_t mpitch= metric_image.pitch;
                    size_t mslicePitch= mpitch*d_sz[1]*k;
                    size_t mcolPitch= j*mpitch;
                    char *m_ptr= (char *)(metric_image.ptr);
                    char * slice_m= m_ptr+  mslicePitch;
                    float * row_M= (float *)(slice_m+ mcolPitch);


                    size_t sKSpitch= sKS.pitch;
                    size_t sKSslicePitch= sKSpitch*d_sz[1]*k;
                    size_t sKScolPitch= j*sKSpitch;
                    char *sKS_ptr= (char *)(sKS.ptr);
                    char * slice_sKS= sKS_ptr+  sKSslicePitch;
                    float * row_sKS= (float *)(slice_sKS+ sKScolPitch);

                    size_t sSSpitch= sSS.pitch;
                    size_t sSSslicePitch= sSSpitch*d_sz[1]*k;
                    size_t sSScolPitch= j*sSSpitch;
                    char *sSS_ptr= (char *)(sSS.ptr);
                    char * slice_sSS= sSS_ptr+  sSSslicePitch;
                    float * row_sSS= (float *)(slice_sSS+ sSScolPitch);

                    size_t sKKpitch= sKK.pitch;
                    size_t sKKslicePitch= sKKpitch*d_sz[1]*k;
                    size_t sKKcolPitch= j*sKKpitch;
                    char *sKK_ptr= (char *)(sKK.ptr);
                    char * slice_sKK= sKK_ptr+  sKKslicePitch;
                    float * row_sKK= (float *)(slice_sKK+ sKKcolPitch);

                    size_t valSpitch= valS.pitch;
                    size_t valSslicePitch= valSpitch*d_sz[1]*k;
                    size_t valScolPitch= j*valSpitch;
                    char *valS_ptr= (char *)(valS.ptr);
                    char * slice_valS= valS_ptr+  valSslicePitch;
                    float * row_valS= (float *)(slice_valS+ valScolPitch);

                    size_t valKpitch= valK.pitch;
                    size_t valKslicePitch= valKpitch*d_sz[1]*k;
                    size_t valKcolPitch= j*valKpitch;
                    char *valK_ptr= (char *)(valK.ptr);
                    char * slice_valK= valK_ptr+  valKslicePitch;
                    float * row_valK= (float *)(slice_valK+ valKcolPitch);


                    float sSS_val=row_sSS[i];
                    float sKS_val=row_sKS[i];
                    float sKK_val=row_sKK[i];
                    float valS_val=row_valS[i];
                    float valK_val=row_valK[i];

                    float sSS_sKK = sSS_val * sKK_val;

                    if(fabs(sSS_sKK) > LIMCCJAC && fabs(sKK_val) > LIMCCJAC )
                    {
                       row_M[i]+= -sKS_val*sKS_val/ sSS_sKK;


                       float first_term= -2*sKS_val/ sSS_sKK;

                       float detF2 = ComputeSingleJacobianMatrixAtIndex(field,i,j,k,1,phase,phase_xyz)+1;
                       if(detF2 <=0)
                           detF2=1E-5;
                       float detF= mf(detF2);


                       float3 M1t = ComputeImageGradient(b0_img,i,j,k);
                       float M1[3]= {M1t.x *detF,M1t.y *detF,M1t.z*detF};
                       float M2= 0;
                       M1[phase_xyz]+=M2;

                       float second_term= (valS_val - sKS_val/ sKK_val *valK_val);
                       update[0] = first_term * second_term *M1[0] ;
                       update[1] = first_term * second_term *M1[1] ;
                       update[2] = first_term * second_term *M1[2] ;
                       
                   }
                }  // at x


                ////////////////////////   at x+1 ///////////////////////////////////////////

                {
                    int h=1;
                    int nindex[3]={(int)i,(int)j,(int)k};
                    nindex[phase]+=h;


                    if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
                    {
                        size_t sKSpitch= sKS.pitch;
                        size_t sKSslicePitch= sKSpitch*d_sz[1]*nindex[2];
                        size_t sKScolPitch= nindex[1]*sKSpitch;
                        char *sKS_ptr= (char *)(sKS.ptr);
                        char * slice_sKS= sKS_ptr+  sKSslicePitch;
                        float * row_sKS= (float *)(slice_sKS+ sKScolPitch);

                        size_t sSSpitch= sSS.pitch;
                        size_t sSSslicePitch= sSSpitch*d_sz[1]*nindex[2];
                        size_t sSScolPitch= nindex[1]*sSSpitch;
                        char *sSS_ptr= (char *)(sSS.ptr);
                        char * slice_sSS= sSS_ptr+  sSSslicePitch;
                        float * row_sSS= (float *)(slice_sSS+ sSScolPitch);

                        size_t sKKpitch= sKK.pitch;
                        size_t sKKslicePitch= sKKpitch*d_sz[1]*nindex[2];
                        size_t sKKcolPitch= nindex[1]*sKKpitch;
                        char *sKK_ptr= (char *)(sKK.ptr);
                        char * slice_sKK= sKK_ptr+  sKKslicePitch;
                        float * row_sKK= (float *)(slice_sKK+ sKKcolPitch);

                        size_t valSpitch= valS.pitch;
                        size_t valSslicePitch= valSpitch*d_sz[1]*nindex[2];
                        size_t valScolPitch= nindex[1]*valSpitch;
                        char *valS_ptr= (char *)(valS.ptr);
                        char * slice_valS= valS_ptr+  valSslicePitch;
                        float * row_valS= (float *)(slice_valS+ valScolPitch);

                        size_t valKpitch= valK.pitch;
                        size_t valKslicePitch= valKpitch*d_sz[1]*nindex[2];
                        size_t valKcolPitch= nindex[1]*valKpitch;
                        char *valK_ptr= (char *)(valK.ptr);
                        char * slice_valK= valK_ptr+  valKslicePitch;
                        float * row_valK= (float *)(slice_valK+ valKcolPitch);


                        size_t bimgpitch= b0_img.pitch;
                        size_t bimgslicePitch= bimgpitch*d_sz[1]*nindex[2];
                        size_t bimgcolPitch= nindex[1]*bimgpitch;
                        char *bimg_ptr= (char *)(b0_img.ptr);
                        char * slice_bimg= bimg_ptr+  bimgslicePitch;
                        float * row_bimg= (float *)(slice_bimg+ bimgcolPitch);


                        float valb_center=row_bimg[nindex[0]];
                        float sSS_val=row_sSS[nindex[0]];
                        float sKS_val=row_sKS[nindex[0]];
                        float sKK_val=row_sKK[nindex[0]];
                        float valS_val=row_valS[nindex[0]];
                        float valK_val=row_valK[nindex[0]];

                       float sSS_sKK = sSS_val * sKK_val;


                       if(fabs(sSS_sKK) > LIMCCJAC && fabs(sKK_val) > LIMCCJAC )
                       {
                          float first_term= -2*sKS_val/ sSS_sKK;

                          float detF2 = ComputeSingleJacobianMatrixAtIndex(field,nindex[0],nindex[1],nindex[2],1,phase,phase_xyz)+1;
                          if(detF2 <=0)
                              detF2=1E-5;
                          float detF= mf(detF2);


                          float3 M1t = ComputeImageGradient(b0_img,nindex[0],nindex[1],nindex[2]);
                          float M1[3]= {M1t.x ,M1t.y ,M1t.z};
                          M1[0]*= detF*a_b;
                          M1[1]*= detF*a_b;
                          M1[2]*= detF*a_b;
                          float M2= d_new_phase[phase_xyz]*dmf(detF2)* valb_center* -0.5/d_spc[phase]/h ;;
                          M1[phase_xyz]+=M2;

                          float second_term= (valS_val - sKS_val/ sKK_val *valK_val);
                          //update[phase_xyz] += first_term * second_term *M1[phase_xyz] ;
                          update[0] += first_term * second_term *M1[0] ;
                          update[1] += first_term * second_term *M1[1] ;
                          update[2] += first_term * second_term *M1[2] ;
                      }
                    }
                } //x+1


                ////////////////////////   at x-1 ///////////////////////////////////////////

                {
                    int h=1;
                    int nindex[3]={(int)i,(int)j,(int)k};
                    nindex[phase]-=h;


                    if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
                    {
                        size_t sKSpitch= sKS.pitch;
                        size_t sKSslicePitch= sKSpitch*d_sz[1]*nindex[2];
                        size_t sKScolPitch= nindex[1]*sKSpitch;
                        char *sKS_ptr= (char *)(sKS.ptr);
                        char * slice_sKS= sKS_ptr+  sKSslicePitch;
                        float * row_sKS= (float *)(slice_sKS+ sKScolPitch);

                        size_t sSSpitch= sSS.pitch;
                        size_t sSSslicePitch= sSSpitch*d_sz[1]*nindex[2];
                        size_t sSScolPitch= nindex[1]*sSSpitch;
                        char *sSS_ptr= (char *)(sSS.ptr);
                        char * slice_sSS= sSS_ptr+  sSSslicePitch;
                        float * row_sSS= (float *)(slice_sSS+ sSScolPitch);

                        size_t sKKpitch= sKK.pitch;
                        size_t sKKslicePitch= sKKpitch*d_sz[1]*nindex[2];
                        size_t sKKcolPitch= nindex[1]*sKKpitch;
                        char *sKK_ptr= (char *)(sKK.ptr);
                        char * slice_sKK= sKK_ptr+  sKKslicePitch;
                        float * row_sKK= (float *)(slice_sKK+ sKKcolPitch);

                        size_t valSpitch= valS.pitch;
                        size_t valSslicePitch= valSpitch*d_sz[1]*nindex[2];
                        size_t valScolPitch= nindex[1]*valSpitch;
                        char *valS_ptr= (char *)(valS.ptr);
                        char * slice_valS= valS_ptr+  valSslicePitch;
                        float * row_valS= (float *)(slice_valS+ valScolPitch);

                        size_t valKpitch= valK.pitch;
                        size_t valKslicePitch= valKpitch*d_sz[1]*nindex[2];
                        size_t valKcolPitch= nindex[1]*valKpitch;
                        char *valK_ptr= (char *)(valK.ptr);
                        char * slice_valK= valK_ptr+  valKslicePitch;
                        float * row_valK= (float *)(slice_valK+ valKcolPitch);

                        size_t bimgpitch= b0_img.pitch;
                        size_t bimgslicePitch= bimgpitch*d_sz[1]*nindex[2];
                        size_t bimgcolPitch= nindex[1]*bimgpitch;
                        char *bimg_ptr= (char *)(b0_img.ptr);
                        char * slice_bimg= bimg_ptr+  bimgslicePitch;
                        float * row_bimg= (float *)(slice_bimg+ bimgcolPitch);


                        float valb_center=row_bimg[nindex[0]];
                        float sSS_val=row_sSS[nindex[0]];
                        float sKS_val=row_sKS[nindex[0]];
                        float sKK_val=row_sKK[nindex[0]];
                        float valS_val=row_valS[nindex[0]];
                        float valK_val=row_valK[nindex[0]];

                       float sSS_sKK = sSS_val * sKK_val;

                       if(fabs(sSS_sKK) > LIMCCJAC && fabs(sKK_val) > LIMCCJAC )
                       {
                          float first_term= -2*sKS_val/ sSS_sKK;

                          float detF2 = ComputeSingleJacobianMatrixAtIndex(field,nindex[0],nindex[1],nindex[2],1,phase,phase_xyz)+1;
                          if(detF2 <=0)
                              detF2=1E-5;
                          float detF= mf(detF2);


                          float3 M1t = ComputeImageGradient(b0_img,nindex[0],nindex[1],nindex[2]);
                          float M1[3]= {M1t.x ,M1t.y ,M1t.z};
                          M1[0]*= detF*a_b;
                          M1[1]*= detF*a_b;
                          M1[2]*= detF*a_b;
                          float M2= d_new_phase[phase_xyz]*dmf(detF2)* valb_center* 0.5/d_spc[phase]/h ;;
                          M1[phase_xyz]+=M2;

                          float second_term= (valS_val - sKS_val/ sKK_val *valK_val);
                      //    update[phase_xyz] += first_term * second_term *M1[phase_xyz] ;
                          update[0] += first_term * second_term *M1[0] ;
                          update[1] += first_term * second_term *M1[1] ;
                          update[2] += first_term * second_term *M1[2] ;
                      }

                    }
                } //x-1


                size_t ufpitch= updateField.pitch;
                size_t ufslicePitch= ufpitch*d_sz[1]*k;
                size_t ufcolPitch= j*ufpitch;
                char *uf_ptr= (char *)(updateField.ptr);
                char * slice_uf= uf_ptr+  ufslicePitch;
                float * row_uf= (float *)(slice_uf+ ufcolPitch);

                row_uf[3*i]= update[0];
                row_uf[3*i+1]= update[1];
                row_uf[3*i+2]= update[2];
            }
        }
     }
}




void ComputeMetric_CCJacS_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img, cudaPitchedPtr str_img,
     int3 data_sz, float3 data_spc,
     float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
     cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV,
     cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
     const float3 phase_vector,
     int kernel_sz, float *h_kernel, float &metric_value		   )
{
    float h_d_dir[]= {d00,d01,d02,d10,d11,d12,d20,d21,d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));
    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));

    gpuErrchk(cudaMemcpyToSymbol(c_Kernel, h_kernel, kernel_sz * sizeof(float)));


    cudaPitchedPtr metric_image={0};
    cudaExtent extent =  make_cudaExtent(up_img.pitch,data_sz.y,data_sz.z);
    cudaMalloc3D(&metric_image, extent);
    cudaMemset3D(metric_image,0,extent);


    float new_phase[3];
    new_phase[0]= d00*phase_vector.x + d01*phase_vector.y +d02*phase_vector.z ;
    new_phase[1]= d10*phase_vector.x + d11*phase_vector.y +d12*phase_vector.z ;
    new_phase[2]= d20*phase_vector.x + d21*phase_vector.y +d22*phase_vector.z ;
    gpuErrchk(cudaMemcpyToSymbol(d_new_phase, &new_phase, 3 * sizeof(float)));

    int phase_xyz,phase;
    if( (fabs(phase_vector.x) > fabs(phase_vector.y))  && (fabs(phase_vector.x) > fabs(phase_vector.z)))
        phase=0;
    else if( (fabs(phase_vector.y) > fabs(phase_vector.x))  && (fabs(phase_vector.y) > fabs(phase_vector.z)))
        phase=1;
    else phase=2;

    if( (fabs(new_phase[0]) > fabs(new_phase[1]))  && (fabs(new_phase[0]) > fabs(new_phase[2])))
        phase_xyz=0;
    else if( (fabs(new_phase[1]) > fabs(new_phase[0]))  && (fabs(new_phase[1]) > fabs(new_phase[2])))
        phase_xyz=1;
    else phase_xyz=2;


    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x/PER_GROUP), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z) );

    cudaPitchedPtr sSS={0};
    cudaMalloc3D(&sSS, extent);    cudaMemset3D(sSS,0,extent);
    cudaPitchedPtr valS={0};
    cudaMalloc3D(&valS, extent);    cudaMemset3D(valS,0,extent);


    {
        cudaPitchedPtr sKS={0};
        cudaMalloc3D(&sKS, extent);    cudaMemset3D(sKS,0,extent);
        cudaPitchedPtr sKK={0};
        cudaMalloc3D(&sKK, extent);    cudaMemset3D(sKK,0,extent);
        cudaPitchedPtr valK={0};
        cudaMalloc3D(&valK, extent);    cudaMemset3D(valK,0,extent);

        cudaPitchedPtr detimg={0};
        cudaMalloc3D(&detimg, extent);    cudaMemset3D(detimg,0,extent);
        computeDetImg<<< blockSize,gridSize>>>( up_img,def_FINV,detimg,phase,phase_xyz);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        computeFiniteDiffStructs<<< blockSize,gridSize>>>( detimg, str_img, sKS,sSS,sKK,valS,valK);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        ComputeMetric_CCJacS_kernel<<< blockSize,gridSize>>>( up_img, str_img, def_FINV, updateFieldF, metric_image, phase, phase_xyz, kernel_sz ,sKS,sSS,sKK,valS,valK);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        cudaFree(valK.ptr);
        cudaFree(sKK.ptr);
        cudaFree(sKS.ptr);
        cudaFree(detimg.ptr);
    }

    {
        cudaPitchedPtr sKS={0};
        cudaMalloc3D(&sKS, extent);    cudaMemset3D(sKS,0,extent);
        cudaPitchedPtr sKK={0};
        cudaMalloc3D(&sKK, extent);    cudaMemset3D(sKK,0,extent);
        cudaPitchedPtr valK={0};
        cudaMalloc3D(&valK, extent);    cudaMemset3D(valK,0,extent);
        cudaMemset3D(valS,0,extent);
        cudaMemset3D(sSS,0,extent);

        cudaPitchedPtr detimg={0};
        cudaMalloc3D(&detimg, extent);    cudaMemset3D(detimg,0,extent);
        computeDetImg<<< blockSize,gridSize>>>( down_img,def_MINV,detimg,phase,phase_xyz);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        computeFiniteDiffStructs<<< blockSize,gridSize>>>( detimg, str_img,  sKS,sSS,sKK,valS,valK);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        ComputeMetric_CCJacS_kernel<<< blockSize,gridSize>>>( down_img, str_img, def_MINV, updateFieldM, metric_image, phase, phase_xyz, kernel_sz ,sKS,sSS,sKK,valS,valK);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        cudaFree(valK.ptr);
        cudaFree(sKK.ptr);
        cudaFree(sKS.ptr);
        cudaFree(detimg.ptr);
    }

    cudaFree(sSS.ptr);
    cudaFree(valS.ptr);



    float* dev_out;
    float out;
    cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

    ScalarFindSum<<<gSize, bSize>>>((float *)metric_image.ptr, metric_image.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
    cudaDeviceSynchronize();
    ScalarFindSum<<<1, bSize>>>(dev_out, gSize, dev_out);
    cudaDeviceSynchronize();
    
    


    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_out);
    metric_value=out/data_sz.x/data_sz.y/data_sz.z;
    
    cudaFree(metric_image.ptr);
}











__global__ void
ComputeMetric_MSJac_kernel( cudaPitchedPtr up_img, cudaPitchedPtr down_img,                             
                            cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV, 
                            cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                            cudaPitchedPtr metric_image,
                            int phase, int phase_xyz,
                            int kernel_sz )
{
    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {

            float updateF[3]={0,0,0};
            float updateM[3]={0,0,0};

            if(i>=1 && j>=1 && k>=1 && i<=d_sz[0]-2  && j<=d_sz[1]-2  && k<=d_sz[2]-2)
            {
                float3 gradI2= ComputeImageGradient(up_img,i,j,k);
                float3 gradJ2= ComputeImageGradient(down_img,i,j,k);

                int mid_ind=(kernel_sz-1)/2;
                float b= c_Kernel[  mid_ind ];
                float a=0;
                if((kernel_sz-1)/2 > 0)
                    a=c_Kernel[  mid_ind -1];

                float a_b= a/b;
                a_b=0;


                ////////////////////////   at x ///////////////////////////////////////////
                {
                    float detf = ComputeSingleJacobianMatrixAtIndex(def_FINV,i,j,k,1,phase,phase_xyz)+1;
                    float detm = ComputeSingleJacobianMatrixAtIndex(def_MINV,i,j,k,1,phase,phase_xyz)+1;


                    if(detf <=0)
                        detf=1E-5;
                    if(detm <=0)
                        detm=1E-5;

                    detf= mf(detf);
                    detm= mf(detm);

                    size_t upitch= up_img.pitch;
                    size_t uslicePitch= upitch*d_sz[1]*k;
                    size_t ucolPitch= j*upitch;
                    char *u_ptr= (char *)(up_img.ptr);
                    char * slice_u= u_ptr+  uslicePitch;
                    float * row_up= (float *)(slice_u+ ucolPitch);

                    size_t dpitch= down_img.pitch;
                    size_t dslicePitch= dpitch*d_sz[1]*k;
                    size_t dcolPitch= j*dpitch;
                    char *d_ptr= (char *)(down_img.ptr);
                    char * slice_d= d_ptr+  dslicePitch;
                    float * row_down= (float *)(slice_d+ dcolPitch);

                    size_t mpitch= metric_image.pitch;
                    size_t mslicePitch= mpitch*d_sz[1]*k;
                    size_t mcolPitch= j*mpitch;
                    char *m_ptr= (char *)(metric_image.ptr);
                    char * slice_m= m_ptr+  mslicePitch;
                    float * row_metric= (float *)(slice_m+ mcolPitch);


                    float valf = row_up[i]*detf;
                    float valm = row_down[i]*detm;
                    float K= valf-valm;
                    row_metric[i]= K*K;
                    
                    updateF[0]= 2*K*gradI2.x*detf;
                    updateF[1]= 2*K*gradI2.y*detf;
                    updateF[2]= 2*K*gradI2.z*detf;
                    updateM[0]= -2*K*gradJ2.x*detm;
                    updateM[1]= -2*K*gradJ2.y*detm;
                    updateM[2]= -2*K*gradJ2.z*detm;
                }


                ////////////////////////   at x+1 ///////////////////////////////////////////
                {

                    for(int h=1;h<2;h++)
                    {
                        int nindex[3]={(int)i,(int)j,(int)k};
                        nindex[phase]+=h;

                        if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
                        {
                            if(mid_ind-h>=0)
                                a= c_Kernel[mid_ind-h];
                            else
                                a=0;
                            a_b=0;

                            float detf2 = ComputeSingleJacobianMatrixAtIndex(def_FINV,nindex[0],nindex[1],nindex[2],h,phase,phase_xyz)+1;
                            float detm2 = ComputeSingleJacobianMatrixAtIndex(def_MINV,nindex[0],nindex[1],nindex[2],h,phase,phase_xyz)+1;

                            if(detf2 <=0)
                                detf2=1E-5;
                            if(detm2 <=0)
                                detm2=1E-5;

                            float detf= mf(detf2);
                            float detm= mf(detm2);

                            float3 gradI2= ComputeImageGradient(up_img,nindex[0],nindex[1],nindex[2]);
                            float3 gradJ2= ComputeImageGradient(down_img,nindex[0],nindex[1],nindex[2]);

                            float gradI[3]={gradI2.x,gradI2.y,gradI2.z};
                            float gradJ[3]={gradJ2.x,gradJ2.y,gradJ2.z};


                            size_t upitch= up_img.pitch;
                            size_t uslicePitch= upitch*d_sz[1]*nindex[2];
                            size_t ucolPitch= nindex[1]*upitch;
                            char *u_ptr= (char *)(up_img.ptr);
                            char * slice_u= u_ptr+  uslicePitch;
                            float * row_up= (float *)(slice_u+ ucolPitch);

                            size_t dpitch= down_img.pitch;
                            size_t dslicePitch= dpitch*d_sz[1]*nindex[2];
                            size_t dcolPitch= nindex[1]*dpitch;
                            char *d_ptr= (char *)(down_img.ptr);
                            char * slice_d= d_ptr+  dslicePitch;
                            float * row_down= (float *)(slice_d+ dcolPitch);

                            float fval = row_up[nindex[0]];
                            float mval = row_down[nindex[0]];


                            float K =(fval*detf -mval*detm);
                            updateF[phase_xyz]+= 2 * K *  ( gradI[phase_xyz]*detf*a_b  + d_new_phase[phase_xyz]*dmf(detf2)* fval* -0.5/d_spc[phase]/h) ;
                            updateM[phase_xyz]-= 2*  K *  ( gradJ[phase_xyz]*detm*a_b  + d_new_phase[phase_xyz]*dmf(detm2)* mval* -0.5/d_spc[phase]/h);
                                                        
                        }
                    }
             } // x+1



                ////////////////////////   at x-1 ///////////////////////////////////////////
                {
                    for(int h=1;h<2;h++)
                    {
                        int nindex[3]={(int)i,(int)j,(int)k};
                        nindex[phase]-=h;


                        if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
                        {
                            if(mid_ind-h>=0)
                                a= c_Kernel[mid_ind-h];
                            else
                                a=0;
                            a_b=0;

                            float detf2 = ComputeSingleJacobianMatrixAtIndex(def_FINV,nindex[0],nindex[1],nindex[2],h,phase,phase_xyz)+1;
                            float detm2 = ComputeSingleJacobianMatrixAtIndex(def_MINV,nindex[0],nindex[1],nindex[2],h,phase,phase_xyz)+1;

                            if(detf2 <=0)
                                detf2=1E-5;
                            if(detm2 <=0)
                                detm2=1E-5;

                            float detf= mf(detf2);
                            float detm= mf(detm2);

                            float3 gradI2= ComputeImageGradient(up_img,nindex[0],nindex[1],nindex[2]);
                            float3 gradJ2= ComputeImageGradient(down_img,nindex[0],nindex[1],nindex[2]);

                            float gradI[3]={gradI2.x,gradI2.y,gradI2.z};
                            float gradJ[3]={gradJ2.x,gradJ2.y,gradJ2.z};

                            size_t upitch= up_img.pitch;
                            size_t uslicePitch= upitch*d_sz[1]*nindex[2];
                            size_t ucolPitch= nindex[1]*upitch;
                            char *u_ptr= (char *)(up_img.ptr);
                            char * slice_u= u_ptr+  uslicePitch;
                            float * row_up= (float *)(slice_u+ ucolPitch);

                            size_t dpitch= down_img.pitch;
                            size_t dslicePitch= dpitch*d_sz[1]*nindex[2];
                            size_t dcolPitch= nindex[1]*dpitch;
                            char *d_ptr= (char *)(down_img.ptr);
                            char * slice_d= d_ptr+  dslicePitch;
                            float * row_down= (float *)(slice_d+ dcolPitch);

                            float fval = row_up[nindex[0]];
                            float mval = row_down[nindex[0]];


                            float K =(fval*detf -mval*detm);
                            updateF[phase_xyz]+=  2*  K*  ( gradI[phase_xyz]*detf*a_b  + d_new_phase[phase_xyz]*dmf(detf2)*fval*  0.5/d_spc[phase]/h);
                            updateM[phase_xyz]-=  2*  K*  ( gradJ[phase_xyz]*detm*a_b  + d_new_phase[phase_xyz]*dmf(detm2)* mval* 0.5/d_spc[phase]/h);
                            
                        }
                    }
               } // x-1


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


void ComputeMetric_MSJac_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img,
		   int3 data_sz, float3 data_spc, 
		   float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
		   cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV,
     cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
     const float3 phase_vector,
     int kernel_sz, float *h_kernel, float &metric_value		   )
{
    float h_d_dir[]= {d00,d01,d02,d10,d11,d12,d20,d21,d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));      
    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));      
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));

    gpuErrchk(cudaMemcpyToSymbol(c_Kernel, h_kernel, kernel_sz * sizeof(float)));


    cudaPitchedPtr metric_image={0};
    cudaExtent extent =  make_cudaExtent(up_img.pitch,data_sz.y,data_sz.z);
    cudaMalloc3D(&metric_image, extent);
    cudaMemset3D(metric_image,0,extent);


    float new_phase[3];
    new_phase[0]= d00*phase_vector.x + d01*phase_vector.y +d02*phase_vector.z ;
    new_phase[1]= d10*phase_vector.x + d11*phase_vector.y +d12*phase_vector.z ;
    new_phase[2]= d20*phase_vector.x + d21*phase_vector.y +d22*phase_vector.z ;
    gpuErrchk(cudaMemcpyToSymbol(d_new_phase, &new_phase, 3 * sizeof(float)));

    int phase_xyz,phase;
    if( (fabs(phase_vector.x) > fabs(phase_vector.y))  && (fabs(phase_vector.x) > fabs(phase_vector.z)))
        phase=0;
    else if( (fabs(phase_vector.y) > fabs(phase_vector.x))  && (fabs(phase_vector.y) > fabs(phase_vector.z)))
        phase=1;
    else phase=2;

    if( (fabs(new_phase[0]) > fabs(new_phase[1]))  && (fabs(new_phase[0]) > fabs(new_phase[2])))
        phase_xyz=0;
    else if( (fabs(new_phase[1]) > fabs(new_phase[0]))  && (fabs(new_phase[1]) > fabs(new_phase[2])))
        phase_xyz=1;
    else phase_xyz=2;


    
    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x/PER_GROUP), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z) );

    ComputeMetric_MSJac_kernel<<< blockSize,gridSize>>>( up_img, down_img, def_FINV,def_MINV, updateFieldF, updateFieldM, metric_image, phase, phase_xyz, kernel_sz );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    float* dev_out;
    float out;
    cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

    ScalarFindSum<<<gSize, bSize>>>((float *)metric_image.ptr, metric_image.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
    cudaDeviceSynchronize();
    ScalarFindSum<<<1, bSize>>>(dev_out, gSize, dev_out);
    cudaDeviceSynchronize();

    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_out);
    metric_value=out/data_sz.x/data_sz.y/data_sz.z;

    cudaFree(metric_image.ptr);
}		   





__global__ void
ComputeMetric_MSJacSingle_kernel( cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                                  cudaTextureObject_t up_grad_img_x, cudaTextureObject_t up_grad_img_y, cudaTextureObject_t up_grad_img_z,
                                  cudaTextureObject_t down_grad_img_x, cudaTextureObject_t down_grad_img_y, cudaTextureObject_t down_grad_img_z,
                            cudaPitchedPtr def_FINV,
                            cudaPitchedPtr updateFieldFINV,
                            cudaPitchedPtr metric_image,
                            int phase, int phase_xyz,
                            int kernel_sz )
{
    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {

            float update[3]={0,0,0};

            if(i>=1 && j>=1 && k>=1 && i<=d_sz[0]-2  && j<=d_sz[1]-2  && k<=d_sz[2]-2)
            {
                int mid_ind=(kernel_sz-1)/2;
                float b= c_Kernel[  mid_ind ];

                ////////////////////////   at x ///////////////////////////////////////////
                {
                    size_t upitch= up_img.pitch;
                    size_t uslicePitch= upitch*d_sz[1]*k;
                    size_t ucolPitch= j*upitch;
                    char *u_ptr= (char *)(up_img.ptr);
                    char * slice_u= u_ptr+  uslicePitch;
                    float * row_up= (float *)(slice_u+ ucolPitch);

                    size_t dpitch= down_img.pitch;
                    size_t dslicePitch= dpitch*d_sz[1]*k;
                    size_t dcolPitch= j*dpitch;
                    char *d_ptr= (char *)(down_img.ptr);
                    char * slice_d= d_ptr+  dslicePitch;
                    float * row_down= (float *)(slice_d+ dcolPitch);

                    size_t mpitch= metric_image.pitch;
                    size_t mslicePitch= mpitch*d_sz[1]*k;
                    size_t mcolPitch= j*mpitch;
                    char *m_ptr= (char *)(metric_image.ptr);
                    char * slice_m= m_ptr+  mslicePitch;
                    float * row_metric= (float *)(slice_m+ mcolPitch);

                    size_t fpitch= def_FINV.pitch;
                    size_t fslicePitch= fpitch*d_sz[1]*k;
                    size_t fcolPitch= j*fpitch;
                    char *f_ptr= (char *)(def_FINV.ptr);
                    char * slice_f= f_ptr+  fslicePitch;
                    float * row_f= (float *)(slice_f+ fcolPitch);

                    float valf = row_up[i];
                    float valm = row_down[i];
                    float det = ComputeSingleJacobianMatrixAtIndex(def_FINV,i,j,k,1,phase,phase_xyz);
                    if(det<=-1)
                        det=-1+1E-5;


                    float K= valf*(1+det)-valm*(1-det);
                    row_metric[i]= K*K;

                    float x= (d_dir[0]*i  + d_dir[1]*j + d_dir[2]*k)* d_spc[0] ;
                    float y= (d_dir[3]*i  + d_dir[4]*j + d_dir[5]*k)* d_spc[1] ;
                    float z= (d_dir[6]*i  + d_dir[7]*j + d_dir[8]*k)* d_spc[2] ;

                    float up_grad_x=0,up_grad_y=0,up_grad_z=0;
                    float down_grad_x=0,down_grad_y=0,down_grad_z=0;

                    {
                        float xw= x + row_f[3*i];
                        float yw= y + row_f[3*i+1];
                        float zw= z + row_f[3*i+2];

                        float iw = (d_dir[0]*xw  + d_dir[3]*yw  + d_dir[6]*zw)/ d_spc[0] ;
                        float jw = (d_dir[1]*xw  + d_dir[4]*yw  + d_dir[7]*zw)/ d_spc[1] ;
                        float kw = (d_dir[2]*xw  + d_dir[5]*yw  + d_dir[8]*zw)/ d_spc[2] ;

                        up_grad_x =tex3D<float>(up_grad_img_x, iw+0.5, jw +0.5, kw+0.5);
                        up_grad_y =tex3D<float>(up_grad_img_y, iw+0.5, jw +0.5, kw+0.5);
                        up_grad_z =tex3D<float>(up_grad_img_z, iw+0.5, jw +0.5, kw+0.5);
                    }
                    {
                        float xw= x - row_f[3*i];
                        float yw= y - row_f[3*i+1];
                        float zw= z - row_f[3*i+2];

                        float iw = (d_dir[0]*xw  + d_dir[3]*yw  + d_dir[6]*zw)/ d_spc[0] ;
                        float jw = (d_dir[1]*xw  + d_dir[4]*yw  + d_dir[7]*zw)/ d_spc[1] ;
                        float kw = (d_dir[2]*xw  + d_dir[5]*yw  + d_dir[8]*zw)/ d_spc[2] ;

                        down_grad_x =tex3D<float>(down_grad_img_x, iw+0.5, jw +0.5, kw+0.5);
                        down_grad_y =tex3D<float>(down_grad_img_y, iw+0.5, jw +0.5, kw+0.5);
                        down_grad_z =tex3D<float>(down_grad_img_z, iw+0.5, jw +0.5, kw+0.5);
                    }


                    update[0]= 2*K*(  (up_grad_x*(1+det)) -  (down_grad_x*(1-det)) );
                    update[1]= 2*K*(  (up_grad_y*(1+det)) -  (down_grad_y*(1-det)) );
                    update[2]= 2*K*(  (up_grad_z*(1+det)) -  (down_grad_z*(1-det)) );


                }


                ////////////////////////   at x+1 ///////////////////////////////////////////
                {
                    for(int h=1;h<2;h++)
                    {
                        int nindex[3]={(int)i,(int)j,(int)k};
                        nindex[phase]+=h;

                        if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
                        {
                            float a=0;
                            if(mid_ind-h >=0 )
                                a=c_Kernel[  mid_ind -h];
                            float a_b= a/b;
                            a_b=0;


                            size_t upitch= up_img.pitch;
                            size_t uslicePitch= upitch*d_sz[1]*nindex[2];
                            size_t ucolPitch= nindex[1]*upitch;
                            char *u_ptr= (char *)(up_img.ptr);
                            char * slice_u= u_ptr+  uslicePitch;
                            float * row_up= (float *)(slice_u+ ucolPitch);

                            size_t dpitch= down_img.pitch;
                            size_t dslicePitch= dpitch*d_sz[1]*nindex[2];
                            size_t dcolPitch= nindex[1]*dpitch;
                            char *d_ptr= (char *)(down_img.ptr);
                            char * slice_d= d_ptr+  dslicePitch;
                            float * row_down= (float *)(slice_d+ dcolPitch);

                            size_t fpitch= def_FINV.pitch;
                            size_t fslicePitch= fpitch*d_sz[1]*nindex[2];
                            size_t fcolPitch= nindex[1]*fpitch;
                            char *f_ptr= (char *)(def_FINV.ptr);
                            char * slice_f= f_ptr+  fslicePitch;
                            float * row_f= (float *)(slice_f+ fcolPitch);

                            float valf = row_up[nindex[0]];
                            float valm = row_down[nindex[0]];
                            float det = ComputeSingleJacobianMatrixAtIndex(def_FINV,nindex[0],nindex[1],nindex[2],1,phase,phase_xyz);
                            if(det<=-1)
                                det=-1+1E-5;

                            float K = valf*(1+det)-valm*(1-det);


                            float x= (d_dir[0]*nindex[0]  + d_dir[1]*nindex[1] + d_dir[2]*nindex[2])* d_spc[0] ;
                            float y= (d_dir[3]*nindex[0]  + d_dir[4]*nindex[1] + d_dir[5]*nindex[2])* d_spc[1] ;
                            float z= (d_dir[6]*nindex[0]  + d_dir[7]*nindex[1] + d_dir[8]*nindex[2])* d_spc[2] ;

                            float up_grad_x=0,up_grad_y=0,up_grad_z=0;
                            float down_grad_x=0,down_grad_y=0,down_grad_z=0;

                            {
                                float xw= x + row_f[3*nindex[0]];
                                float yw= y + row_f[3*nindex[0]+1];
                                float zw= z + row_f[3*nindex[0]+2];

                                float iw = (d_dir[0]*xw  + d_dir[3]*yw  + d_dir[6]*zw)/ d_spc[0] ;
                                float jw = (d_dir[1]*xw  + d_dir[4]*yw  + d_dir[7]*zw)/ d_spc[1] ;
                                float kw = (d_dir[2]*xw  + d_dir[5]*yw  + d_dir[8]*zw)/ d_spc[2] ;

                                up_grad_x =tex3D<float>(up_grad_img_x, iw+0.5, jw +0.5, kw+0.5);
                                up_grad_y =tex3D<float>(up_grad_img_y, iw+0.5, jw +0.5, kw+0.5);
                                up_grad_z =tex3D<float>(up_grad_img_z, iw+0.5, jw +0.5, kw+0.5);
                            }
                            {
                                float xw= x - row_f[3*nindex[0]];
                                float yw= y - row_f[3*nindex[0]+1];
                                float zw= z - row_f[3*nindex[0]+2];

                                float iw = (d_dir[0]*xw  + d_dir[3]*yw  + d_dir[6]*zw)/ d_spc[0] ;
                                float jw = (d_dir[1]*xw  + d_dir[4]*yw  + d_dir[7]*zw)/ d_spc[1] ;
                                float kw = (d_dir[2]*xw  + d_dir[5]*yw  + d_dir[8]*zw)/ d_spc[2] ;

                                down_grad_x =tex3D<float>(down_grad_img_x, iw+0.5, jw +0.5, kw+0.5);
                                down_grad_y =tex3D<float>(down_grad_img_y, iw+0.5, jw +0.5, kw+0.5);
                                down_grad_z =tex3D<float>(down_grad_img_z, iw+0.5, jw +0.5, kw+0.5);
                            }

                            float gradI[3]={up_grad_x,up_grad_y,up_grad_z};
                            float gradJ[3]={down_grad_x,down_grad_y,down_grad_z};

                            update[phase_xyz]+= 2 * K * (   (gradI[phase_xyz]*a_b*(1+det) + d_new_phase[phase_xyz]*valf*-0.5/d_spc[phase]/h)
                                                            -(  gradJ[phase_xyz]*a_b*(1-det) - d_new_phase[phase_xyz]*valm*-0.5/d_spc[phase]/h) );


                        }
                    }
                }    // x+1



                ////////////////////////   at x-1 ///////////////////////////////////////////
                {
                    for(int h=1;h<2;h++)
                    {
                        int nindex[3]={(int)i,(int)j,(int)k};
                        nindex[phase]-=h;

                        if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
                        {
                            float a=0;
                            if(mid_ind-h >=0 )
                                a=c_Kernel[  mid_ind -h];
                            float a_b= a/b;
                            a_b=0;


                            size_t upitch= up_img.pitch;
                            size_t uslicePitch= upitch*d_sz[1]*nindex[2];
                            size_t ucolPitch= nindex[1]*upitch;
                            char *u_ptr= (char *)(up_img.ptr);
                            char * slice_u= u_ptr+  uslicePitch;
                            float * row_up= (float *)(slice_u+ ucolPitch);

                            size_t dpitch= down_img.pitch;
                            size_t dslicePitch= dpitch*d_sz[1]*nindex[2];
                            size_t dcolPitch= nindex[1]*dpitch;
                            char *d_ptr= (char *)(down_img.ptr);
                            char * slice_d= d_ptr+  dslicePitch;
                            float * row_down= (float *)(slice_d+ dcolPitch);

                            size_t fpitch= def_FINV.pitch;
                            size_t fslicePitch= fpitch*d_sz[1]*nindex[2];
                            size_t fcolPitch= nindex[1]*fpitch;
                            char *f_ptr= (char *)(def_FINV.ptr);
                            char * slice_f= f_ptr+  fslicePitch;
                            float * row_f= (float *)(slice_f+ fcolPitch);

                            float valf = row_up[nindex[0]];
                            float valm = row_down[nindex[0]];
                            float det = ComputeSingleJacobianMatrixAtIndex(def_FINV,nindex[0],nindex[1],nindex[2],1,phase,phase_xyz);
                            if(det<=-1)
                                det=-1+1E-5;

                            float K = valf*(1+det)-valm*(1-det);



                            float x= (d_dir[0]*nindex[0]  + d_dir[1]*nindex[1] + d_dir[2]*nindex[2])* d_spc[0] ;
                            float y= (d_dir[3]*nindex[0]  + d_dir[4]*nindex[1] + d_dir[5]*nindex[2])* d_spc[1] ;
                            float z= (d_dir[6]*nindex[0]  + d_dir[7]*nindex[1] + d_dir[8]*nindex[2])* d_spc[2] ;

                            float up_grad_x=0,up_grad_y=0,up_grad_z=0;
                            float down_grad_x=0,down_grad_y=0,down_grad_z=0;

                            {
                                float xw= x + row_f[3*nindex[0]];
                                float yw= y + row_f[3*nindex[0]+1];
                                float zw= z + row_f[3*nindex[0]+2];

                                float iw = (d_dir[0]*xw  + d_dir[3]*yw  + d_dir[6]*zw)/ d_spc[0] ;
                                float jw = (d_dir[1]*xw  + d_dir[4]*yw  + d_dir[7]*zw)/ d_spc[1] ;
                                float kw = (d_dir[2]*xw  + d_dir[5]*yw  + d_dir[8]*zw)/ d_spc[2] ;

                                up_grad_x =tex3D<float>(up_grad_img_x, iw+0.5, jw +0.5, kw+0.5);
                                up_grad_y =tex3D<float>(up_grad_img_y, iw+0.5, jw +0.5, kw+0.5);
                                up_grad_z =tex3D<float>(up_grad_img_z, iw+0.5, jw +0.5, kw+0.5);
                            }
                            {
                                float xw= x - row_f[3*nindex[0]];
                                float yw= y - row_f[3*nindex[0]+1];
                                float zw= z - row_f[3*nindex[0]+2];

                                float iw = (d_dir[0]*xw  + d_dir[3]*yw  + d_dir[6]*zw)/ d_spc[0] ;
                                float jw = (d_dir[1]*xw  + d_dir[4]*yw  + d_dir[7]*zw)/ d_spc[1] ;
                                float kw = (d_dir[2]*xw  + d_dir[5]*yw  + d_dir[8]*zw)/ d_spc[2] ;

                                down_grad_x =tex3D<float>(down_grad_img_x, iw+0.5, jw +0.5, kw+0.5);
                                down_grad_y =tex3D<float>(down_grad_img_y, iw+0.5, jw +0.5, kw+0.5);
                                down_grad_z =tex3D<float>(down_grad_img_z, iw+0.5, jw +0.5, kw+0.5);
                            }

                            float gradI[3]={up_grad_x,up_grad_y,up_grad_z};
                            float gradJ[3]={down_grad_x,down_grad_y,down_grad_z};


                            update[phase_xyz]+= 2 * K * (   (gradI[phase_xyz]*a_b*(1+det) + d_new_phase[phase_xyz]*valf*0.5/d_spc[phase]/h)
                                                            -(gradJ[phase_xyz]*a_b*(1-det) - d_new_phase[phase_xyz]*valm*0.5/d_spc[phase]/h) );
                        }
                    }
                }    // x+1



                size_t upitch= updateFieldFINV.pitch;
                size_t uslicePitch= upitch*d_sz[1]*k;
                size_t ucolPitch= j*upitch;
                char *u_ptr= (char *)(updateFieldFINV.ptr);
                char * slice_u= u_ptr+  uslicePitch;
                float * row_uf= (float *)(slice_u+ ucolPitch);

                row_uf[3*i]=-update[0];
                row_uf[3*i+1]= -update[1];
                row_uf[3*i+2]= -update[2];
            }
        }
     }
}




void ComputeMetric_MSJacSingle_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                                    cudaTextureObject_t up_grad_img_x, cudaTextureObject_t up_grad_img_y, cudaTextureObject_t up_grad_img_z,
                                    cudaTextureObject_t down_grad_img_x, cudaTextureObject_t down_grad_img_y, cudaTextureObject_t down_grad_img_z,
     int3 data_sz, float3 data_spc,
     float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
     cudaPitchedPtr def_FINV,
        cudaPitchedPtr updateFieldFINV,
                   float3 phase_vector,int kernel_sz, float* h_kernel, float &metric_value)
{

    float h_d_dir[]= {d00,d01,d02,d10,d11,d12,d20,d21,d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));
    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));

    gpuErrchk(cudaMemcpyToSymbol(c_Kernel, h_kernel, kernel_sz * sizeof(float)));


    cudaPitchedPtr metric_image={0};
    cudaExtent extent =  make_cudaExtent(up_img.pitch,data_sz.y,data_sz.z);
    cudaMalloc3D(&metric_image, extent);
    cudaMemset3D(metric_image,0,extent);


    float new_phase[3];
    new_phase[0]= d00*phase_vector.x + d01*phase_vector.y +d02*phase_vector.z ;
    new_phase[1]= d10*phase_vector.x + d11*phase_vector.y +d12*phase_vector.z ;
    new_phase[2]= d20*phase_vector.x + d21*phase_vector.y +d22*phase_vector.z ;
    gpuErrchk(cudaMemcpyToSymbol(d_new_phase, &new_phase, 3 * sizeof(float)));

    int phase_xyz,phase;
    if( (fabs(phase_vector.x) > fabs(phase_vector.y))  && (fabs(phase_vector.x) > fabs(phase_vector.z)))
        phase=0;
    else if( (fabs(phase_vector.y) > fabs(phase_vector.x))  && (fabs(phase_vector.y) > fabs(phase_vector.z)))
        phase=1;
    else phase=2;

    if( (fabs(new_phase[0]) > fabs(new_phase[1]))  && (fabs(new_phase[0]) > fabs(new_phase[2])))
        phase_xyz=0;
    else if( (fabs(new_phase[1]) > fabs(new_phase[0]))  && (fabs(new_phase[1]) > fabs(new_phase[2])))
        phase_xyz=1;
    else phase_xyz=2;



    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x/PER_GROUP), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z) );

    ComputeMetric_MSJacSingle_kernel<<< blockSize,gridSize>>>( up_img, down_img, up_grad_img_x,up_grad_img_y,up_grad_img_z,down_grad_img_x,down_grad_img_y,down_grad_img_z,def_FINV,updateFieldFINV,  metric_image, phase, phase_xyz, kernel_sz );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    float* dev_out;
    float out;
    cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

    ScalarFindSum<<<gSize, bSize>>>((float *)metric_image.ptr, metric_image.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
    cudaDeviceSynchronize();
    ScalarFindSum<<<1, bSize>>>(dev_out, gSize, dev_out);
    cudaDeviceSynchronize();

    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_out);
    metric_value=out/data_sz.x/data_sz.y/data_sz.z;

    cudaFree(metric_image.ptr);

}


__global__ void
Compute_K_image( cudaPitchedPtr up_img, cudaPitchedPtr down_img, cudaPitchedPtr K_img)
{
    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            size_t pitch= up_img.pitch;
            size_t slicePitch= pitch*d_sz[1]*k;
            size_t colPitch= j*pitch;

            char *u_ptr= (char *)(up_img.ptr);
            char * slice_u= u_ptr+  slicePitch;
            float * row_up= (float *)(slice_u+ colPitch);

            char *d_ptr= (char *)(down_img.ptr);
            char * slice_d= d_ptr+  slicePitch;
            float * row_down= (float *)(slice_d+ colPitch);

            char *K_ptr= (char *)(K_img.ptr);
            char * slice_K= K_ptr+  slicePitch;
            float * row_K= (float *)(slice_K+ colPitch);


            float a = row_up[i];
            float b= row_down[i];

            float a_b= a+b;

            if(a_b>LIMCCSK)
                row_K[i] = 2*a*b/a_b;
        }
    }

}




__global__ void
ComputeMetric_CCSK_kernel( cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                           cudaPitchedPtr K_img, cudaPitchedPtr str_img,
                           cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                           cudaPitchedPtr metric_img)
{

    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;


    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {

            size_t pitch= K_img.pitch;
            char *K_ptr= (char *)(K_img.ptr);
            char *str_ptr= (char *)(str_img.ptr);
            char *metric_ptr= (char *)(metric_img.ptr);
            char *up_ptr= (char *)(up_img.ptr);
            char *down_ptr= (char *)(down_img.ptr);

            size_t slicePitch= pitch*d_sz[1]*k;
            size_t colPitch= j*pitch;

            char * slice_M= metric_ptr+  slicePitch;
            float * row_M= (float *)(slice_M+ colPitch);
            char * slice_u= up_ptr+  slicePitch;
            float * row_up= (float *)(slice_u+ colPitch);
            char * slice_d= down_ptr+  slicePitch;
            float * row_down= (float *)(slice_d+ colPitch);


            float updateF[3]={0,0,0};
            float updateM[3]={0,0,0};


            int start[3],end[3];

            start[2]=k-WIN_RAD_Z;
            if(start[2]<0)
                start[2]=0;
            start[1]=j-WIN_RAD;
            if(start[1]<0)
                start[1]=0;
            start[0]=i-WIN_RAD;
            if(start[0]<0)
                start[0]=0;

            end[2]=k+WIN_RAD_Z+1;
            if(end[2]>d_sz[2])
                end[2]=d_sz[2];
            end[1]=j+WIN_RAD+1;
            if(end[1]>d_sz[1])
                end[1]=d_sz[1];
            end[0]=i+WIN_RAD+1;
            if(end[0]>d_sz[0])
                end[0]=d_sz[0];


            float suma2 = 0.0;
            float suma = 0.0;
            float  sumac=0;
            float sumc2 = 0.0;
            float sumc = 0.0;
            int N=0;

            float valK_center;
            float valS_center;

            for(int z=start[2];z<end[2];z++)
            {
                size_t KslicePitch= pitch*d_sz[1]*z;

                for(int y=start[1];y<end[1];y++)
                {
                    size_t KcolPitch= y*pitch;

                    char * slice_K= K_ptr+  KslicePitch;
                    float * row_K= (float *)(slice_K+ KcolPitch);
                    char * slice_str= str_ptr+  KslicePitch;
                    float * row_str= (float *)(slice_str+ KcolPitch);

                    for(int x=start[0];x<end[0];x++)
                    {
                        float Kim= row_K[x];
                        float c= row_str[x];

                        if(z==k && y==j && x==i)
                        {
                            valK_center=Kim;
                            valS_center=c;
                        }

                        suma2 += Kim * Kim;
                        suma += Kim;
                        sumc2 += c * c;
                        sumc += c;
                        sumac += Kim*c;

                        N++;
                    }
                }
            }


            float Kmean = suma/N;
            float Smean= sumc/N;

            float valK = valK_center-Kmean;
            float valS = valS_center -Smean;

            float sKK = suma2 - Kmean*suma;
            float sSS = sumc2 - Smean*sumc;
            float sKS = sumac - Kmean*sumc;


            float sSS_sKK = sSS * sKK;
            if(fabs(sSS_sKK) > LIMCCSK && fabs(sKK) > LIMCCSK )
            {

                row_M[i]= -sKS*sKS/ sSS_sKK;

                float first_term= -2*sKS/sSS_sKK *(valS - sKS/sKK*valK);
                float fval = row_up[i];
                float mval = row_down[i];

                float sm_mval_fval=(mval+fval);

                if(sm_mval_fval*sm_mval_fval > LIMCCSK)
                {
                    {
                        float grad_term =2* mval*mval/sm_mval_fval/sm_mval_fval;
                        float3 gradI2= ComputeImageGradient(up_img,i,j,k);

                        updateF[0]= first_term*grad_term * gradI2.x;
                        updateF[1]= first_term*grad_term * gradI2.y;
                        updateF[2]= first_term*grad_term * gradI2.z;
                    }
                    {
                        float grad_term= 2* fval*fval/sm_mval_fval/sm_mval_fval;
                        float3 gradJ2= ComputeImageGradient(down_img,i,j,k);

                        updateM[0]= first_term*grad_term * gradJ2.x;
                        updateM[1]= first_term*grad_term * gradJ2.y;
                        updateM[2]= first_term*grad_term * gradJ2.z;
                    }
                }
            }

            size_t upitch= updateFieldF.pitch;
            size_t uslicePitch= upitch*d_sz[1]*k;
            size_t ucolPitch= j*upitch;
            char *uf_ptr= (char *)(updateFieldF.ptr);
            char * slice_uf= uf_ptr+  uslicePitch;
            float * row_uf= (float *)(slice_uf+ ucolPitch);

            size_t dpitch= updateFieldM.pitch;
            size_t dslicePitch= dpitch*d_sz[1]*k;
            size_t dcolPitch= j*dpitch;
            char *df_ptr= (char *)(updateFieldM.ptr);
            char * slice_df= df_ptr+  dslicePitch;
            float * row_df= (float *)(slice_df+ dcolPitch);


            row_uf[3*i]= updateF[0];
            row_uf[3*i+1]= updateF[1];
            row_uf[3*i+2]= updateF[2];

            row_df[3*i]= updateM[0];
            row_df[3*i+1]= updateM[1];
            row_df[3*i+2]= updateM[2];


        }
    }
}


void ComputeMetric_CCSK_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img, cudaPitchedPtr str_img,
                             int3 data_sz, float3 data_spc,
                             float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                             cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                             float &metric_value)

{
    float h_d_dir[]= {d00,d01,d02,d10,d11,d12,d20,d21,d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));
    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));

    cudaPitchedPtr metric_image={0};
    //cudaExtent extent =  make_cudaExtent(1*sizeof(float)*data_sz.x,data_sz.y,data_sz.z);
    cudaExtent extent =  make_cudaExtent(1*up_img.pitch,data_sz.y,data_sz.z);
    cudaMalloc3D(&metric_image, extent);
    cudaMemset3D(metric_image,0,extent);

    cudaPitchedPtr K_image={0};
    cudaMalloc3D(&K_image, extent);
    cudaMemset3D(K_image,0,extent);


    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x/PER_GROUP), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z) );

    Compute_K_image<<< blockSize,gridSize>>>( up_img, down_img,K_image);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    ComputeMetric_CCSK_kernel<<< blockSize,gridSize>>>(up_img,down_img, K_image, str_img, updateFieldF, updateFieldM, metric_image );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    float* dev_out;
    float out;
    cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

    ScalarFindSum<<<gSize, bSize>>>((float *)metric_image.ptr, metric_image.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
    cudaDeviceSynchronize();
    ScalarFindSum<<<1, bSize>>>(dev_out, gSize, dev_out);
    cudaDeviceSynchronize();

    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_out);
    metric_value=out/data_sz.x/data_sz.y/data_sz.z;



    cudaFree(metric_image.ptr);
    cudaFree(K_image.ptr);

}











__global__ void
ComputeMetric_CC_kernel( cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                           cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                           cudaPitchedPtr metric_img)
{

    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {

            size_t pitch= up_img.pitch;
            char *metric_ptr= (char *)(metric_img.ptr);
            char *up_ptr= (char *)(up_img.ptr);
            char *down_ptr= (char *)(down_img.ptr);

            size_t slicePitch= pitch*d_sz[1]*k;
            size_t colPitch= j*pitch;

            char * slice_M= metric_ptr+  slicePitch;
            float * row_M= (float *)(slice_M+ colPitch);


            float updateF[3]={0,0,0};
            float updateM[3]={0,0,0};

            int start[3],end[3];

            start[2]=k-WIN_RAD_Z;
            if(start[2]<0)
                start[2]=0;
            start[1]=j-WIN_RAD;
            if(start[1]<0)
                start[1]=0;
            start[0]=i-WIN_RAD;
            if(start[0]<0)
                start[0]=0;

            end[2]=k+WIN_RAD_Z+1;
            if(end[2]>d_sz[2])
                end[2]=d_sz[2];
            end[1]=j+WIN_RAD+1;
            if(end[1]>d_sz[1])
                end[1]=d_sz[1];
            end[0]=i+WIN_RAD+1;
            if(end[0]>d_sz[0])
                end[0]=d_sz[0];

            float suma2 = 0.0;
            float suma = 0.0;
            float  sumac=0;
            float sumc2 = 0.0;
            float sumc = 0.0;
            int N=0;

            float valF_center;
            float valM_center;

            for(int z=start[2];z<end[2];z++)
            {
                size_t uslicePitch= pitch*d_sz[1]*z;

                for(int y=start[1];y<end[1];y++)
                {
                    size_t ucolPitch= y*pitch;

                    char * slice_u= up_ptr+  uslicePitch;
                    float * row_up= (float *)(slice_u+ ucolPitch);
                    char * slice_d= down_ptr+  uslicePitch;
                    float * row_down= (float *)(slice_d+ ucolPitch);

                    for(int x=start[0];x<end[0];x++)
                    {
                        float f= row_up[x];
                        float m= row_down[x];

                        if(z==k && y==j && x==i)
                        {
                            valF_center=f;
                            valM_center=m;
                        }

                        suma2 += f * f;
                        suma += f;
                        sumc2 += m * m;
                        sumc += m;
                        sumac += f*m;

                        N++;
                    }
                }
            }


            float Fmean = suma/N;
            float Mmean= sumc/N;

            float valF = valF_center -Fmean;
            float valM = valM_center -Mmean;

            float sFF = suma2 - Fmean*suma;
            float sMM = sumc2 - Mmean*sumc;
            float sFM = sumac - Fmean*sumc;

            float sFF_sMM = sFF * sMM;
            if(fabs(sFF_sMM) >LIMCC && fabs(sMM) > LIMCC)
            {
                row_M[i]+= -sFM*sFM/ sFF_sMM;

                float first_termF= -2* sFM/sFF_sMM *    (valM - sFM/sFF * valF) ;
                float first_termM= -2* sFM/sFF_sMM *    (valF - sFM/sMM * valM) ;

                float3 gradI= ComputeImageGradient(up_img,i,j,k);
                float3 gradJ= ComputeImageGradient(down_img,i,j,k);

                updateF[0] = first_termF * gradI.x;
                updateF[1] = first_termF * gradI.y;
                updateF[2] = first_termF * gradI.z;
                updateM[0] = first_termM * gradJ.x;
                updateM[1] = first_termM * gradJ.y;
                updateM[2] = first_termM * gradJ.z;
            }

            size_t upitch= updateFieldF.pitch;
            size_t uslicePitch= upitch*d_sz[1]*k;
            size_t ucolPitch= j*upitch;
            char *uf_ptr= (char *)(updateFieldF.ptr);
            char * slice_uf= uf_ptr+  uslicePitch;
            float * row_uf= (float *)(slice_uf+ ucolPitch);

            size_t dpitch= updateFieldM.pitch;
            size_t dslicePitch= dpitch*d_sz[1]*k;
            size_t dcolPitch= j*dpitch;
            char *df_ptr= (char *)(updateFieldM.ptr);
            char * slice_df= df_ptr+  dslicePitch;
            float * row_df= (float *)(slice_df+ dcolPitch);


            row_uf[3*i]= updateF[0];
            row_uf[3*i+1]= updateF[1];
            row_uf[3*i+2]= updateF[2];

            row_df[3*i]= updateM[0];
            row_df[3*i+1]= updateM[1];
            row_df[3*i+2]= updateM[2];

        }
    }
}

void ComputeMetric_CC_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                             int3 data_sz, float3 data_spc,
                             float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
                             cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                             float &metric_value)

{
    float h_d_dir[]= {d00,d01,d02,d10,d11,d12,d20,d21,d22};
    gpuErrchk(cudaMemcpyToSymbol(d_dir, &h_d_dir, 9 * sizeof(float)));
    float h_d_spc[]= {data_spc.x,data_spc.y,data_spc.z};
    gpuErrchk(cudaMemcpyToSymbol(d_spc, &h_d_spc, 3 * sizeof(float)));
    int h_d_sz[]= {data_sz.x,data_sz.y,data_sz.z};
    gpuErrchk(cudaMemcpyToSymbol(d_sz, &h_d_sz, 3 * sizeof(int)));

    cudaPitchedPtr metric_image={0};
    cudaExtent extent =  make_cudaExtent(up_img.pitch,data_sz.y,data_sz.z);
    cudaMalloc3D(&metric_image, extent);
    cudaMemset3D(metric_image,0,extent);


    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    const dim3 gridSize(std::ceil(1.*data_sz.x / blockSize.x/PER_GROUP), std::ceil(1.*data_sz.y / blockSize.y), std::ceil(1.*data_sz.z / blockSize.z) );


    ComputeMetric_CC_kernel<<< blockSize,gridSize>>>(up_img,down_img, updateFieldF, updateFieldM, metric_image );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    float* dev_out;
    float out;
    cudaMalloc((void**)&dev_out, sizeof(float)*gSize);

    ScalarFindSum<<<gSize, bSize>>>((float *)metric_image.ptr, metric_image.pitch/sizeof(float)*data_sz.y*data_sz.z,dev_out);
    cudaDeviceSynchronize();
    ScalarFindSum<<<1, bSize>>>(dev_out, gSize, dev_out);
    cudaDeviceSynchronize();

    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_out);
    metric_value=out/data_sz.x/data_sz.y/data_sz.z;

    cudaFree(metric_image.ptr);

}




#endif
