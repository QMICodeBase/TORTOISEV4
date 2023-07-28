#ifndef _COMPUTEMETRICDEV_CU
#define _COMPUTEMETRICDEV_CU


#include "cuda_utils.h"


#define BLOCKSIZE 64
#define PER_SLICE 1
#define PER_GROUP 1

extern __constant__ int d_sz[3];
extern __constant__ float d_dir[9];
extern __constant__ float d_spc[3];


const int bSize2=1024 ;
const int gSize2=24 ;

#define DPHI 0.05


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



__device__ void ComputeSingleJacobianMatrixAtIndex(cudaPitchedPtr field ,int i, int j,int k,float B[3][3])
{    
    B[0][0]=1;
    B[1][1]=1;
    B[2][2]=1;

    if(i<1 || i> d_sz[0]-2 || j<1 || j> d_sz[1]-2 || k<1 || k> d_sz[2]-2)
        return;

    float SD[3][3];
    SD[0][0]=d_dir[0]/d_spc[0];   SD[0][1]=d_dir[3]/d_spc[0];   SD[0][2]=d_dir[6]/d_spc[0];
    SD[1][0]=d_dir[1]/d_spc[1];   SD[1][1]=d_dir[4]/d_spc[1];   SD[1][2]=d_dir[7]/d_spc[1];
    SD[2][0]=d_dir[2]/d_spc[2];   SD[2][1]=d_dir[5]/d_spc[2];   SD[2][2]=d_dir[8]/d_spc[2];



    float A[3][3];

    float grad;
    {
        size_t pitch= field.pitch;
        char *ptr= (char *)(field.ptr);
        size_t slicePitch= pitch*d_sz[1]*k;
        size_t colPitch= j*pitch;
        char * slice= ptr+  slicePitch;
        float * row= (float *)(slice+ colPitch);

        grad=0.5*(row[3*(i+1)]- row[3*(i-1)]);
        A[0][0]=grad;
        grad=0.5*(row[3*(i+1)+1]- row[3*(i-1)+1]);
        A[1][0]=grad;
        grad=0.5*(row[3*(i+1)+2]- row[3*(i-1)+2]);
        A[2][0]=grad;
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
        A[0][1]=grad;
        grad=0.5*(row1[3*i+1]-row2[3*i+1]);
        A[1][1]=grad;
        grad=0.5*(row1[3*i+2]-row2[3*i+2]);
        A[2][1]=grad;
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
        A[0][2]=grad;
        grad=0.5*(row1[3*i+1]-row2[3*i+1]);
        A[1][2]=grad;
        grad=0.5*(row1[3*i+2]-row2[3*i+2]);
        A[2][2]=grad;
    }


    MatrixMultiply(A,SD,B);
    B[0][0]+=1;
    B[1][1]+=1;
    B[2][2]+=1;

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


__device__ void ComputeRotationMatrix(float A[3][3], float R[3][3])
{
    float AAT[3][3]={0};
    float AT[3][3];

    MatrixTranspose(A,AT);
    MatrixMultiply(A,AT,AAT);


    float vals[3]={0};
    float vecs[3][3]={0};


    ComputeEigens(AAT, vals, vecs);

    vals[0]= pow(vals[0],-0.5);
    vals[1]= pow(vals[1],-0.5);
    vals[2]= pow(vals[2],-0.5);

    float valsm[3][3]={0};
    valsm[0][0]=vals[0];
    valsm[1][1]=vals[1];
    valsm[2][2]=vals[2];

    float UT[3][3];
    MatrixTranspose(vecs,UT);

    float temp[3][3]={0}, temp2[3][3]={0};
    MatrixMultiply(vecs,valsm,temp);
    MatrixMultiply(temp,UT,temp2);
    MatrixMultiply(temp2,A,R);

}


__global__ void
ComputeMetric_DEV_kernel( cudaPitchedPtr up_img, cudaPitchedPtr down_img,
                            cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV, 
                            cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                            cudaPitchedPtr metric_image)

{
    uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    for(int i=PER_GROUP*ii;i<PER_GROUP*ii+PER_GROUP;i++)
    {
        if(i<d_sz[0] && j <d_sz[1] && k<d_sz[2])
        {
            float SD[3][3];
            SD[0][0]=d_dir[0]/d_spc[0];   SD[0][1]=d_dir[3]/d_spc[0];   SD[0][2]=d_dir[6]/d_spc[0];
            SD[1][0]=d_dir[1]/d_spc[1];   SD[1][1]=d_dir[4]/d_spc[1];   SD[1][2]=d_dir[7]/d_spc[1];
            SD[2][0]=d_dir[2]/d_spc[2];   SD[2][1]=d_dir[5]/d_spc[2];   SD[2][2]=d_dir[8]/d_spc[2];


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


                    /////////////////// Metric computation////////////////
                    float F[3][3], M[3][3],Fi[3][3], Mi[3][3];
                    F[0][0]=row_f[6*i+0]; F[0][1]=row_f[6*i+1]; F[0][2]=row_f[6*i+2];
                    F[1][0]=row_f[6*i+1]; F[1][1]=row_f[6*i+3]; F[1][2]=row_f[6*i+4];
                    F[2][0]=row_f[6*i+2]; F[2][1]=row_f[6*i+4]; F[2][2]=row_f[6*i+5];

                    M[0][0]=row_m[6*i+0]; M[0][1]=row_m[6*i+1]; M[0][2]=row_m[6*i+2];
                    M[1][0]=row_m[6*i+1]; M[1][1]=row_m[6*i+3]; M[1][2]=row_m[6*i+4];
                    M[2][0]=row_m[6*i+2]; M[2][1]=row_m[6*i+4]; M[2][2]=row_m[6*i+5];

                    float Rf[3][3]={0};
                    float RfT[3][3]={0};
                    float Rm[3][3]={0};
                    float RmT[3][3]={0};
                    float Af[3][3]={0};
                    float Am[3][3]={0};
                    float temp[3][3];

                    ComputeSingleJacobianMatrixAtIndex(def_FINV,i,j,k,Af);
                    ComputeSingleJacobianMatrixAtIndex(def_MINV,i,j,k,Am);
                    ComputeRotationMatrix(Af,Rf);
                    ComputeRotationMatrix(Am,Rm);

                    MatrixTranspose(Rf,RfT);
                    MatrixTranspose(Rm,RmT);

                    MatrixMultiply(RfT,F,temp);
                    MatrixMultiply(temp,Rf,Fi);
                    MatrixMultiply(RmT,M,temp);
                    MatrixMultiply(temp,Rm,Mi);


                    float metric_val = (Fi[0][0]-Mi[0][0])*(Fi[0][0]-Mi[0][0])+
                                     2*(Fi[0][1]-Mi[0][1])*(Fi[0][1]-Mi[0][1])+
                                     2*(Fi[0][2]-Mi[0][2])*(Fi[0][2]-Mi[0][2])+
                                       (Fi[1][1]-Mi[1][1])*(Fi[1][1]-Mi[1][1])+
                                     2*(Fi[1][2]-Mi[1][2])*(Fi[1][2]-Mi[1][2])+
                                       (Fi[2][2]-Mi[2][2])*(Fi[2][2]-Mi[2][2]);


                    metric_val=sqrt(metric_val);
                    row_metric[i]=metric_val;


                    /////////////////Gradient computation///////////////////////////

                    float gradI[3][6]={0}, gradJ[3][6]={0};
                    for(int v=0;v<6;v++)
                    {
                        float3 gradIt= ComputeImageGradient(up_img,i,j,k,v);
                        float3 gradJt= ComputeImageGradient(down_img,i,j,k,v);

                        gradI[0][v]=gradIt.x;
                        gradJ[0][v]=gradJt.x;
                        gradI[1][v]=gradIt.y;
                        gradJ[1][v]=gradJt.y;
                        gradI[2][v]=gradIt.z;
                        gradJ[2][v]=gradJt.z;
                    }

                    for(int gdim=0;gdim<3;gdim++)
                    {
                        float F2[3][3], M2[3][3],F2i[3][3], M2i[3][3];

                        /////for i
                        F2[0][0]=gradI[gdim][0]; F2[0][1]=gradI[gdim][1]; F2[0][2]=gradI[gdim][2];
                        F2[1][0]=gradI[gdim][1]; F2[1][1]=gradI[gdim][3]; F2[1][2]=gradI[gdim][4];
                        F2[2][0]=gradI[gdim][2]; F2[2][1]=gradI[gdim][4]; F2[2][2]=gradI[gdim][5];

                        M2[0][0]=gradJ[gdim][0]; M2[0][1]=gradJ[gdim][1]; M2[0][2]=gradJ[gdim][2];
                        M2[1][0]=gradJ[gdim][1]; M2[1][1]=gradJ[gdim][3]; M2[1][2]=gradJ[gdim][4];
                        M2[2][0]=gradJ[gdim][2]; M2[2][1]=gradJ[gdim][4]; M2[2][2]=gradJ[gdim][5];


                        float temp[3][3];
                        MatrixMultiply(RfT,F2,temp);
                        MatrixMultiply(temp,Rf,F2i);
                        MatrixMultiply(RmT,M2,temp);
                        MatrixMultiply(temp,Rm,M2i);



                        float smf= 2 *(
                                     (Fi[0][0]-Mi[0][0])*F2i[0][0]+
                                   2*(Fi[0][1]-Mi[0][1])*F2i[0][1]+
                                   2*(Fi[0][2]-Mi[0][2])*F2i[0][2]+
                                     (Fi[1][1]-Mi[1][1])*F2i[1][1]+
                                   2*(Fi[1][2]-Mi[1][2])*F2i[1][2]+
                                     (Fi[2][2]-Mi[2][2])*F2i[2][2]
                                     );

                        float smm= 2 *(
                                     (Fi[0][0]-Mi[0][0])*M2i[0][0]+
                                   2*(Fi[0][1]-Mi[0][1])*M2i[0][1]+
                                   2*(Fi[0][2]-Mi[0][2])*M2i[0][2]+
                                     (Fi[1][1]-Mi[1][1])*M2i[1][1]+
                                   2*(Fi[1][2]-Mi[1][2])*M2i[1][2]+
                                     (Fi[2][2]-Mi[2][2])*M2i[2][2]
                                     );


                        updateF[gdim]= smf;
                        updateM[gdim]= -smm;
                    } //for gdim
                } // at x



                ///////////////////////////for neighbors of x //////////////////////////
                for(int dim=0;dim<3;dim++)
                {
                    ////////////////////i+1////////////////////////////
                    {

                        int ii=i;
                        int jj=j;
                        int kk=k;
                        if(dim==0)
                            ii++;
                        if(dim==1)
                            jj++;
                        if(dim==2)
                            kk++;

                        size_t fpitch= up_img.pitch;
                        size_t fslicePitch= fpitch*d_sz[1]*kk;
                        size_t fcolPitch= jj*fpitch;
                        char *f_ptr= (char *)(up_img.ptr);
                        char * slice_f= f_ptr+  fslicePitch;
                        float * row_f= (float *)(slice_f+ fcolPitch);

                        size_t mpitch= down_img.pitch;
                        size_t mslicePitch= mpitch*d_sz[1]*kk;
                        size_t mcolPitch= jj*mpitch;
                        char *m_ptr= (char *)(down_img.ptr);
                        char * slice_m= m_ptr+  mslicePitch;
                        float * row_m= (float *)(slice_m+ mcolPitch);

                        float F[3][3], M[3][3];
                        F[0][0]=row_f[6*ii+0]; F[0][1]=row_f[6*ii+1]; F[0][2]=row_f[6*ii+2];
                        F[1][0]=row_f[6*ii+1]; F[1][1]=row_f[6*ii+3]; F[1][2]=row_f[6*ii+4];
                        F[2][0]=row_f[6*ii+2]; F[2][1]=row_f[6*ii+4]; F[2][2]=row_f[6*ii+5];

                        M[0][0]=row_m[6*ii+0]; M[0][1]=row_m[6*ii+1]; M[0][2]=row_m[6*ii+2];
                        M[1][0]=row_m[6*ii+1]; M[1][1]=row_m[6*ii+3]; M[1][2]=row_m[6*ii+4];
                        M[2][0]=row_m[6*ii+2]; M[2][1]=row_m[6*ii+4]; M[2][2]=row_m[6*ii+5];

                        float Rf[3][3]={0};
                        float RfT[3][3]={0};
                        float Rm[3][3]={0};
                        float RmT[3][3]={0};
                        float Af[3][3]={0};
                        float Am[3][3]={0};
                        float temp[3][3]={0};
                        float Fi[3][3], Mi[3][3];
                        ComputeSingleJacobianMatrixAtIndex(def_FINV,ii,jj,kk,Af);
                        ComputeSingleJacobianMatrixAtIndex(def_MINV,ii,jj,kk,Am);
                        ComputeRotationMatrix(Af,Rf);
                        ComputeRotationMatrix(Am,Rm);
                        MatrixTranspose(Rf,RfT);
                        MatrixTranspose(Rm,RmT);


                        MatrixMultiply(RfT,F,temp);
                        MatrixMultiply(temp,Rf,Fi);
                        MatrixMultiply(RmT,M,temp);
                        MatrixMultiply(temp,Rm,Mi);

                        {
                            for(int px=0;px<3;px++)
                            {
                                {
                                    float Rpos[3][3]={0}, Rneg[3][3]={0} ;
                                    Af[px][0]-= DPHI *SD[dim][0];
                                    Af[px][1]-= DPHI *SD[dim][1];
                                    Af[px][2]-= DPHI *SD[dim][2];
                                    ComputeRotationMatrix(Af,Rpos);
                                    Af[px][0]+= 2*DPHI *SD[dim][0];
                                    Af[px][1]+= 2*DPHI *SD[dim][1];
                                    Af[px][2]+= 2*DPHI *SD[dim][2];
                                    ComputeRotationMatrix(Af,Rneg);
                                    Af[px][0]-= DPHI *SD[dim][0];
                                    Af[px][1]-= DPHI *SD[dim][1];
                                    Af[px][2]-= DPHI *SD[dim][2];

                                    Rpos[0][0]= 0.5*(Rpos[0][0]-Rneg[0][0]);Rpos[0][1]= 0.5*(Rpos[0][1]-Rneg[0][1]);Rpos[0][2]= 0.5*(Rpos[0][2]-Rneg[0][2]);
                                    Rpos[1][0]= 0.5*(Rpos[1][0]-Rneg[1][0]);Rpos[1][1]= 0.5*(Rpos[1][1]-Rneg[1][1]);Rpos[1][2]= 0.5*(Rpos[1][2]-Rneg[1][2]);
                                    Rpos[2][0]= 0.5*(Rpos[2][0]-Rneg[2][0]);Rpos[2][1]= 0.5*(Rpos[2][1]-Rneg[2][1]);Rpos[2][2]= 0.5*(Rpos[2][2]-Rneg[2][2]);

                                    MatrixTranspose(Rpos,Rneg);

                                    float  res1[3][3];
                                    MatrixMultiply(Rneg,F,temp);
                                    MatrixMultiply(temp,Rf,res1);                                    

                                    updateF[px]+= 2*(Fi[0][0]-Mi[0][0]) * (res1[0][0]+ res1[0][0])+
                                                  4*(Fi[0][1]-Mi[0][1]) * (res1[0][1]+ res1[1][0])+
                                                  4*(Fi[0][2]-Mi[0][2]) * (res1[0][2]+ res1[2][0])+
                                                  2*(Fi[1][1]-Mi[1][1]) * (res1[1][1]+ res1[1][1])+
                                                  4*(Fi[1][2]-Mi[1][2]) * (res1[1][2]+ res1[2][1])+
                                                  2*(Fi[2][2]-Mi[2][2]) * (res1[2][2]+ res1[2][2]);
                                }
                                {
                                    float Rpos[3][3]={0}, Rneg[3][3]={0} ;
                                    Am[px][0]-= DPHI *SD[dim][0];
                                    Am[px][1]-= DPHI *SD[dim][1];
                                    Am[px][2]-= DPHI *SD[dim][2];
                                    ComputeRotationMatrix(Am,Rpos);
                                    Am[px][0]+= 2*DPHI *SD[dim][0];
                                    Am[px][1]+= 2*DPHI *SD[dim][1];
                                    Am[px][2]+= 2*DPHI *SD[dim][2];
                                    ComputeRotationMatrix(Am,Rneg);
                                    Am[px][0]-= DPHI *SD[dim][0];
                                    Am[px][1]-= DPHI *SD[dim][1];
                                    Am[px][2]-= DPHI *SD[dim][2];

                                    Rpos[0][0]= 0.5*(Rpos[0][0]-Rneg[0][0]);Rpos[0][1]= 0.5*(Rpos[0][1]-Rneg[0][1]);Rpos[0][2]= 0.5*(Rpos[0][2]-Rneg[0][2]);
                                    Rpos[1][0]= 0.5*(Rpos[1][0]-Rneg[1][0]);Rpos[1][1]= 0.5*(Rpos[1][1]-Rneg[1][1]);Rpos[1][2]= 0.5*(Rpos[1][2]-Rneg[1][2]);
                                    Rpos[2][0]= 0.5*(Rpos[2][0]-Rneg[2][0]);Rpos[2][1]= 0.5*(Rpos[2][1]-Rneg[2][1]);Rpos[2][2]= 0.5*(Rpos[2][2]-Rneg[2][2]);

                                    MatrixTranspose(Rpos,Rneg);

                                    float  res1[3][3];
                                    MatrixMultiply(Rneg,M,temp);
                                    MatrixMultiply(temp,Rm,res1);

                                    updateM[px]-= 2*(Fi[0][0]-Mi[0][0]) * (res1[0][0]+ res1[0][0])+
                                                  4*(Fi[0][1]-Mi[0][1]) * (res1[0][1]+ res1[1][0])+
                                                  4*(Fi[0][2]-Mi[0][2]) * (res1[0][2]+ res1[2][0])+
                                                  2*(Fi[1][1]-Mi[1][1]) * (res1[1][1]+ res1[1][1])+
                                                  4*(Fi[1][2]-Mi[1][2]) * (res1[1][2]+ res1[2][1])+
                                                  2*(Fi[2][2]-Mi[2][2]) * (res1[2][2]+ res1[2][2]);
                                }
                            } //for px
                        } //scope
                    } //i+1

                    ////////////////////i-1////////////////////////////
                    {
                        int ii=i;
                        int jj=j;
                        int kk=k;
                        if(dim==0)
                            ii--;
                        if(dim==1)
                            jj--;
                        if(dim==2)
                            kk--;

                        size_t fpitch= up_img.pitch;
                        size_t fslicePitch= fpitch*d_sz[1]*kk;
                        size_t fcolPitch= jj*fpitch;
                        char *f_ptr= (char *)(up_img.ptr);
                        char * slice_f= f_ptr+  fslicePitch;
                        float * row_f= (float *)(slice_f+ fcolPitch);

                        size_t mpitch= down_img.pitch;
                        size_t mslicePitch= mpitch*d_sz[1]*kk;
                        size_t mcolPitch= jj*mpitch;
                        char *m_ptr= (char *)(down_img.ptr);
                        char * slice_m= m_ptr+  mslicePitch;
                        float * row_m= (float *)(slice_m+ mcolPitch);


                        float F[3][3], M[3][3];
                        F[0][0]=row_f[6*ii+0]; F[0][1]=row_f[6*ii+1]; F[0][2]=row_f[6*ii+2];
                        F[1][0]=row_f[6*ii+1]; F[1][1]=row_f[6*ii+3]; F[1][2]=row_f[6*ii+4];
                        F[2][0]=row_f[6*ii+2]; F[2][1]=row_f[6*ii+4]; F[2][2]=row_f[6*ii+5];

                        M[0][0]=row_m[6*ii+0]; M[0][1]=row_m[6*ii+1]; M[0][2]=row_m[6*ii+2];
                        M[1][0]=row_m[6*ii+1]; M[1][1]=row_m[6*ii+3]; M[1][2]=row_m[6*ii+4];
                        M[2][0]=row_m[6*ii+2]; M[2][1]=row_m[6*ii+4]; M[2][2]=row_m[6*ii+5];

                        float Rf[3][3]={0};
                        float RfT[3][3]={0};
                        float Rm[3][3]={0};
                        float RmT[3][3]={0};
                        float Af[3][3]={0};
                        float Am[3][3]={0};
                        float Fi[3][3], Mi[3][3];
                        float temp[3][3];

                        ComputeSingleJacobianMatrixAtIndex(def_FINV,ii,jj,kk,Af);
                        ComputeSingleJacobianMatrixAtIndex(def_MINV,ii,jj,kk,Am);
                        ComputeRotationMatrix(Af,Rf);
                        ComputeRotationMatrix(Am,Rm);
                        MatrixTranspose(Rf,RfT);
                        MatrixTranspose(Rm,RmT);

                        MatrixMultiply(RfT,F,temp);
                        MatrixMultiply(temp,Rf,Fi);
                        MatrixMultiply(RmT,M,temp);
                        MatrixMultiply(temp,Rm,Mi);

                        {
                            for(int px=0;px<3;px++)
                            {
                                {
                                    float Rpos[3][3]={0}, Rneg[3][3]={0} ;
                                    Af[px][0]+= DPHI *SD[dim][0];
                                    Af[px][1]+= DPHI *SD[dim][1];
                                    Af[px][2]+= DPHI *SD[dim][2];
                                    ComputeRotationMatrix(Af,Rpos);
                                    Af[px][0]-= 2*DPHI *SD[dim][0];
                                    Af[px][1]-= 2*DPHI *SD[dim][1];
                                    Af[px][2]-= 2*DPHI *SD[dim][2];
                                    ComputeRotationMatrix(Af,Rneg);
                                    Af[px][0]+= DPHI *SD[dim][0];
                                    Af[px][1]+= DPHI *SD[dim][1];
                                    Af[px][2]+= DPHI *SD[dim][2];

                                    Rpos[0][0]= 0.5*(Rpos[0][0]-Rneg[0][0]);Rpos[0][1]= 0.5*(Rpos[0][1]-Rneg[0][1]);Rpos[0][2]= 0.5*(Rpos[0][2]-Rneg[0][2]);
                                    Rpos[1][0]= 0.5*(Rpos[1][0]-Rneg[1][0]);Rpos[1][1]= 0.5*(Rpos[1][1]-Rneg[1][1]);Rpos[1][2]= 0.5*(Rpos[1][2]-Rneg[1][2]);
                                    Rpos[2][0]= 0.5*(Rpos[2][0]-Rneg[2][0]);Rpos[2][1]= 0.5*(Rpos[2][1]-Rneg[2][1]);Rpos[2][2]= 0.5*(Rpos[2][2]-Rneg[2][2]);

                                    MatrixTranspose(Rpos,Rneg);

                                    float res1[3][3];
                                    MatrixMultiply(Rneg,F,temp);
                                    MatrixMultiply(temp,Rf,res1);


                                    updateF[px]+= 2*(Fi[0][0]-Mi[0][0]) * (res1[0][0]+ res1[0][0])+
                                                  4*(Fi[0][1]-Mi[0][1]) * (res1[0][1]+ res1[1][0])+
                                                  4*(Fi[0][2]-Mi[0][2]) * (res1[0][2]+ res1[2][0])+
                                                  2*(Fi[1][1]-Mi[1][1]) * (res1[1][1]+ res1[1][1])+
                                                  4*(Fi[1][2]-Mi[1][2]) * (res1[1][2]+ res1[2][1])+
                                                  2*(Fi[2][2]-Mi[2][2]) * (res1[2][2]+ res1[2][2]);
                                }
                                {
                                    float Rpos[3][3]={0}, Rneg[3][3]={0} ;
                                    Am[px][0]+= DPHI *SD[dim][0];
                                    Am[px][1]+= DPHI *SD[dim][1];
                                    Am[px][2]+= DPHI *SD[dim][2];
                                    ComputeRotationMatrix(Am,Rpos);
                                    Am[px][0]-= 2*DPHI *SD[dim][0];
                                    Am[px][1]-= 2*DPHI *SD[dim][1];
                                    Am[px][2]-= 2*DPHI *SD[dim][2];
                                    ComputeRotationMatrix(Am,Rneg);
                                    Am[px][0]+= DPHI *SD[dim][0];
                                    Am[px][1]+= DPHI *SD[dim][1];
                                    Am[px][2]+= DPHI *SD[dim][2];

                                    Rpos[0][0]= 0.5*(Rpos[0][0]-Rneg[0][0]);Rpos[0][1]= 0.5*(Rpos[0][1]-Rneg[0][1]);Rpos[0][2]= 0.5*(Rpos[0][2]-Rneg[0][2]);
                                    Rpos[1][0]= 0.5*(Rpos[1][0]-Rneg[1][0]);Rpos[1][1]= 0.5*(Rpos[1][1]-Rneg[1][1]);Rpos[1][2]= 0.5*(Rpos[1][2]-Rneg[1][2]);
                                    Rpos[2][0]= 0.5*(Rpos[2][0]-Rneg[2][0]);Rpos[2][1]= 0.5*(Rpos[2][1]-Rneg[2][1]);Rpos[2][2]= 0.5*(Rpos[2][2]-Rneg[2][2]);

                                    MatrixTranspose(Rpos,Rneg);

                                    float temp[3][3], res1[3][3];
                                    MatrixMultiply(Rneg,M,temp);
                                    MatrixMultiply(temp,Rm,res1);


                                    updateM[px]-= 2*(Fi[0][0]-Mi[0][0]) * (res1[0][0]+ res1[0][0])+
                                                  4*(Fi[0][1]-Mi[0][1]) * (res1[0][1]+ res1[1][0])+
                                                  4*(Fi[0][2]-Mi[0][2]) * (res1[0][2]+ res1[2][0])+
                                                  2*(Fi[1][1]-Mi[1][1]) * (res1[1][1]+ res1[1][1])+
                                                  4*(Fi[1][2]-Mi[1][2]) * (res1[1][2]+ res1[2][1])+
                                                  2*(Fi[2][2]-Mi[2][2]) * (res1[2][2]+ res1[2][2]);
                                }
                            } //for px
                        } //scope
                    } //i-1
                } //for dim                




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



void ComputeMetric_DEV_cuda(cudaPitchedPtr up_img, cudaPitchedPtr down_img,
		   int3 data_sz, float3 data_spc, 
		   float d00,float d01,float d02,float d10,float d11,float d12,float d20,float d21,float d22,
		   cudaPitchedPtr def_FINV, cudaPitchedPtr def_MINV,
                   cudaPitchedPtr updateFieldF, cudaPitchedPtr updateFieldM,
                   float &metric_value		   )
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


    ComputeMetric_DEV_kernel<<< blockSize,gridSize>>>( up_img, down_img, def_FINV,def_MINV, updateFieldF, updateFieldM, metric_image);
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
