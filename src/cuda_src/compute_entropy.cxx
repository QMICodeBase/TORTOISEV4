#ifndef _COMPUTEENTROPY_CXX
#define _COMPUTEENTROPY_CXX

#include "compute_entropy.h"




float  ComputeEntropy(CUDAIMAGE::Pointer img, int Nbins, float low_lim, float high_lim)
{
    float value;        
    ComputeEntropy_cuda(img->getFloatdata(),
                        img->sz,
                        Nbins,
                        low_lim,high_lim   ,
                        value );       
    return value;
}



void ComputeJointEntropy(CUDAIMAGE::Pointer img1, float low_lim1, float high_lim1, CUDAIMAGE::Pointer img2, float low_lim2, float high_lim2, int Nbins,float &entropy_j,float &entropy_img1,float &entropy_img2)
{
    float valuec, value1, value2;
    ComputeJointEntropy_cuda( img1->getFloatdata(), low_lim1,  high_lim1,
                              img2->getFloatdata(),  low_lim2,  high_lim2,
                              img1->sz,
                              Nbins,
                              valuec, value1, value2);
    entropy_j=valuec;
    entropy_img1=value1;
    entropy_img2=value2;

}



#endif
