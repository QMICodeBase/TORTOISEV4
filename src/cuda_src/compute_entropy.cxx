#ifndef _COMPUTEENTROPY_CXX
#define _COMPUTEENTROPY_CXX

#include "compute_entropy.h"
#include "gpu_capture.h"




float  ComputeEntropy(CUDAIMAGE::Pointer img, int Nbins, float low_lim, float high_lim)
{
    float value;
    gpucap::Rec cap("ComputeEntropy");
    cap.param("Nbins",(double)Nbins).param("low_lim",low_lim).param("high_lim",high_lim).in("img",img);
    ComputeEntropy_cuda(img->getFloatdata(),
                        img->sz,
                        Nbins,
                        low_lim,high_lim   ,
                        value );
    cap.scalar("entropy",value).save();
    return value;
}



void ComputeJointEntropy(CUDAIMAGE::Pointer img1, float low_lim1, float high_lim1, CUDAIMAGE::Pointer img2, float low_lim2, float high_lim2, int Nbins,float &entropy_j,float &entropy_img1,float &entropy_img2)
{
    float valuec, value1, value2;
    gpucap::Rec cap("ComputeJointEntropy");
    if(cap.on())
    {
        cap.param("Nbins",(double)Nbins);
        cap.param("lims",std::vector<double>{low_lim1,high_lim1,low_lim2,high_lim2});
        cap.in("img1",img1).in("img2",img2);
    }
    ComputeJointEntropy_cuda( img1->getFloatdata(), low_lim1,  high_lim1,
                              img2->getFloatdata(),  low_lim2,  high_lim2,
                              img1->sz,
                              Nbins,
                              valuec, value1, value2);
    entropy_j=valuec;
    entropy_img1=value1;
    entropy_img2=value2;
    if(cap.on())
    {
        cap.scalar("entropy_joint",valuec).scalar("entropy_img1",value1).scalar("entropy_img2",value2);
        cap.save();
    }

}



#endif
