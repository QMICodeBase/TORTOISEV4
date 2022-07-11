#ifndef _GIBBSMAIN_CXX_
#define _GIBBSMAIN_CXX_


#include "defines.h"
#include "unring.h"


int main(int argc, char* argv[])
{
    if(argc<3)
    {
        std::cout<<"Usage: Gibbs input_nifti  output_nifti kspace_coverage(1,0.875,0.75) nsh(optional) minW(optional) maxW"<<std::endl;
        return EXIT_FAILURE;
    }

    TORTOISE t;

    std::string input_name(argv[1]);
    std::string output_name(argv[2]);
    float gibbs_kspace_coverage= atof(argv[3]);
    int gibbs_nsh=25;
    int gibbs_minW=1;
    int gibbs_maxW=3;

    if(argc>4)
        gibbs_nsh=atoi(argv[4]);
    if(argc>5)
        gibbs_minW=atoi(argv[5]);
    if(argc>6)
        gibbs_maxW=atoi(argv[6]);


    float ks_cov= gibbs_kspace_coverage;

    ImageType4D::Pointer dwis= readImageD<ImageType4D>(input_name);

    if(ks_cov>=0.9375)
    {
        std::cout<<"Gibbs correction with full k-space coverage"<<std::endl;
        dwis=UnRingFull(dwis,gibbs_nsh,gibbs_minW,gibbs_maxW);
    }
    else
    {
        if(ks_cov<0.9375 && ks_cov >=0.8125)
        {
            std::cout<<"Gibbs correction with 7/8 k-space coverage"<<std::endl;
            dwis=UnRing78(dwis,gibbs_nsh,gibbs_minW,gibbs_maxW);
        }
        else
        {
            if(ks_cov>0.65)
            {
                std::cout<<"Gibbs correction with 6/8 k-space coverage"<<std::endl;
                dwis=UnRing68(dwis,gibbs_nsh,gibbs_minW,gibbs_maxW);
            }
            else
            {
                std::cout<<"K-space coverage in the data is less than 65\%. Skipping Gibbs ringing correction. "<<std::endl;
            }
        }
    }
    writeImageD<ImageType4D>(dwis,output_name);

}


#endif
