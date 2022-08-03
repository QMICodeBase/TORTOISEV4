#ifndef _GIBBSMAIN_CXX_
#define _GIBBSMAIN_CXX_


#include "defines.h"
#include "unring.h"


int main(int argc, char* argv[])
{
    if(argc<3)
    {
        std::cout<<"Usage: Gibbs input_nifti  output_nifti kspace_coverage(1,0.875,0.75) phase_encoding_dir(0: horizontal, 1:vertical) nsh(optional) minW(optional) maxW(optional)  "<<std::endl;
        return EXIT_FAILURE;
    }

    TORTOISE t;

    std::string input_name(argv[1]);
    std::string output_name(argv[2]);
    float gibbs_kspace_coverage= atof(argv[3]);
    int gibbs_nsh=25;
    int gibbs_minW=1;
    int gibbs_maxW=3;

    short phase=atoi(argv[4]);

    if(argc>5)
        gibbs_nsh=atoi(argv[5]);
    if(argc>6)
        gibbs_minW=atoi(argv[7]);
    if(argc>8)
        gibbs_maxW=atoi(argv[7]);


    float ks_cov= gibbs_kspace_coverage;

    ImageType4D::Pointer dwis= readImageD<ImageType4D>(input_name);
    if(phase==0)
    {
        ImageType4D::SizeType new_size;
        new_size[0]=dwis->GetLargestPossibleRegion().GetSize()[1];
        new_size[1]=dwis->GetLargestPossibleRegion().GetSize()[0];
        new_size[2]=dwis->GetLargestPossibleRegion().GetSize()[2];
        new_size[3]=dwis->GetLargestPossibleRegion().GetSize()[3];

        ImageType4D::IndexType start; start.Fill(0);
        ImageType4D::RegionType reg(start,new_size);

        ImageType4D::Pointer dwis2= ImageType4D::New();
        dwis2->SetRegions(reg);
        dwis2->Allocate();
        dwis2->FillBuffer(0);

        itk::ImageRegionIteratorWithIndex<ImageType4D> it(dwis2,dwis2->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            ImageType4D::IndexType ind4= it.GetIndex();
            ImageType4D::IndexType ind4_old=ind4;
            ind4_old[0]=ind4[1];
            ind4_old[1]=ind4[0];
            it.Set(dwis->GetPixel(ind4_old));
        }
        dwis=dwis2;
    }


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

    if(phase==0)
    {
        ImageType4D::SizeType new_size;
        new_size[0]=dwis->GetLargestPossibleRegion().GetSize()[1];
        new_size[1]=dwis->GetLargestPossibleRegion().GetSize()[0];
        new_size[2]=dwis->GetLargestPossibleRegion().GetSize()[2];
        new_size[3]=dwis->GetLargestPossibleRegion().GetSize()[3];

        ImageType4D::IndexType start; start.Fill(0);
        ImageType4D::RegionType reg(start,new_size);

        ImageType4D::Pointer dwis2= ImageType4D::New();
        dwis2->SetRegions(reg);
        dwis2->Allocate();
        dwis2->FillBuffer(0);

        itk::ImageRegionIteratorWithIndex<ImageType4D> it(dwis2,dwis2->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            ImageType4D::IndexType ind4= it.GetIndex();
            ImageType4D::IndexType ind4_old=ind4;
            ind4_old[0]=ind4[1];
            ind4_old[1]=ind4[0];
            it.Set(dwis->GetPixel(ind4_old));
        }
        dwis=dwis2;
    }

    writeImageD<ImageType4D>(dwis,output_name);

}


#endif
