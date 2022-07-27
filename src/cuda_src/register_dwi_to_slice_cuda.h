#ifndef _RegisterDWIToSliceCUDA_H
#define _RegisterDWIToSliceCUDA_H


#include "defines.h"
#include "cuda_image.h"
#include "itkDIFFPREPGradientDescentOptimizerv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4Okan.h"
#include "itkOkanImageRegistrationMethodv4.h"


using OkanQuadraticTransformType=itk::OkanQuadraticTransform<CoordType,3,3>;

void VolumeToSliceRegistration_cuda(ImageType3D::Pointer slice_img, ImageType3D::Pointer dwi_img , vnl_matrix<int> slspec,std::vector<float> lim_arr,std::vector<OkanQuadraticTransformType::Pointer> &s2v_transformations, bool do_eddy,std::string phase)
{     
    int Nexc= slspec.rows();
    int MB= slspec.cols();

    CUDAIMAGE::Pointer dwi_img_cuda =CUDAIMAGE::New();
    dwi_img_cuda->SetImageFromITK(dwi_img);
    if(dwi_img_cuda->getFloatdata().ptr !=nullptr)
        dwi_img_cuda->CreateTexture();

    ImageType3D::SizeType sz =slice_img->GetLargestPossibleRegion().GetSize();                    
    s2v_transformations.resize(sz[2]);


    OkanQuadraticTransformType::ParametersType flags, grd_scales;
    grd_scales.SetSize(OkanQuadraticTransformType::NQUADPARAMS);
    flags.SetSize(OkanQuadraticTransformType::NQUADPARAMS);
    flags.Fill(0);
    flags[0]=flags[1]=flags[2]=flags[3]=flags[4]=flags[5]=1;
    if(do_eddy)
    {
        flags[6]=flags[7]=flags[8]=flags[9]=flags[10]=flags[11]=1;
        flags[12]=flags[13]=1;
    }


    ImageType3D::SpacingType res = slice_img->GetSpacing();
    grd_scales[0]= res[0]*1.25;
    grd_scales[1]= res[1]*1.25;
    grd_scales[2]= res[2]*1.25;

    grd_scales[3]=0.04;
    grd_scales[4]=0.04;
    grd_scales[5]=0.04;


    grd_scales[6]= res[2]*1.5 /   ( sz[0]/2.*res[0]    )*2;
    grd_scales[7]= res[2]*1.5 /   ( sz[1]/2.*res[1]    )*2.;
    grd_scales[8]= res[2]*1.5 /   ( sz[2]/2.*res[2]    )*2.;


    grd_scales[9]=  0.5*res[2]*10. /   ( sz[0]/2.*res[0]    ) / ( sz[1]/2.*res[1]    );
    grd_scales[10]= 0.5*res[2]*10. /   ( sz[0]/2.*res[0]    ) / ( sz[2]/2.*res[2]    );
    grd_scales[11]= 0.5*res[2]*10. /   ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    );

    grd_scales[12]= res[2]*5. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
    grd_scales[13]= res[2]*8. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    )/2.;

    grd_scales[14]=  5.*res[2]*4 /   ( sz[0]/2.*res[0]    ) / ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    );

    grd_scales[15]=  5.*res[2]*1 /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
    grd_scales[16]=  5.*res[2]*1. /   ( sz[1]/2.*res[1]    ) / ( sz[1]/2.*res[1]    ) / ( sz[1]/2.*res[1]    );
    grd_scales[17]=  5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[2]/2.*res[2]    ) / ( sz[2]/2.*res[2]    );
    grd_scales[18]=  5.*res[2]*1. /   ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    ) / ( sz[2]/2.*res[2]    );
    grd_scales[19]=  5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
    grd_scales[20]=  5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );

    grd_scales=grd_scales/1.5;


    for(int e=0;e<Nexc;e++)
    {                
        ImageType3D::Pointer  temp_slice_img_itk=ImageType3D::New();

        sz[2]=MB;
        ImageType3D::IndexType start;start.Fill(0);
        ImageType3D::RegionType reg(start,sz);
        temp_slice_img_itk->SetRegions(reg);
        temp_slice_img_itk->Allocate();
        temp_slice_img_itk->FillBuffer(0);
        temp_slice_img_itk->SetDirection(slice_img->GetDirection());

        ImageType3D::SpacingType spc= slice_img->GetSpacing();
        if(MB>1)
        {
           spc[2]=(slspec(e,1)-slspec(e,0))*spc[2];
        }
        temp_slice_img_itk->SetSpacing(spc);

        ImageType3D::IndexType orig_ind;
        orig_ind[0]=0;
        orig_ind[1]=0;
        orig_ind[2]=slspec(e,0);


        ImageType3D::PointType orig;
        slice_img->TransformIndexToPhysicalPoint(orig_ind,orig);
        temp_slice_img_itk->SetOrigin(orig);
        for(int kk=0;kk<MB;kk++)
        {
            int k=slspec(e,kk);
            ImageType3D::IndexType ind3,ind3_v2;
            ind3_v2[2]=kk;
            ind3[2]=k;
            for(int j=0;j<sz[1];j++)
            {
                ind3[1]=j;
                ind3_v2[1]=j;
                for(int i=0;i<sz[0];i++)
                {
                    ind3[0]=i;
                    ind3_v2[0]=i;

                    temp_slice_img_itk->SetPixel(ind3_v2,slice_img->GetPixel(ind3));
                }
            }
        }


        typedef itk::MattesMutualInformationImageToImageMetricv4Okan<ImageType3D,ImageType3D> MetricType;
        MetricType::Pointer         metric        = MetricType::New();
        metric->SetNumberOfHistogramBins(64);
        metric->SetUseMovingImageGradientFilter(false);
        metric->SetUseFixedImageGradientFilter(false);
        metric->SetFixedMin(lim_arr[0]);
        metric->SetFixedMax(lim_arr[1]);
        metric->SetMovingMin(lim_arr[2]);
        metric->SetMovingMax(lim_arr[3]);


        CUDAIMAGE::Pointer temp_slice_img_itk_cuda= CUDAIMAGE::New();
        temp_slice_img_itk_cuda->SetImageFromITK(temp_slice_img_itk);

        OkanQuadraticTransformType::Pointer  initialTransform = OkanQuadraticTransformType::New();
        initialTransform->SetPhase(phase);
        initialTransform->SetIdentity();
        initialTransform->SetParametersForOptimizationFlags(flags);


        metric->SetMovingTransform(initialTransform);

        using OptimizerType=itk::DIFFPREPGradientDescentOptimizerv4<double>;
        OptimizerType::Pointer      optimizer     = OptimizerType::New();
        optimizer->SetOptimizationFlags(flags);
        optimizer->SetGradScales(grd_scales);
        optimizer->SetNumberHalves(5);
        optimizer->SetBrkEps(0.0005);
        optimizer->SetNBins(64);
        optimizer->SetLimits(lim_arr);
        optimizer->SetFixedCudaImage(temp_slice_img_itk_cuda);
        optimizer->SetMovingCudaImage(dwi_img_cuda);
        optimizer->SetMetric(metric);
        optimizer->StartOptimization(false);

        TransformType::ParametersType params= optimizer->GetParameters();
        initialTransform->SetParameters(params);

        for(int kk=0;kk<MB;kk++)
            s2v_transformations[ slspec(e,kk)]=initialTransform;

    } //for e in Nexc


}





#endif
