#ifndef _RegisterDWIToB0CUDA_CXX
#define _RegisterDWIToB0CUDA_CXX

#include "register_dwi_to_b0_cuda.h"
#include "cuda_image.h"
#include "itkMattesMutualInformationImageToImageMetricv4Okan.h"
#include "itkOkanImageRegistrationMethodv4.h"
#include "gaussian_smooth_image.h"
#include "resample_image.h"


CUDAIMAGE::Pointer  CreateVirtualImg(CUDAIMAGE::Pointer img, int downsample_factor )
{
    CUDAIMAGE::Pointer virtual_img= CUDAIMAGE::New();

    virtual_img->dir= img->dir;
    virtual_img->spc.x = img->spc.x * downsample_factor;
    virtual_img->spc.y = img->spc.y * downsample_factor;
    virtual_img->spc.z = img->spc.z * downsample_factor;

    virtual_img->sz.x = static_cast<float>(
                std::floor( (double)img->sz.x / (double)downsample_factor ) );
    if(virtual_img->sz.x<1)
        virtual_img->sz.x=1;

    virtual_img->sz.y = static_cast<float>(
                std::floor( (double)img->sz.y / (double)downsample_factor ) );
    if(virtual_img->sz.y<1)
        virtual_img->sz.y=1;

    virtual_img->sz.z = static_cast<float>(
                std::floor( (double)img->sz.z / (double)downsample_factor ) );
    if(virtual_img->sz.z<1)
        virtual_img->sz.z=1;

    float3 ici, oci;
    ici.x= ((double)img->sz.x -1) /2.;
    ici.y= ((double)img->sz.y -1) /2.;
    ici.z= ((double)img->sz.z -1) /2.;

    oci.x= ((double)virtual_img->sz.x -1) /2.;
    oci.y= ((double)virtual_img->sz.y -1) /2.;
    oci.z= ((double)virtual_img->sz.z -1) /2.;

    float3 s1i1, s2i2, diff;
    s2i2.x = virtual_img->spc.x * oci.x;
    s2i2.y = virtual_img->spc.y * oci.y;
    s2i2.z = virtual_img->spc.z * oci.z;

    s1i1.x = img->spc.x * ici.x;
    s1i1.y = img->spc.y * ici.y;
    s1i1.z = img->spc.z * ici.z;

    diff.x= s1i1.x- s2i2.x;
    diff.y= s1i1.y- s2i2.y;
    diff.z= s1i1.z- s2i2.z;

    virtual_img->orig.x = virtual_img->dir(0,0)* diff.x + virtual_img->dir(0,1)* diff.y + virtual_img->dir(0,2)* diff.z +  (double)img->orig.x;
    virtual_img->orig.y = virtual_img->dir(1,0)* diff.x + virtual_img->dir(1,1)* diff.y + virtual_img->dir(1,2)* diff.z +  (double)img->orig.y;
    virtual_img->orig.z = virtual_img->dir(2,0)* diff.x + virtual_img->dir(2,1)* diff.y + virtual_img->dir(2,2)* diff.z +  (double)img->orig.z;

    return virtual_img;
}

TransformType::Pointer  RegisterDWIToB0_cuda(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string phase, MeccSettings *mecc_settings, bool initialize,std::vector<float> lim_arr, TransformType::Pointer minit_trans )
{    

    typedef itk::MattesMutualInformationImageToImageMetricv4Okan<ImageType3D,ImageType3D> MetricType;
    MetricType::Pointer         metric        = MetricType::New();
    metric->SetNumberOfHistogramBins(mecc_settings->getNBins());
    metric->SetUseMovingImageGradientFilter(false);
    metric->SetUseFixedImageGradientFilter(false);
    metric->SetFixedMin(lim_arr[0]);
    metric->SetFixedMax(lim_arr[1]);
    metric->SetMovingMin(lim_arr[2]);
    metric->SetMovingMax(lim_arr[3]);


    CUDAIMAGE::Pointer fixed_img_cuda= CUDAIMAGE::New();
    fixed_img_cuda->SetImageFromITK(fixed_img);
    CUDAIMAGE::Pointer moving_img_cuda= CUDAIMAGE::New();
    moving_img_cuda->SetImageFromITK(moving_img);
    

    TransformType::Pointer  initialTransform = TransformType::New();
    initialTransform->SetPhase(phase);
    initialTransform->SetIdentity();
    if(minit_trans)
        initialTransform->SetParameters(minit_trans->GetParameters());


    TransformType::ParametersType flags, grd_scales;
    flags.SetSize(TransformType::NQUADPARAMS);
    flags.Fill(0);
    grd_scales.SetSize(TransformType::NQUADPARAMS);
    for(int i=0;i<TransformType::NQUADPARAMS;i++)
    {
        flags[i]= mecc_settings->getFlags()[i];
    }
    initialTransform->SetParametersForOptimizationFlags(flags);

    TransformType::ParametersType init_params= initialTransform->GetParameters();
    init_params[0]=0;           //THIS IS NOW JUST DWI TO B=0 registration. No need to initialize. It is causing problems with spherical data
    init_params[1]=0;
    init_params[2]=0;
    init_params[21]=0;
    init_params[22]=0;
    init_params[23]=0;

    initialTransform->SetParameters(init_params);


    ImageType3D::SizeType sz= fixed_img->GetLargestPossibleRegion().GetSize();
    ImageType3D::SpacingType res = fixed_img->GetSpacing();

    int ph=0;
    if(phase=="vertical")
        ph=1;
    if(phase=="slice")
        ph=2;

    if( mecc_settings->getGrdSteps().size()==TransformType::NQUADPARAMS)
    {
        for(int i=0;i<24;i++)
            grd_scales[i]=  mecc_settings->getGrdSteps()[i];
    }
    else
    {
        grd_scales[0]= res[0]*1.25;
        grd_scales[1]= res[1]*1.25;
        grd_scales[2]= res[2]*1.25;

        grd_scales[3]=0.05;
        grd_scales[4]=0.05;
        grd_scales[5]=0.05;


        grd_scales[6]= res[2]*1.5 /   ( sz[0]/2.*res[0]    )*2;
        grd_scales[7]= res[2]*1.5 /   ( sz[1]/2.*res[1]    )*2.;
        grd_scales[8]= res[2]*1.5 /   ( sz[2]/2.*res[2]    )*2.;


        grd_scales[9]=  0.5*res[2]*10. /   ( sz[0]/2.*res[0]    ) / ( sz[1]/2.*res[1]    );
        grd_scales[10]= 0.5*res[2]*10. /   ( sz[0]/2.*res[0]    ) / ( sz[2]/2.*res[2]    );
        grd_scales[11]= 0.5*res[2]*10. /   ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    );

        grd_scales[12]= res[2]*5. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
        grd_scales[13]= res[2]*8. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    )/2.;

        grd_scales[14]= 2* 5.*res[2]*4 /   ( sz[0]/2.*res[0]    ) / ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    );

        grd_scales[15]=  2*5.*res[2]*1 /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
        grd_scales[16]= 2* 5.*res[2]*1. /   ( sz[1]/2.*res[1]    ) / ( sz[1]/2.*res[1]    ) / ( sz[1]/2.*res[1]    );
        grd_scales[17]= 2* 5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[2]/2.*res[2]    ) / ( sz[2]/2.*res[2]    );
        grd_scales[18]= 2* 5.*res[2]*1. /   ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    ) / ( sz[2]/2.*res[2]    );
        grd_scales[19]=  2*5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
        grd_scales[20]= 2* 5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );

        grd_scales[21]= res[0]*1.25;
        grd_scales[22]= res[1]*1.25;
        grd_scales[23]= res[2]*1.25;
    }  
    
    

    if(initialize)
    {
        TransformType::ParametersType flags2;
        flags2.SetSize(TransformType::NQUADPARAMS);
        flags2.Fill(0);
        flags2[0]=mecc_settings->getFlags()[0];
        flags2[1]=mecc_settings->getFlags()[1];
        flags2[2]=mecc_settings->getFlags()[2];
        flags2[3]=mecc_settings->getFlags()[3];
        flags2[4]=mecc_settings->getFlags()[4];
        flags2[5]=mecc_settings->getFlags()[5];

        TransformType::ParametersType grd_scales2= grd_scales;
        grd_scales2[3]=2*grd_scales2[3];
        grd_scales2[4]=2*grd_scales2[4];
        grd_scales2[5]=2*grd_scales2[5];

        typedef itk::OkanImageRegistrationMethodv4<ImageType3D,ImageType3D, TransformType,ImageType3D >           RegistrationType;
        RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
        shrinkFactorsPerLevel.SetSize( 1 );
        shrinkFactorsPerLevel[0] = 3;
        RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
        smoothingSigmasPerLevel.SetSize(1 );
        smoothingSigmasPerLevel[0] = 1.;

        for(int st=0;st<shrinkFactorsPerLevel.Size();st++)
        {            
            CUDAIMAGE::Pointer fixed_img_cuda_current= fixed_img_cuda;
            CUDAIMAGE::Pointer moving_img_cuda_current= moving_img_cuda;
            if(smoothingSigmasPerLevel[st]!=0)
            {
                fixed_img_cuda_current= GaussianSmoothImage(fixed_img_cuda_current,smoothingSigmasPerLevel[st]*smoothingSigmasPerLevel[st]);
                moving_img_cuda_current= GaussianSmoothImage(moving_img_cuda_current,smoothingSigmasPerLevel[st]*smoothingSigmasPerLevel[st]);
            }
            if(shrinkFactorsPerLevel[st]!=1)
            {
                CUDAIMAGE::Pointer virtual_img= CreateVirtualImg(fixed_img_cuda_current, shrinkFactorsPerLevel[st] );
                fixed_img_cuda_current=ResampleImage(fixed_img_cuda_current,virtual_img);
                virtual_img= CreateVirtualImg(moving_img_cuda_current,shrinkFactorsPerLevel[st]);
                moving_img_cuda_current=ResampleImage(moving_img_cuda_current,virtual_img);                
            }
            if(moving_img_cuda_current->getFloatdata().ptr !=nullptr)
                moving_img_cuda_current->CreateTexture();            
            
            metric->SetMovingTransform(initialTransform);

            typedef itk::DIFFPREPGradientDescentOptimizerv4<double> OptimizerType;
            OptimizerType::Pointer      optimizer     = OptimizerType::New();
            optimizer->SetOptimizationFlags(flags2);
            optimizer->SetGradScales(grd_scales2);
            optimizer->SetNumberHalves(mecc_settings->getNumberHalves());
            optimizer->SetBrkEps(mecc_settings->getBrkEps());
            optimizer->SetNBins(mecc_settings->getNBins());
            optimizer->SetLimits(lim_arr);
            optimizer->SetFixedCudaImage(fixed_img_cuda_current);
            optimizer->SetMovingCudaImage(moving_img_cuda_current);                   
            optimizer->SetMetric(metric);
            optimizer->StartOptimization(false);


            TransformType::ParametersType params= optimizer->GetParameters();
            initialTransform->SetParameters(params);


        }       
    }


    TransformType::ParametersType finalParameters=initialTransform->GetParameters();

        {
            using OptimizerType= itk::DIFFPREPGradientDescentOptimizerv4<double> ;
            typedef itk::OkanImageRegistrationMethodv4<ImageType3D,ImageType3D, TransformType,ImageType3D >           RegistrationType;

        RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
            shrinkFactorsPerLevel.SetSize( 3 );
            shrinkFactorsPerLevel[0] = 4;
            shrinkFactorsPerLevel[1] = 2;
            shrinkFactorsPerLevel[2] = 1;

            RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
            smoothingSigmasPerLevel.SetSize(3 );
            smoothingSigmasPerLevel[0] = 1.;
            smoothingSigmasPerLevel[1] = 0.25;
            smoothingSigmasPerLevel[2] = 0.;

            for(int st=0;st<shrinkFactorsPerLevel.Size();st++)
            {
                CUDAIMAGE::Pointer fixed_img_cuda_current= fixed_img_cuda;
                CUDAIMAGE::Pointer moving_img_cuda_current= moving_img_cuda;
                if(smoothingSigmasPerLevel[st]!=0)
                {
                    fixed_img_cuda_current= GaussianSmoothImage(fixed_img_cuda_current,smoothingSigmasPerLevel[st]*smoothingSigmasPerLevel[st]);
                    moving_img_cuda_current= GaussianSmoothImage(moving_img_cuda_current,smoothingSigmasPerLevel[st]*smoothingSigmasPerLevel[st]);
                }
                if(shrinkFactorsPerLevel[st]!=1)
                {
                    CUDAIMAGE::Pointer virtual_img= CreateVirtualImg(fixed_img_cuda_current,shrinkFactorsPerLevel[st]);
                    fixed_img_cuda_current=ResampleImage(fixed_img_cuda_current,virtual_img);
                    virtual_img= CreateVirtualImg(moving_img_cuda_current,shrinkFactorsPerLevel[st]);
                    moving_img_cuda_current=ResampleImage(moving_img_cuda_current,virtual_img);
                }
                if(moving_img_cuda_current->getFloatdata().ptr !=nullptr)
                    moving_img_cuda_current->CreateTexture();

                metric->SetMovingTransform(initialTransform);

                typedef itk::DIFFPREPGradientDescentOptimizerv4<double> OptimizerType;
                OptimizerType::Pointer      optimizer     = OptimizerType::New();
                optimizer->SetOptimizationFlags(flags);
                optimizer->SetGradScales(grd_scales);
                optimizer->SetNumberHalves(mecc_settings->getNumberHalves());
                optimizer->SetBrkEps(mecc_settings->getBrkEps());
                optimizer->SetNBins(mecc_settings->getNBins());
                optimizer->SetLimits(lim_arr);
                optimizer->SetFixedCudaImage(fixed_img_cuda_current);
                optimizer->SetMovingCudaImage(moving_img_cuda_current);                
                optimizer->SetMetric(metric);
                optimizer->StartOptimization(false);

                TransformType::ParametersType params= optimizer->GetParameters();
                initialTransform->SetParameters(params);
            }
        }



    TransformType::Pointer finalTransform = TransformType::New();

    finalTransform->SetPhase(phase);
    finalTransform->SetIdentity();
    finalTransform->SetParametersForOptimizationFlags(flags);
    finalTransform->SetParameters( initialTransform->GetParameters() );

    return finalTransform;
}




#endif
