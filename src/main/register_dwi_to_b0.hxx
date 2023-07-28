#ifndef _RegisterDWIToB0_HXX
#define _RegisterDWIToB0_HXX

#include "defines.h"


#include "itkOkanQuadraticTransform.h"
#include "itkMattesMutualInformationImageToImageMetricv4Okan.h"
#include "itkDIFFPREPGradientDescentOptimizerv4.h"
#include "itkOkanImageRegistrationMethodv4.h"
#include "TORTOISE.h"

#include "itkImageRegistrationMethodv4.h"
using  TransformType=itk::OkanQuadraticTransform<double,3,3>;


TransformType::Pointer  RegisterDWIToB0(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string phase, MeccSettings *mecc_settings, bool initialize,std::vector<float> lim_arr, int vol,  TransformType::Pointer minit_trans=nullptr )
{    
    int NITK= TORTOISE::GetAvailableITKThreadFor();

    typedef itk::MattesMutualInformationImageToImageMetricv4Okan<ImageType3D,ImageType3D> MetricType;
    MetricType::Pointer         metric        = MetricType::New();
    metric->SetNumberOfHistogramBins(mecc_settings->getNBins());
    metric->SetUseMovingImageGradientFilter(false);
    metric->SetUseFixedImageGradientFilter(false);
    metric->SetFixedMin(lim_arr[0]);
    metric->SetFixedMax(lim_arr[1]);
    metric->SetMovingMin(lim_arr[2]);
    metric->SetMovingMax(lim_arr[3]);


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

    if( mecc_settings->getGrdSteps().size()==24)
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

        grd_scales[21]= res[0]/1.25;
        grd_scales[22]= res[1]/1.25;
        grd_scales[23]= res[2]/1.25;
    }  

    if(initialize)
    {
        MetricType::Pointer         metric2        = MetricType::New();
        metric2->SetNumberOfHistogramBins((mecc_settings->getNBins()));
        metric2->SetUseMovingImageGradientFilter(false);
        metric2->SetUseFixedImageGradientFilter(false);

        typedef itk::DIFFPREPGradientDescentOptimizerv4<double> OptimizerType;
        OptimizerType::Pointer      optimizer     = OptimizerType::New();

        TransformType::ParametersType flags2;
        flags2.SetSize(24);
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

        optimizer->SetOptimizationFlags(flags2);
        optimizer->SetGradScales(grd_scales2);
        optimizer->SetNumberHalves(mecc_settings->getNumberHalves());
        optimizer->SetBrkEps(mecc_settings->getBrkEps());


        typedef itk::OkanImageRegistrationMethodv4<ImageType3D,ImageType3D, TransformType,ImageType3D >           RegistrationType;
        RegistrationType::Pointer   registration  = RegistrationType::New();
        registration->SetFixedImage(fixed_img);
        registration->SetMovingImage(moving_img);
        registration->SetMetricSamplingPercentage(1.);        
        registration->SetOptimizer(optimizer);
        registration->SetInitialTransform(initialTransform);
        registration->InPlaceOn();        
        registration->SetNumberOfWorkUnits(NITK);
        //registration->SetNumberOfThreads(NITK);
        metric2->SetMaximumNumberOfWorkUnits(NITK);
        registration->SetMetric(        metric2        );



        RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
        shrinkFactorsPerLevel.SetSize( 1 );
        shrinkFactorsPerLevel[0] = 3;


        RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
        smoothingSigmasPerLevel.SetSize(1 );
        smoothingSigmasPerLevel[0] = 1.;


        registration->SetNumberOfLevels( 1 );
        registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
        registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);
        registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
        try
          {
          registration->Update();
          }
        catch( itk::ExceptionObject & err )
          {
          std::cerr << "ExceptionObject caught inprerigid:!" << std::endl;
          std::cerr << err << std::endl;
          std::cerr<< "In volume: "<< vol<<std::endl;
          std::cerr<< initialTransform->GetParameters()<<std::endl;
          return nullptr;
          }        
    }    

    TransformType::ParametersType finalParameters=initialTransform->GetParameters();    

        {
            using OptimizerType= itk::DIFFPREPGradientDescentOptimizerv4<double> ;
            OptimizerType::Pointer      optimizer     = OptimizerType::New();
            optimizer->SetOptimizationFlags(flags);
            optimizer->SetGradScales(grd_scales);
            optimizer->SetNumberHalves(mecc_settings->getNumberHalves());
            optimizer->SetBrkEps(mecc_settings->getBrkEps());



            typedef itk::OkanImageRegistrationMethodv4<ImageType3D,ImageType3D, TransformType,ImageType3D >           RegistrationType;
            RegistrationType::Pointer   registration  = RegistrationType::New();
            registration->SetFixedImage(fixed_img);
            registration->SetMovingImage(moving_img);
            registration->SetMetricSamplingPercentage(1.);            
            registration->SetOptimizer(optimizer);
            registration->SetInitialTransform(initialTransform);
            registration->InPlaceOn();
            registration->SetNumberOfWorkUnits(NITK);
            //registration->SetNumberOfThreads(NITK);
            metric->SetMaximumNumberOfWorkUnits(NITK);
            registration->SetMetric(        metric        );


            RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
            shrinkFactorsPerLevel.SetSize( 3 );
            shrinkFactorsPerLevel[0] = 4;
            shrinkFactorsPerLevel[1] = 2;
            shrinkFactorsPerLevel[2] = 1;

            RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
            smoothingSigmasPerLevel.SetSize( 3 );
            smoothingSigmasPerLevel[0] = 1.;
            smoothingSigmasPerLevel[1] = 0.25;
            smoothingSigmasPerLevel[2] = 0.;

            registration->SetNumberOfLevels( 3 );
            registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
            registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);
            registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
            try
              {
              registration->Update();
              }
            catch( itk::ExceptionObject & err )
              {
              std::cerr << "ExceptionObject caught !" << std::endl;
              std::cerr << err << std::endl;
              std::cerr<< "In volume: "<< vol<<std::endl;
              std::cerr<< initialTransform->GetParameters()<<std::endl;
              return nullptr;
              }            
            finalParameters =  initialTransform->GetParameters();
        }



    TransformType::Pointer finalTransform = TransformType::New();

    finalTransform->SetPhase(phase);
    finalTransform->SetIdentity();
    finalTransform->SetParametersForOptimizationFlags(flags);
    finalTransform->SetParameters( finalParameters );

    return finalTransform;
}




#endif
