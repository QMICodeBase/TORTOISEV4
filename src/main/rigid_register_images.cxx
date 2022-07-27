#ifndef RIGIDREGISTERIMAGES_CXX
#define RIGIDREGISTERIMAGES_CXX

#include "rigid_register_images.h"


#include "itkEuler3DTransform.h"
#include "itkCorrelationImageToImageMetricv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkCenteredTransformInitializer.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkConjugateGradientLineSearchOptimizerv4.h"



QuadraticTransformType::Pointer CompositeLinearToQuadratic(const CompositeTransformType * compositeTransform, std::string phase)
{
    typedef   itk::Euler3DTransform<double> RigidTransformType;
    RigidTransformType::Pointer total_rigid= RigidTransformType::New();
    total_rigid->SetIdentity();
    total_rigid->SetComputeZYX(true);

    for( unsigned int n = 0; n < compositeTransform->GetNumberOfTransforms(); n++ )
    {
        RigidTransformType::Pointer curr_trans = dynamic_cast<RigidTransformType* const>( compositeTransform->GetNthTransform( n ).GetPointer() );
        total_rigid->Compose( curr_trans, true );
    }

    QuadraticTransformType::Pointer quad_trans= QuadraticTransformType::New();

    quad_trans->SetPhase(phase);
    quad_trans->SetIdentity();

    QuadraticTransformType::ParametersType params = quad_trans->GetParameters();
    params[0]=total_rigid->GetOffset()[0];
    params[1]=total_rigid->GetOffset()[1];
    params[2]=total_rigid->GetOffset()[2];
    params[3]= total_rigid->GetAngleX();
    params[4]= total_rigid->GetAngleY();
    params[5]= total_rigid->GetAngleZ();

    quad_trans->SetParameters(params);

    return quad_trans;
}



RigidTransformType::Pointer RigidRegisterImagesEuler(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string metric_type,float lr)
{
    int NITK= TORTOISE::GetAvailableITKThreadFor();

    typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType3;
    MetricType3::Pointer m= MetricType3::New();
    m->SetNumberOfHistogramBins(50);
    m->SetMaximumNumberOfWorkUnits(NITK);   

    typedef itk::CorrelationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType2;
    MetricType2::Pointer m2= MetricType2::New();
    m2->SetMaximumNumberOfWorkUnits(NITK);        

    using MetricType =itk::ImageToImageMetricv4<ImageType3D,ImageType3D> ;
    MetricType::Pointer         metric        = nullptr;

    if(metric_type=="CC")
        metric=m2;
    else
        metric=m;

    RigidTransformType::Pointer initial_transform = RigidTransformType::New();
    initial_transform->SetIdentity();

    typedef itk::CenteredTransformInitializer<RigidTransformType, ImageType3D, ImageType3D> TransformInitializerType;
    typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();

    initializer->SetTransform( initial_transform );
    initializer->SetFixedImage( fixed_img );
    initializer->SetMovingImage( moving_img );
    initializer->GeometryOn();
    initializer->InitializeTransform();



    using RigidRegistrationType = itk::ImageRegistrationMethodv4<ImageType3D, ImageType3D, RigidTransformType> ;
    RigidRegistrationType::Pointer rigidRegistration = RigidRegistrationType::New();

    rigidRegistration->SetFixedImage( 0, fixed_img );
    rigidRegistration->SetMovingImage( 0, moving_img );
    rigidRegistration->SetMetric( metric );
    rigidRegistration->SetNumberOfWorkUnits(NITK);
    rigidRegistration->GetMultiThreader()->SetMaximumNumberOfThreads(NITK);
    rigidRegistration->GetMultiThreader()->SetNumberOfWorkUnits(NITK);


    RigidRegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize( 4 );
    shrinkFactorsPerLevel[0] = 6;
    shrinkFactorsPerLevel[1] = 4;
    shrinkFactorsPerLevel[2] = 2;
    shrinkFactorsPerLevel[3] = 1;

    RigidRegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize( 4 );
    smoothingSigmasPerLevel[0] = 1.;
    smoothingSigmasPerLevel[1] = 0.5;
    smoothingSigmasPerLevel[2] = 0.05;
    smoothingSigmasPerLevel[3] = 0.002;

    std::vector<unsigned int> currentStageIterations;
    currentStageIterations.push_back(1000);
    currentStageIterations.push_back(100);
    currentStageIterations.push_back(100);
    currentStageIterations.push_back(100);



    rigidRegistration->SetNumberOfLevels( 4 );
    rigidRegistration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
    rigidRegistration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
    rigidRegistration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);
    initial_transform->SetComputeZYX(true);



    itk::ContinuousIndex<double,3> mid_ind;
    mid_ind[0]= (fixed_img->GetLargestPossibleRegion().GetSize()[0]-1)/2.;
    mid_ind[1]= (fixed_img->GetLargestPossibleRegion().GetSize()[1]-1)/2.;
    mid_ind[2]= (fixed_img->GetLargestPossibleRegion().GetSize()[2]-1)/2.;
    ImageType3D::PointType mid_pt;
    fixed_img->TransformContinuousIndexToPhysicalPoint(mid_ind,mid_pt);


    RigidTransformType::FixedParametersType rot_center;
    rot_center.set_size(3);
    rot_center[0]=mid_pt[0];
    rot_center[1]=mid_pt[1];
    rot_center[2]=mid_pt[2];
    initial_transform->SetFixedParameters(rot_center);


    rigidRegistration->SetMetricSamplingStrategy(RigidRegistrationType::MetricSamplingStrategyEnum::NONE);
    rigidRegistration->SetInitialTransform(initial_transform);
    rigidRegistration->SetInPlace(true);


    float learningRate = lr;
    using ScalesEstimatorType= itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;

    ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric( metric );
    scalesEstimator->SetTransformForward( true );

    //CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();

    typedef itk::ConjugateGradientLineSearchOptimizerv4Template<double> ConjugateGradientDescentOptimizerType;
    typename ConjugateGradientDescentOptimizerType::Pointer optimizer = ConjugateGradientDescentOptimizerType::New();
    optimizer->SetLowerLimit( 0 );
    optimizer->SetUpperLimit( 2 );
    optimizer->SetEpsilon( 0.2 );
    optimizer->SetLearningRate( learningRate );
    optimizer->SetMaximumStepSizeInPhysicalUnits( learningRate );
    optimizer->SetNumberOfIterations( 1000 );
    optimizer->SetScalesEstimator( scalesEstimator );
    optimizer->SetMinimumConvergenceValue( 1E-5 );
    optimizer->SetConvergenceWindowSize( 10 );
    optimizer->SetDoEstimateLearningRateAtEachIteration( true);
    optimizer->SetDoEstimateLearningRateOnce( false );    
    //optimizer->AddObserver(itk::IterationEvent(), observer );

    rigidRegistration->SetOptimizer(optimizer);

    try
      {

      rigidRegistration->Update();
      }
    catch( itk::ExceptionObject & e )
      {
      std::cout << "Exception caught: " << e << std::endl;

      }

    RigidTransformType::Pointer final_trans = const_cast<RigidTransformType *>(rigidRegistration->GetOutput()->Get() );
    return final_trans;


}


QuadraticTransformType::Pointer RigidRegisterImages(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string metric_type)
{ 

    RigidTransformType::Pointer final_trans= RigidRegisterImagesEuler(fixed_img,  moving_img, metric_type);

    CompositeTransformType::Pointer initial_rigids= CompositeTransformType::New();
    initial_rigids->AddTransform( final_trans );

    QuadraticTransformType::Pointer final_quad= CompositeLinearToQuadratic(initial_rigids,"vertical");

    return final_quad;
}




#endif
