#ifndef RIGIDREGISTERIMAGES_CXX
#define RIGIDREGISTERIMAGES_CXX

#include "rigid_register_images.h"


#include "itkEuler3DTransform.h"
#include "itkCorrelationImageToImageMetricv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkANTSNeighborhoodCorrelationImageToImageMetricv4.h"
#include "itkCenteredTransformInitializer.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkConjugateGradientLineSearchOptimizerv4.h"
#include "itkImageMomentsCalculator.h"
#include "itkMultiStartOptimizerv4.h"
#include "itkAmoebaOptimizerv4.h"


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



RigidTransformType::Pointer RigidRegisterImagesEuler(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string metric_type,float lr,bool gd, RigidTransformType::Pointer in_trans)
{
    int NITK= TORTOISE::GetAvailableITKThreadFor();

    typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType3;
    MetricType3::Pointer m= MetricType3::New();
    m->SetNumberOfHistogramBins(40);
    m->SetMaximumNumberOfWorkUnits(NITK);


    typedef itk::CorrelationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType2;
    MetricType2::Pointer m2= MetricType2::New();
    m2->SetMaximumNumberOfWorkUnits(NITK);

    typedef itk::ANTSNeighborhoodCorrelationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType4;
    MetricType4::Pointer m3= MetricType4::New();
    MetricType4::RadiusType rad; rad.Fill(4);
    m3->SetMaximumNumberOfWorkUnits(NITK);
    m3->SetRadius(rad);



    using MetricType =itk::ImageToImageMetricv4<ImageType3D,ImageType3D> ;
    MetricType::Pointer         metric        = nullptr;
    if(metric_type=="CC")
        metric=m2;
    else if(metric_type=="CC2")
        metric=m3;
    else
        metric=m;

    RigidTransformType::Pointer initial_transform = RigidTransformType::New();
    initial_transform->SetIdentity();

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


    if(in_trans==nullptr)
    {
        typedef itk::CenteredTransformInitializer<RigidTransformType, ImageType3D, ImageType3D> TransformInitializerType;
        typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();

        initializer->SetTransform( initial_transform );
        initializer->SetFixedImage( fixed_img );
        initializer->SetMovingImage( moving_img );
        //initializer->GeometryOn();
        initializer->MomentsOn();
        initializer->InitializeTransform();
    }
    else
    {
        initial_transform->SetParameters(in_trans->GetParameters());
    }


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
    smoothingSigmasPerLevel[0] = 3;
    smoothingSigmasPerLevel[1] = 2.;
    smoothingSigmasPerLevel[2] = 1.;
    smoothingSigmasPerLevel[3] = 0.25;

    std::vector<unsigned int> currentStageIterations;
    currentStageIterations.push_back(10000);
    currentStageIterations.push_back(1000);
    currentStageIterations.push_back(1000);
    currentStageIterations.push_back(100);



    rigidRegistration->SetNumberOfLevels( 4 );
    rigidRegistration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
    rigidRegistration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
    rigidRegistration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);
    rigidRegistration->SetMetricSamplingPercentage(1.);


    rigidRegistration->SetMetricSamplingStrategy(RigidRegistrationType::MetricSamplingStrategyEnum::NONE);
    rigidRegistration->SetInitialTransform(initial_transform);
    rigidRegistration->SetInPlace(true);


    float learningRate = lr;
    using ScalesEstimatorType= itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;

    ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric( metric );
    scalesEstimator->SetTransformForward( true );

    //CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();

    if(!gd)
    {
        typedef itk::ConjugateGradientLineSearchOptimizerv4Template<double> ConjugateGradientDescentOptimizerType;
        typename ConjugateGradientDescentOptimizerType::Pointer optimizer = ConjugateGradientDescentOptimizerType::New();
        optimizer->SetLowerLimit( 0 );
        optimizer->SetUpperLimit( 4 );
        optimizer->SetEpsilon( 0.15 );
        optimizer->SetLearningRate( learningRate );
        optimizer->SetMaximumStepSizeInPhysicalUnits( learningRate );
        optimizer->SetNumberOfIterations( 1000 );
        optimizer->SetScalesEstimator( scalesEstimator );
        optimizer->SetMinimumConvergenceValue( 1E-5 );
        optimizer->SetConvergenceWindowSize( 10 );
        optimizer->SetDoEstimateLearningRateAtEachIteration( true);
        optimizer->SetDoEstimateLearningRateOnce( false );

        rigidRegistration->SetOptimizer(optimizer);
    }
    else
    {
        std::cout<<"Doing gradient descent instead of conjugate..."<<std::endl;
        //typedef itk::AmoebaOptimizerv4 OptimizerType;
        typedef itk::GradientDescentOptimizerv4 OptimizerType;
        OptimizerType::Pointer optimizer = OptimizerType::New();
        optimizer->SetLearningRate( learningRate );
        optimizer->SetMaximumStepSizeInPhysicalUnits( learningRate );
        optimizer->SetNumberOfIterations( 1000 );
        optimizer->SetScalesEstimator( scalesEstimator );
        optimizer->SetMinimumConvergenceValue( 1E-5 );
        optimizer->SetConvergenceWindowSize( 10 );
        optimizer->SetDoEstimateLearningRateAtEachIteration( false);

        rigidRegistration->SetOptimizer(optimizer);

    }

    //optimizer->AddObserver(itk::IterationEvent(), observer );



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



RigidTransformType::Pointer RigidRegisterImagesEulerSmall(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string metric_type)
{
    int NITK= TORTOISE::GetAvailableITKThreadFor();

    typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType3;
    MetricType3::Pointer m= MetricType3::New();
    m->SetNumberOfHistogramBins(40);
    m->SetMaximumNumberOfWorkUnits(NITK);


    typedef itk::CorrelationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType2;
    MetricType2::Pointer m2= MetricType2::New();
    m2->SetMaximumNumberOfWorkUnits(NITK);

    typedef itk::ANTSNeighborhoodCorrelationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType4;
    MetricType4::Pointer m3= MetricType4::New();
    MetricType4::RadiusType rad; rad.Fill(4);
    m3->SetMaximumNumberOfWorkUnits(NITK);
    m3->SetRadius(rad);



    using MetricType =itk::ImageToImageMetricv4<ImageType3D,ImageType3D> ;
    MetricType::Pointer         metric        = nullptr;
    if(metric_type=="CC")
        metric=m2;
    else if(metric_type=="CC2")
        metric=m3;
    else
        metric=m;

    RigidTransformType::Pointer initial_transform = RigidTransformType::New();
    initial_transform->SetIdentity();
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

    typedef itk::CenteredTransformInitializer<RigidTransformType, ImageType3D, ImageType3D> TransformInitializerType;
    typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();

    initializer->SetTransform( initial_transform );
    initializer->SetFixedImage( fixed_img );
    initializer->SetMovingImage( moving_img );
    //initializer->GeometryOn();
    initializer->MomentsOn();
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
    shrinkFactorsPerLevel.SetSize( 2 );
    shrinkFactorsPerLevel[0] = 4;
    shrinkFactorsPerLevel[1] = 2;

    RigidRegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize( 2 );
    smoothingSigmasPerLevel[0] = 2.5;
    smoothingSigmasPerLevel[1] = 1.5;

    std::vector<unsigned int> currentStageIterations;
    currentStageIterations.push_back(20);
    currentStageIterations.push_back(10);


    rigidRegistration->SetNumberOfLevels( 2 );
    rigidRegistration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
    rigidRegistration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
    rigidRegistration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);
    rigidRegistration->SetMetricSamplingPercentage(1.);


    rigidRegistration->SetMetricSamplingStrategy(RigidRegistrationType::MetricSamplingStrategyEnum::NONE);
    rigidRegistration->SetInitialTransform(initial_transform);
    rigidRegistration->SetInPlace(true);

  //  std::cout<< initial_transform->GetParameters()<<std::endl;


    float learningRate = 0.01;
    using ScalesEstimatorType= itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;

    ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric( metric );
    scalesEstimator->SetTransformForward( true );

    //CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();


    typedef itk::GradientDescentOptimizerv4 OptimizerType;
    OptimizerType::Pointer optimizer = OptimizerType::New();

    //typedef itk::ConjugateGradientLineSearchOptimizerv4Template<double> ConjugateGradientDescentOptimizerType;
    //typename ConjugateGradientDescentOptimizerType::Pointer optimizer = ConjugateGradientDescentOptimizerType::New();
    //optimizer->SetLowerLimit( 0 );
    //optimizer->SetUpperLimit( 4 );
    //optimizer->SetEpsilon( 0.15 );
    optimizer->SetLearningRate( learningRate );
    optimizer->SetMaximumStepSizeInPhysicalUnits( learningRate );
    //optimizer->SetNumberOfIterations( 1000 );
    optimizer->SetScalesEstimator( scalesEstimator );
    optimizer->SetMinimumConvergenceValue( 1E-5 );
    optimizer->SetConvergenceWindowSize( 10 );
    //optimizer->SetDoEstimateLearningRateAtEachIteration( true);
    //optimizer->SetDoEstimateLearningRateOnce( false );
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


RigidTransformType::Pointer MultiStartRigidSearch(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string metric_type)
{
    typedef double RealType;
    typedef float  PixelType;

      /** Define All Parameters Here */
    double       pi = vnl_math::pi;                // probably a vnl alternative
    RealType     searchfactor = 22.5;                // in degrees, passed by user
    unsigned int mibins = 40;                      // for mattes MI metric
    RealType     degtorad = 0.0174532925;          // to convert degrees to radians
    RealType     localoptimizerlearningrate = 0.1; // for local search via conjgrad
    unsigned int localoptimizeriterations = 20;    // for local search via conjgrad
      // piover4 is (+/-) for cross-section of the sphere to multi-start search in increments
      // of searchfactor ( converted from degrees to radians ).
      // the search is centered +/- from the principal axis alignment of the images.
    RealType piover4 = pi / 4; // works in preliminary practical examples in 3D, in 2D use pi.

    typedef itk::ImageMomentsCalculator<ImageType3D>                 ImageCalculatorType;
    typedef itk::Vector<float, 3>                              VectorType;
    typedef itk::MultiStartOptimizerv4         OptimizerType;
    typedef  OptimizerType::ScalesType ScalesType;
    //typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType3D, ImageType3D, ImageType3D> MetricType;


    typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType3;
    typedef itk::CorrelationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType2;
    using MetricType =itk::ImageToImageMetricv4<ImageType3D,ImageType3D> ;

    typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType>        RegistrationParameterScalesFromPhysicalShiftType;
    typedef  itk::ConjugateGradientLineSearchOptimizerv4 LocalOptimizerType;





    VectorType ccg1,ccg2;


    searchfactor *= degtorad; // convert degrees to radians
    ImageType3D::Pointer image1 = fixed_img;
    ImageType3D::Pointer image2 = moving_img;

    ImageCalculatorType::Pointer calculator1 = ImageCalculatorType::New();
    ImageCalculatorType::Pointer calculator2 = ImageCalculatorType::New();
    calculator1->SetImage(  image1 );
    calculator2->SetImage(  image2 );
    ImageCalculatorType::VectorType fixed_center;
    fixed_center.Fill(0);
    ImageCalculatorType::VectorType moving_center;
    moving_center.Fill(0);
    try
    {
        calculator1->Compute();
        fixed_center = calculator1->GetCenterOfGravity();
        ccg1 = calculator1->GetCenterOfGravity();
        try
          {
          calculator2->Compute();
          moving_center = calculator2->GetCenterOfGravity();
          ccg2 = calculator2->GetCenterOfGravity();
          }
        catch( ... )
          {
          std::cerr << " zero image2 error ";
          fixed_center.Fill(0);
          }
    }
    catch( ... )
    {
        std::cerr << " zero image1 error ";
    }


    RigidTransformType::Pointer affine1 = RigidTransformType::New(); // translation to center
    RigidTransformType::OffsetType trans = affine1->GetOffset();
    itk::Point<double, 3> trans2;
    for( unsigned int i = 0; i < 3; i++ )
    {
       trans[i] = moving_center[i] - fixed_center[i];
       trans2[i] =  fixed_center[i] * ( 1 );
    }
    affine1->SetIdentity();
    affine1->SetOffset( trans );
    affine1->SetCenter( trans2 );




    RigidTransformType::Pointer affinesearch = RigidTransformType::New();
    affinesearch->SetIdentity();
    affinesearch->SetCenter( trans2 );


    OptimizerType::Pointer  mstartOptimizer = OptimizerType::New();
    MetricType::ParametersType newparams(  affine1->GetParameters() );





    MetricType3::Pointer m= MetricType3::New();
    m->SetNumberOfHistogramBins(mibins);

    MetricType2::Pointer m2= MetricType2::New();


    MetricType::Pointer         mimetric        = nullptr;
    if(metric_type=="CC")
        mimetric=m2;
    else
        mimetric=m;


    //MetricType::Pointer mimetric = MetricType::New();
    //mimetric->SetNumberOfHistogramBins( mibins );
    mimetric->SetFixedImage( image1 );
    mimetric->SetMovingImage( image2 );
    mimetric->SetMovingTransform( affinesearch );
    mimetric->SetParameters( newparams );
    mimetric->Initialize();


    RegistrationParameterScalesFromPhysicalShiftType::Pointer shiftScaleEstimator =      RegistrationParameterScalesFromPhysicalShiftType::New();
    shiftScaleEstimator->SetMetric( mimetric );
    shiftScaleEstimator->SetTransformForward( true ); // by default, scales for the moving transform
    RegistrationParameterScalesFromPhysicalShiftType::ScalesType         movingScales( affinesearch->GetNumberOfParameters() );
    shiftScaleEstimator->EstimateScales( movingScales );
    mstartOptimizer->SetScales( movingScales );
    mstartOptimizer->SetMetric( mimetric );
    OptimizerType::ParametersListType parametersList = mstartOptimizer->GetParametersList();

    affinesearch->SetComputeZYX(true);
    affinesearch->SetIdentity();
    affinesearch->SetCenter( trans2 );
    affinesearch->SetOffset( trans );
    parametersList.push_back( affinesearch->GetParameters() );
    for( double ang1 = ( piover4 * (-1) ); ang1 <= ( piover4 + searchfactor ); ang1 = ang1 + searchfactor )
    {
        for( double ang2 = ( piover4 * (-1) ); ang2 <= ( piover4 + searchfactor ); ang2 = ang2 + searchfactor )
        {
            for( double ang3 = ( piover4 * (-1) ); ang3 <= ( piover4 + searchfactor ); ang3 = ang3 + searchfactor )
            {
                affinesearch->SetIdentity();
                affinesearch->SetCenter( trans2 );
                affinesearch->SetOffset( trans );
                affinesearch->SetRotation(ang1,ang2,ang3);
                parametersList.push_back( affinesearch->GetParameters() );
            }
        }
    }
    mstartOptimizer->SetParametersList( parametersList );


    LocalOptimizerType::Pointer  localoptimizer = LocalOptimizerType::New();
    localoptimizer->SetMetric( mimetric );
    localoptimizer->SetScales( movingScales );
    localoptimizer->SetLearningRate( localoptimizerlearningrate );
    localoptimizer->SetMaximumStepSizeInPhysicalUnits( localoptimizerlearningrate ); // * sqrt( small_step )
    localoptimizer->SetNumberOfIterations( localoptimizeriterations );
    localoptimizer->SetLowerLimit( 0 );
    localoptimizer->SetUpperLimit( 2 );
    localoptimizer->SetEpsilon( 0.1 );
    localoptimizer->SetMaximumLineSearchIterations( 20 );
    localoptimizer->SetDoEstimateLearningRateOnce( true );
    localoptimizer->SetMinimumConvergenceValue( 1.e-6 );
    localoptimizer->SetConvergenceWindowSize( 3 );
    if( localoptimizeriterations > 0 )
    {
        mstartOptimizer->SetLocalOptimizer( localoptimizer );
    }
    mstartOptimizer->StartOptimization();

    RigidTransformType::Pointer bestaffine = RigidTransformType::New();
    bestaffine->SetCenter( trans2 );
    bestaffine->SetParameters( mstartOptimizer->GetBestParameters() );

    return bestaffine;

}




#endif
