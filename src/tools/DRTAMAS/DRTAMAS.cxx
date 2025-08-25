#include "DRTAMAS.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/write_3D_image_to_4D_file.h"
#include "rigid_register_images.h"

#include "itkConjugateGradientLineSearchOptimizerv4.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkCommand.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkResampleImageFilter.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

#include "DRTAMAS_utilities_cp.h"
#include "DRTAMAS_Diffeo.h"
#include "../tools/ResampleDWIs/resample_dwis.h"

#include "itkImageToHistogramFilter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkHistogramMatchingImageFilter.h"


class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef  itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );
protected:
  CommandIterationUpdate(): m_CumulativeIterationIndex(0) {};
public:
  typedef itk::ConjugateGradientLineSearchOptimizerv4Template<double> OptimizerType;
  typedef   const OptimizerType *                            OptimizerPointer;
  void Execute(itk::Object *caller, const itk::EventObject & event) ITK_OVERRIDE
    {
    Execute( (const itk::Object *)caller, event);
    }
  void Execute(const itk::Object * object, const itk::EventObject & event) ITK_OVERRIDE
    {
    OptimizerPointer optimizer =  static_cast< OptimizerPointer >( object );
    if( optimizer == ITK_NULLPTR)
      {
      return; // in this unlikely context, just do nothing.
      }
    if( !(itk::IterationEvent().CheckEvent( &event )) )
      {
      return;
      }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << std::setprecision(5) << optimizer->GetCurrentPosition() << "  " <<      m_CumulativeIterationIndex++ << std::endl;
    }
private:
  unsigned int m_CumulativeIterationIndex;
};






void DRTAMAS::Process()
{
    Step0_ReadImages();
    if(parser->getStep()==0)
    {        
        Step0_AffineRegistration();
        Step0_TransformAndWriteAffineImage();
    }


    if(! parser->getOnlyAffine())
    {
        if(parser->getStep()<=1)
        {
            if(parser->getStep()==1)
            {
                std::string moving_tensor_fname = parser->getMovingTensor();
                std::string moving_tensor_aff_fname = moving_tensor_fname.substr(0,moving_tensor_fname.rfind(".nii"))+ "_aff.nii";
                this->moving_tensor_aff = ReadAndOrientTensor(moving_tensor_aff_fname);

                std::string moving_affine_trans_name= moving_tensor_fname.substr(0,moving_tensor_fname.rfind(".nii"))+ "_aff.txt";
                itk::TransformFileReader::Pointer reader=itk::TransformFileReader::New();
                reader->SetFileName(moving_affine_trans_name);
                reader->Update();
                typedef itk::TransformFileReader::TransformListType * TransformListType;
                TransformListType transforms = reader->GetTransformList();
                itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
                this->my_affine_trans = static_cast<AffineTransformType*>((*it).GetPointer());
            }
            Step1_DiffeoRegistration();
        }

        Step2_WriteImages();
    }
    std::cout<<"DRTAMAS completed sucessfully..."<<std::endl;

}


void DRTAMAS::Step2_WriteImages()
{
    std::string nm=parser->getMovingTensor();
    nm= nm.substr(0,nm.rfind(".nii"))+"_def_MINV.nii";
    writeImageD<DisplacementFieldType>(this->def,nm);

    DisplacementFieldTransformType::Pointer disp_trans= DisplacementFieldTransformType::New();
    disp_trans->SetDisplacementField(this->def);
    CompositeTransformType::Pointer composite_trans=    CompositeTransformType::New();
    composite_trans->AddTransform(this->my_affine_trans);
    composite_trans->AddTransform(disp_trans);

    DisplacementFieldType::Pointer comb_field=DisplacementFieldType::New();
    comb_field->SetRegions(this->def->GetLargestPossibleRegion());
    comb_field->Allocate();
    comb_field->SetSpacing(this->def->GetSpacing());
    comb_field->SetOrigin(this->def->GetOrigin());
    comb_field->SetDirection(this->def->GetDirection());


    {
        itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(comb_field,comb_field->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            ImageType3D::IndexType ind3 = it.GetIndex();
            ImageType3D::PointType pt,pt_trans;
            this->fixed_tensor->TransformIndexToPhysicalPoint(ind3,pt);
            pt_trans= composite_trans->TransformPoint(pt);

            DisplacementFieldType::PixelType vec;
            vec[0]=pt_trans[0]-pt[0];
            vec[1]=pt_trans[1]-pt[1];
            vec[2]=pt_trans[2]-pt[2];

            comb_field->SetPixel(ind3,vec);
        }
    }

    std::string moving_dt_name = parser->getMovingTensor();
    std::string output_nii_name = moving_dt_name.substr(0,moving_dt_name.rfind(".nii")) + "_diffeo.nii";
    std::string moving_comb_def_nii_name = moving_dt_name.substr(0,moving_dt_name.rfind(".nii")) + "_aff_def_MINV.nii";

    TransformAndWriteDiffeoImage(moving_tensor,comb_field, fixed_tensor, output_nii_name);
    writeImageD<DisplacementFieldType>(comb_field,moving_comb_def_nii_name);

}


void DRTAMAS::Step1_DiffeoRegistration()
{

    std::vector<ImageType3D::Pointer> moving_structurals_aff;
    moving_structurals_aff.resize(moving_structurals.size());

    for(int s=0;s<moving_structurals.size();s++)
    {
        using ResampleImageFilterType= itk::ResampleImageFilter<ImageType3D, ImageType3D> ;

        ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
        resampleFilter->SetOutputParametersFromImage(this->fixed_tensor);
        resampleFilter->SetInput(this->moving_structurals[s]);
        resampleFilter->SetTransform(this->my_affine_trans);
        resampleFilter->Update();
        ImageType3D::Pointer img= resampleFilter->GetOutput();
        moving_structurals_aff[s]=img;
    }


    DRTAMAS_Diffeo *myDRTAMAS_processor = new DRTAMAS_Diffeo;
    myDRTAMAS_processor->SetFixedTensor(this->fixed_tensor);
    myDRTAMAS_processor->SetMovingTensor(this->moving_tensor_aff);
    myDRTAMAS_processor->SetFixedStructurals(this->fixed_structurals);
    myDRTAMAS_processor->SetMovingStructurals(moving_structurals_aff);
    myDRTAMAS_processor->SetParser(parser);
    myDRTAMAS_processor->Process();

    this->def=myDRTAMAS_processor->getDef();

    delete myDRTAMAS_processor;

}



void DRTAMAS::Step0_TransformAndWriteAffineImage()
{

    std::string moving_dt_name = parser->getMovingTensor();
    std::string output_nii_name = moving_dt_name.substr(0,moving_dt_name.rfind(".nii")) + "_aff.nii";

    this->moving_tensor_aff=TransformAndWriteAffineImage(this->moving_tensor,this->my_affine_trans, this->fixed_tensor, output_nii_name);
}



void DRTAMAS::Step0_ReadImages()
{

    std::string fixed_tensor_fname = parser->getFixedTensor();
    this->fixed_tensor = ReadAndOrientTensor(fixed_tensor_fname);

    std::string moving_tensor_fname = parser->getMovingTensor();
    this->moving_tensor = ReadAndOrientTensor(moving_tensor_fname);

    int Nstr= parser->getNumberOfStructurals();
    for(int s=0;s<Nstr;s++)
    {
        ImageType3D::Pointer fimg = readImageD<ImageType3D>(parser->getFixedStructuralName(s));
        this->fixed_structurals.push_back(fimg);

        ImageType3D::Pointer mimg = readImageD<ImageType3D>(parser->getMovingStructuralName(s));
        this->moving_structurals.push_back(mimg);
    }

}


ImageType3D::Pointer DRTAMAS::PreprocessImage(ImageType3D::Pointer inputImage,
                                              ImageType3D::PixelType lowerScaleValue,
                                              ImageType3D::PixelType upperScaleValue,
                                              float winsorizeLowerQuantile, float winsorizeUpperQuantile,
                                              ImageType3D::Pointer histogramMatchSourceImage )
{
    typedef itk::Statistics::ImageToHistogramFilter<ImageType3D>   HistogramFilterType;
    typedef  HistogramFilterType::InputBooleanObjectType InputBooleanObjectType;
    typedef  HistogramFilterType::HistogramSizeType      HistogramSizeType;
    typedef  HistogramFilterType::HistogramType          HistogramType;

    HistogramSizeType histogramSize( 1 );
    histogramSize[0] = 256;

    InputBooleanObjectType::Pointer autoMinMaxInputObject = InputBooleanObjectType::New();
    autoMinMaxInputObject->Set( true );

    HistogramFilterType::Pointer histogramFilter = HistogramFilterType::New();
    histogramFilter->SetInput( inputImage );
    histogramFilter->SetAutoMinimumMaximumInput( autoMinMaxInputObject );
    histogramFilter->SetHistogramSize( histogramSize );
    histogramFilter->SetMarginalScale( 10.0 );
    histogramFilter->Update();

    float lowerValue = histogramFilter->GetOutput()->Quantile( 0, winsorizeLowerQuantile );
    float upperValue = histogramFilter->GetOutput()->Quantile( 0, winsorizeUpperQuantile );

    typedef itk::IntensityWindowingImageFilter<ImageType3D, ImageType3D> IntensityWindowingImageFilterType;

    IntensityWindowingImageFilterType::Pointer windowingFilter = IntensityWindowingImageFilterType::New();
    windowingFilter->SetInput( inputImage );
    windowingFilter->SetWindowMinimum( lowerValue );
    windowingFilter->SetWindowMaximum( upperValue );
    windowingFilter->SetOutputMinimum( lowerScaleValue );
    windowingFilter->SetOutputMaximum( upperScaleValue );
    windowingFilter->Update();

    ImageType3D::Pointer outputImage = nullptr;
    if( histogramMatchSourceImage )
    {
        typedef itk::HistogramMatchingImageFilter<ImageType3D, ImageType3D> HistogramMatchingFilterType;
        HistogramMatchingFilterType::Pointer matchingFilter = HistogramMatchingFilterType::New();
        matchingFilter->SetSourceImage( windowingFilter->GetOutput() );
        matchingFilter->SetReferenceImage( histogramMatchSourceImage );
        matchingFilter->SetNumberOfHistogramLevels( 256 );
        matchingFilter->SetNumberOfMatchPoints( 12 );
        matchingFilter->ThresholdAtMeanIntensityOn();
        matchingFilter->Update();

        outputImage = matchingFilter->GetOutput();
        outputImage->Update();
        outputImage->DisconnectPipeline();
    }
    else
    {
        outputImage = windowingFilter->GetOutput();
        outputImage->Update();
        outputImage->DisconnectPipeline();
    }
    return outputImage;
}


RigidTransformType::Pointer DRTAMAS::Step00_RigidRegistration(ImageType3D::Pointer fixed_img_orig,ImageType3D::Pointer moving_img_orig)
{
    ImageType3D::Pointer fixed_img =this->PreprocessImage(fixed_img_orig,0,1,0,1);
    ImageType3D::Pointer moving_img =this->PreprocessImage(moving_img_orig,0,1,0,1);

    writeImageD<ImageType3D>(fixed_img,"/qmi08_raid/maxif/Ruifeng_data/subj-13/subject_template_subj_13/HCP_to_subject_template/diffeo/diffeo_wrong_tensor/okan_test/aaa.nii");
    writeImageD<ImageType3D>(moving_img,"/qmi08_raid/maxif/Ruifeng_data/subj-13/subject_template_subj_13/HCP_to_subject_template/diffeo/diffeo_wrong_tensor/okan_test/bbb.nii");


    RigidTransformType::Pointer rigid_trans1= RigidRegisterImagesEuler( fixed_img,  moving_img, "CC",0.25);
    RigidTransformType::Pointer rigid_trans2= RigidRegisterImagesEuler( fixed_img,  moving_img,"MI",0.25);

    auto params1= rigid_trans1->GetParameters();
    auto params2= rigid_trans2->GetParameters();
    auto p1=params1-params2;

    double diff=0;
    diff+= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] +
            p1[3]*p1[3]/400. + p1[4]*p1[4]/400. + p1[5]*p1[5]/400. ;

    RigidTransformType::Pointer rigid_trans=nullptr;
    std::cout<<"R1: "<< params1<<std::endl;
    std::cout<<"R2: "<< params2<<std::endl;
    std::cout<<"MI vs CC diff: "<< diff<<std::endl;
    if(diff<0.005)
        rigid_trans=rigid_trans2;
    else
    {
        std::cout<<"Could not compute the rigid transformation from the structural imageto b=0 image... Starting multistart.... This could take a while"<<std::endl;
        std::cout<<"Better be safe than sorry, right?"<<std::endl;

        RigidTransformType::Pointer rigid_trans1a= RigidRegisterImagesEuler( moving_img, fixed_img,  "CC",0.25,false);
        RigidTransformType::ParametersType b1= rigid_trans1a->GetParameters();

        p1[0]= params1[0]+ b1[0];
        p1[1]= params1[1]+ b1[1];
        p1[2]= params1[2]+ b1[2];

        double diff1= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] ;
        RigidTransformType::Pointer rigid_trans2a= RigidRegisterImagesEuler( moving_img, fixed_img,  "MI",0.25,false);
        RigidTransformType::ParametersType b2= rigid_trans2a->GetParameters();

        std::cout<< "Trans CC F" << rigid_trans1->GetParameters()<<std::endl;
        std::cout<< "Trans CC B" << rigid_trans1a->GetParameters()<<std::endl;
        std::cout<< "Trans MI F" << rigid_trans2->GetParameters()<<std::endl;
        std::cout<< "Trans MI B" << rigid_trans2a->GetParameters()<<std::endl;


        p1[0]= params2[0]+ b2[0];
        p1[1]= params2[1]+ b2[1];
        p1[2]= params2[2]+ b2[2];

        double diff2= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] ;
        std::cout<< "diff1 "<<diff1 << " diff2 " <<diff2 <<std::endl;

        std::string new_metric_type="MI";
        if(diff1 < diff2)
        {
            std::cout<< "CC was determined to be more robust than MI. Switching..."<<std::endl;
            new_metric_type="CC";
        }

        if(diff1<0.001)
        {
            b1[0]= (params1[0] - b1[0])/2.;
            b1[1]= (params1[1] - b1[1])/2.;
            b1[2]= (params1[2] - b1[2])/2.;
            b1[3]= (params1[3] );
            b1[4]= (params1[4] );
            b1[5]= (params1[5] );
            rigid_trans1->SetParameters(b1);

            rigid_trans= RigidRegisterImagesEuler( fixed_img,  moving_img, "CC",0.25,true, rigid_trans1);
        }
        else
        {
            if(diff2<0.001)
            {
                b2[0]= (params2[0] - b2[0])/2.;
                b2[1]= (params2[1] - b2[1])/2.;
                b2[2]= (params2[2] - b2[2])/2.;
                b2[3]= (params2[3] );
                b2[4]= (params2[4] );
                b2[5]= (params2[5] );
                rigid_trans2->SetParameters(b2);

                rigid_trans= RigidRegisterImagesEuler( fixed_img, moving_img, "MI",0.25,true,rigid_trans2);
            }
            else
            {
                std::vector<float> new_res; new_res.resize(3);
                new_res[0]= fixed_img->GetSpacing()[0] * 2;
                new_res[1]= fixed_img->GetSpacing()[1] * 2;
                new_res[2]= fixed_img->GetSpacing()[2] * 2;
                std::vector<float> dummy;
                ImageType3D::Pointer fixed2= resample_3D_image(fixed_img,new_res,dummy,"Linear");
                new_res[0]= moving_img->GetSpacing()[0] * 2;
                new_res[1]= moving_img->GetSpacing()[1] * 2;
                new_res[2]= moving_img->GetSpacing()[2] * 2;
                ImageType3D::Pointer moving2= resample_3D_image(moving_img,new_res,dummy,"Linear");

                rigid_trans1=MultiStartRigidSearch(fixed2,  moving2,new_metric_type);
                rigid_trans= RigidRegisterImagesEuler( fixed_img,  moving_img, new_metric_type,0.25,rigid_trans1);
            }
        }

    }
    return rigid_trans;

}


void DRTAMAS::Step0_AffineRegistration()
{
    ImageType3D::Pointer fixed_TR = ImageType3D::New();
    fixed_TR ->SetRegions(fixed_tensor->GetLargestPossibleRegion());
    fixed_TR->Allocate();
    fixed_TR->SetSpacing(fixed_tensor->GetSpacing());
    fixed_TR->SetOrigin(fixed_tensor->GetOrigin());
    fixed_TR->SetDirection(fixed_tensor->GetDirection());
    {
        itk::ImageRegionIteratorWithIndex<ImageType3D> it(fixed_TR,fixed_TR->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            ImageType3D::IndexType ind3= it.GetIndex();
            auto mat = fixed_tensor->GetPixel(ind3);
            float TR=mat(0,0) + mat(1,1) + mat(2,2);
            if(TR>9000)
                TR=9000;
            it.Set(TR);
        }
    }

    ImageType3D::Pointer moving_TR = ImageType3D::New();
    moving_TR ->SetRegions(moving_tensor->GetLargestPossibleRegion());
    moving_TR->Allocate();
    moving_TR->SetSpacing(moving_tensor->GetSpacing());
    moving_TR->SetOrigin(moving_tensor->GetOrigin());
    moving_TR->SetDirection(moving_tensor->GetDirection());
    {
        itk::ImageRegionIteratorWithIndex<ImageType3D> it(moving_TR,moving_TR->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            ImageType3D::IndexType ind3= it.GetIndex();
            auto mat = moving_tensor->GetPixel(ind3);
            float TR=mat(0,0) + mat(1,1) + mat(2,2);
            if(TR>9000)
                TR=9000;
            it.Set(TR);
        }
    }


    AffineTransformType::Pointer transform= AffineTransformType::New();

    if(parser->getInitialRigidTransform()=="")
    {
        RigidTransformType::Pointer rigid_trans=Step00_RigidRegistration( fixed_TR, moving_TR);
        transform->SetTranslation(rigid_trans->GetTranslation());
        transform->SetOffset(rigid_trans->GetOffset());
        transform->SetFixedParameters(rigid_trans->GetFixedParameters());
        transform->SetMatrix(rigid_trans->GetMatrix());

    }
    else
    {

        itk::TransformFileReader::Pointer reader=itk::TransformFileReader::New();
        reader->SetFileName(parser->getInitialRigidTransform());
        reader->Update();
        typedef itk::TransformFileReader::TransformListType * TransformListType;
        TransformListType transforms = reader->GetTransformList();
        itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
        transform = static_cast<AffineTransformType*>((*it).GetPointer());
    }


    typedef itk::ConjugateGradientLineSearchOptimizerv4Template<double> OptimizerType;
    typedef itk::ImageRegistrationMethodv4<ImageType3D,ImageType3D,AffineTransformType> RegistrationType;
    typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType3D, ImageType3D, ImageType3D, double> MetricType;

    OptimizerType::Pointer      optimizer     = OptimizerType::New();
    MetricType::Pointer         metric        = MetricType::New();
    RegistrationType::Pointer   registration  = RegistrationType::New();


    registration->SetMetric(        metric        );
    registration->SetOptimizer(     optimizer     );
    metric->SetVirtualDomainFromImage( fixed_TR );
    metric->SetNumberOfHistogramBins(50 );



    registration->SetMetricSamplingStrategy(RegistrationType::MetricSamplingStrategyEnum::NONE);
    registration->SetInitialTransform(transform);
    registration->SetInPlace(true);
    registration->SetFixedImage(fixed_TR);
    registration->SetMovingImage(moving_TR);



    typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType> ScalesEstimatorType;
    typename ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric( metric );
    scalesEstimator->SetTransformForward( true );



     optimizer->SetLearningRate( 0.25 );
     optimizer->SetScalesEstimator(scalesEstimator);

     optimizer->SetMaximumStepSizeInPhysicalUnits(  0.25 );
     optimizer->SetNumberOfIterations( 200 );
     optimizer->SetLowerLimit( 0 );
     optimizer->SetUpperLimit( 2.5 );
     optimizer->SetEpsilon( 0.05 );
     optimizer->SetDoEstimateLearningRateOnce( true );
     optimizer->SetMinimumConvergenceValue( 1.e-6 );
     optimizer->SetConvergenceWindowSize( 7 );
     optimizer->SetReturnBestParametersAndValue(true);
     CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
     optimizer->AddObserver( itk::IterationEvent(), observer );


     const unsigned int numberOfLevels = 4;
     typename RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
     shrinkFactorsPerLevel.SetSize( 4 );
     shrinkFactorsPerLevel[0] = 6;
     shrinkFactorsPerLevel[1] = 4;
     shrinkFactorsPerLevel[2] = 2;
     shrinkFactorsPerLevel[3] = 1;

     typename RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
     smoothingSigmasPerLevel.SetSize( 4 );
     smoothingSigmasPerLevel[0] = 3;
     smoothingSigmasPerLevel[1] = 2;
     smoothingSigmasPerLevel[2] = 1;
     smoothingSigmasPerLevel[3] = 0.1;


     registration->SetNumberOfLevels ( numberOfLevels );
     registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
     registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);
     registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );

     try
     {
         registration->Update();
         std::cout << "Optimizer stop condition: "
               << registration->GetOptimizer()->GetStopConditionDescription()
               << std::endl;
     }
     catch( itk::ExceptionObject & err )
     {
         std::cerr << "ExceptionObject caught !" << std::endl;
         std::cerr << err << std::endl;
         return;
     }



     this->my_affine_trans=transform;

}
