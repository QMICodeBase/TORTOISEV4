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
            it.Set(mat(0,0) + mat(1,1) + mat(2,2));
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
            it.Set(mat(0,0) + mat(1,1) + mat(2,2));
        }
    }


    AffineTransformType::Pointer transform= AffineTransformType::New();

    if(parser->getInitialRigidTransform()=="")
    {
        RigidTransformType::Pointer rigid_trans=RigidRegisterImagesEuler( fixed_TR, moving_TR, "MI", 0.5);
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
