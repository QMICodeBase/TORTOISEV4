#include <boost/algorithm/string.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include "itkEuler3DTransform.hxx"


#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCompositeTransform.h"

#include"itkResampleImageFilter.h"

#include "itkImageToImageMetricv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkCenteredTransformInitializer.h"
#include "itkConjugateGradientLineSearchOptimizerv4.h"

#include "itkImageRegistrationMethodv4.h"

#include "itkImageToHistogramFilter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkTransformFileReader.h"

#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/write_3D_image_to_4D_file.h"

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

typedef itk::Vector<double,3> DisplacementType;
typedef itk::Image<DisplacementType,3> DisplacementFieldType;
typedef itk::Image<float,4> ImageType4D;
typedef itk::Image<float,3> ImageType3D;
typedef ImageType3D ImageType;

typedef vnl_matrix_fixed<double,3,3> InternalMatrixType;
using MatrixImageType = itk::Image<InternalMatrixType,3>;

typedef itk::Euler3DTransform<double> RigidTransformType;




void FillLine(DisplacementFieldType::IndexType center_index,int image_d, int disp_d, DisplacementFieldType::Pointer disp_field,MatrixImageType::Pointer L_img,ImageType3D::Pointer filled_img)
{
    DisplacementFieldType::SpacingType spc= disp_field->GetSpacing();
    DisplacementFieldType::SizeType sz = disp_field->GetLargestPossibleRegion().GetSize();

    DisplacementFieldType::IndexType curr_index=center_index;


    for(int d=center_index[image_d];d<sz[image_d]-1;d++)
    {
        curr_index[image_d]=d;

        InternalMatrixType L = L_img->GetPixel(curr_index);

        DisplacementFieldType::IndexType curr_index_p,curr_index_m;
        curr_index_p= curr_index;
        curr_index_p[image_d]= curr_index[image_d]+1;
        curr_index_m= curr_index;
        curr_index_m[image_d]= curr_index[image_d]-1;

        if(filled_img->GetPixel(curr_index_p)==1)
            continue;

        if(filled_img->GetPixel(curr_index_m)==0)
        {
            DisplacementFieldType::PixelType vec_p=disp_field->GetPixel(curr_index_p);
            DisplacementFieldType::PixelType vec = disp_field->GetPixel(curr_index);


                vec_p[disp_d] = spc[image_d]*L(disp_d,image_d) + vec[disp_d];


            disp_field->SetPixel(curr_index_p,vec_p);
        }
        else
        {
            DisplacementFieldType::PixelType vec_p=disp_field->GetPixel(curr_index_p);
            DisplacementFieldType::PixelType vec_m = disp_field->GetPixel(curr_index_m);


                vec_p[disp_d] = 2*spc[image_d]*L(disp_d,image_d) + vec_m[disp_d];            


            disp_field->SetPixel(curr_index_p,vec_p);
        }
        filled_img->SetPixel(curr_index_p,1);
    }


    curr_index=center_index;
    for(int d=center_index[image_d];d>0;d--)
    {
        curr_index[image_d]=d;

        InternalMatrixType L = L_img->GetPixel(curr_index);


        DisplacementFieldType::IndexType curr_index_p,curr_index_m;
        curr_index_p= curr_index;
        curr_index_p[image_d]= curr_index[image_d]+1;

        curr_index_m= curr_index;
        curr_index_m[image_d]= curr_index[image_d]-1;

        if(filled_img->GetPixel(curr_index_m)==1)
            continue;

        if(filled_img->GetPixel(curr_index_p)==0)
        {
            DisplacementFieldType::PixelType vec_m=disp_field->GetPixel(curr_index_m);
            DisplacementFieldType::PixelType vec = disp_field->GetPixel(curr_index);


                vec_m[disp_d] =  vec[disp_d] -spc[image_d]*L(disp_d,image_d);


            disp_field->SetPixel(curr_index_m,vec_m);
        }
        else
        {
            DisplacementFieldType::PixelType vec_m=disp_field->GetPixel(curr_index_m);
            DisplacementFieldType::PixelType vec_p = disp_field->GetPixel(curr_index_p);


                vec_m[disp_d] =  vec_p[disp_d] - 2*spc[image_d]*L(disp_d,image_d);            

            disp_field->SetPixel(curr_index_m,vec_m);
        }
        filled_img->SetPixel(curr_index_m,1);
    }
}





ImageType::Pointer PreprocessImage( ImageType::Pointer  inputImage,
                                              ImageType::PixelType lowerScaleValue,
                                              ImageType::PixelType upperScaleValue,
                                             float winsorizeLowerQuantile, float winsorizeUpperQuantile)

{
  typedef itk::Statistics::ImageToHistogramFilter<ImageType>   HistogramFilterType;
  typedef typename HistogramFilterType::InputBooleanObjectType InputBooleanObjectType;
  typedef typename HistogramFilterType::HistogramSizeType      HistogramSizeType;

  HistogramSizeType histogramSize( 1 );
  histogramSize[0] = 256;

  typename InputBooleanObjectType::Pointer autoMinMaxInputObject = InputBooleanObjectType::New();
  autoMinMaxInputObject->Set( true );

  typename HistogramFilterType::Pointer histogramFilter = HistogramFilterType::New();
  histogramFilter->SetInput( inputImage );
  histogramFilter->SetAutoMinimumMaximumInput( autoMinMaxInputObject );
  histogramFilter->SetHistogramSize( histogramSize );
  histogramFilter->SetMarginalScale( 10.0 );
  histogramFilter->Update();

  float lowerValue = histogramFilter->GetOutput()->Quantile( 0, winsorizeLowerQuantile );
  float upperValue = histogramFilter->GetOutput()->Quantile( 0, winsorizeUpperQuantile );

  typedef itk::IntensityWindowingImageFilter<ImageType, ImageType> IntensityWindowingImageFilterType;

  typename IntensityWindowingImageFilterType::Pointer windowingFilter = IntensityWindowingImageFilterType::New();
  windowingFilter->SetInput( inputImage );
  windowingFilter->SetWindowMinimum( lowerValue );
  windowingFilter->SetWindowMaximum( upperValue );
  windowingFilter->SetOutputMinimum( lowerScaleValue );
  windowingFilter->SetOutputMaximum( upperScaleValue );
  windowingFilter->Update();

  typename ImageType::Pointer outputImage = nullptr;
    outputImage = windowingFilter->GetOutput();
    outputImage->Update();
    outputImage->DisconnectPipeline();

  return outputImage;
}



RigidTransformType::Pointer RigidRegisterB0toStr(ImageType3D::Pointer str_image, ImageType3D::Pointer b0_image)
{
    itk::ContinuousIndex<double,3> mid_ind;
    mid_ind[0]= (str_image->GetLargestPossibleRegion().GetSize()[0]-1)/2.;
    mid_ind[1]= (str_image->GetLargestPossibleRegion().GetSize()[1]-1)/2.;
    mid_ind[2]= (str_image->GetLargestPossibleRegion().GetSize()[2]-1)/2.;
    ImageType3D::PointType mid_pt;
    str_image->TransformContinuousIndexToPhysicalPoint(mid_ind,mid_pt);

    using MetricType =itk::ImageToImageMetricv4<ImageType3D,ImageType3D> ;
    MetricType::Pointer         metric        =  nullptr;



    typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType3D,ImageType3D> MetricType1;
    MetricType1::Pointer m1= MetricType1::New();
    m1->SetNumberOfHistogramBins(100);
    metric=m1;




    RigidTransformType::Pointer initial_transform = RigidTransformType::New();
    initial_transform->SetIdentity();

    RigidTransformType::FixedParametersType rot_center;
    rot_center.set_size(3);
    rot_center[0]=mid_pt[0];
    rot_center[1]=mid_pt[1];
    rot_center[2]=mid_pt[2];
    initial_transform->SetFixedParameters(rot_center);
    initial_transform->SetComputeZYX(true);


    typedef itk::CenteredTransformInitializer<RigidTransformType, ImageType, ImageType> TransformInitializerType;
    typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();
    initializer->SetTransform( initial_transform );
    initializer->SetFixedImage( str_image );
    initializer->SetMovingImage( b0_image );
    initializer->GeometryOn();
    initializer->InitializeTransform();


    using RigidRegistrationType = itk::ImageRegistrationMethodv4<ImageType3D, ImageType3D, RigidTransformType> ;
    RigidRegistrationType::Pointer rigidRegistration = RigidRegistrationType::New();

    ImageType3D::Pointer curr_str_img = PreprocessImage (str_image,0,1,0,1);
    ImageType3D::Pointer curr_b0_img = PreprocessImage (b0_image,0,1,0,1);


    rigidRegistration->SetFixedImage( 0, curr_str_img );
    rigidRegistration->SetMovingImage( 0, curr_b0_img );
    rigidRegistration->SetMetric( metric );


    RigidRegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize( 5 );
    shrinkFactorsPerLevel[0] = 8;
    shrinkFactorsPerLevel[1] = 6;
    shrinkFactorsPerLevel[2] = 4;
    shrinkFactorsPerLevel[3] = 2;
    shrinkFactorsPerLevel[4] = 1;

    RigidRegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize( 5 );
    smoothingSigmasPerLevel[0] = 2.;
    smoothingSigmasPerLevel[1] = 1.;
    smoothingSigmasPerLevel[2] = 0.5;
    smoothingSigmasPerLevel[3] = 0.25;
    smoothingSigmasPerLevel[4] = 0.;

    std::vector<unsigned int> currentStageIterations;
    currentStageIterations.push_back(1000);
    currentStageIterations.push_back(100);
    currentStageIterations.push_back(100);
    currentStageIterations.push_back(100);
    currentStageIterations.push_back(100);

    rigidRegistration->SetNumberOfLevels( 5 );
    rigidRegistration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
    rigidRegistration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
    rigidRegistration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);


    rigidRegistration->SetMetricSamplingStrategy(RigidRegistrationType::NONE);
    rigidRegistration->SetInitialTransform(initial_transform);
    rigidRegistration->SetInPlace(true);


    const float learningRate = 0.25;
    using ScalesEstimatorType= itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;

    ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric( metric );
    scalesEstimator->SetTransformForward( true );


    typedef itk::ConjugateGradientLineSearchOptimizerv4Template<double> ConjugateGradientDescentOptimizerType;
    typename ConjugateGradientDescentOptimizerType::Pointer optimizer = ConjugateGradientDescentOptimizerType::New();
    optimizer->SetLowerLimit( 0 );
    optimizer->SetUpperLimit( 4 );
    optimizer->SetEpsilon( 0.1 );
    optimizer->SetLearningRate( learningRate );
    optimizer->SetMaximumStepSizeInPhysicalUnits( learningRate );
  //  optimizer->SetMaximumLineSearchIterations(100);
    optimizer->SetNumberOfIterations( 100 );
    optimizer->SetScalesEstimator( scalesEstimator );
    optimizer->SetMinimumConvergenceValue( 1E-5 );
    optimizer->SetConvergenceWindowSize( 10 );
    optimizer->SetDoEstimateLearningRateAtEachIteration( true);
    optimizer->SetDoEstimateLearningRateOnce( false );

    rigidRegistration->SetOptimizer(optimizer);

    try
      {

      rigidRegistration->Update();
      }
    catch( itk::ExceptionObject & e )
      {
      std::cout << "Exception caught: " << e << std::endl;

      }



      return const_cast<RigidTransformType *>( rigidRegistration->GetOutput()->Get() ) ;



}


int main(int argc, char *argv[])
{
    if(argc<4)
    {
        std::cout<<"Usage: ConvertLGradNonLinToDefField full_path_to_grad_dev_file diagonal_added_1  (0/1) 3D_gradwarp (0:2D, 1:3D)" <<std::endl;
        return EXIT_FAILURE;                
    }

    std::vector<ImageType3D::Pointer> grad_dev_img;
    grad_dev_img.resize(9);
    for(int t=0;t<9;t++)
        grad_dev_img[t]= read_3D_volume_from_4D(argv[1],t);


    bool added = (bool)(atoi(argv[2]));
    bool grad3d = (bool)(atoi(argv[3]));


    if(added)
    {
        itk::ImageRegionConstIteratorWithIndex<ImageType3D> it(grad_dev_img[0],grad_dev_img[0]->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            ImageType3D::IndexType ind3= it.GetIndex();

            grad_dev_img[0]->SetPixel(ind3,grad_dev_img[0]->GetPixel(ind3)-1);
            grad_dev_img[4]->SetPixel(ind3,grad_dev_img[4]->GetPixel(ind3)-1);
            grad_dev_img[8]->SetPixel(ind3,grad_dev_img[8]->GetPixel(ind3)-1);
        }
    }


    MatrixImageType::Pointer A1_physical_L_img=MatrixImageType::New();
    A1_physical_L_img->SetRegions(grad_dev_img[0]->GetLargestPossibleRegion());
    A1_physical_L_img->Allocate();
    A1_physical_L_img->SetDirection(grad_dev_img[0]->GetDirection());
    A1_physical_L_img->SetSpacing(grad_dev_img[0]->GetSpacing());
    A1_physical_L_img->SetOrigin(grad_dev_img[0]->GetOrigin());
    InternalMatrixType zm; zm.fill(0);
    A1_physical_L_img->FillBuffer(zm);



    itk::ImageRegionConstIteratorWithIndex<ImageType3D> it(grad_dev_img[0],grad_dev_img[0]->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();

        InternalMatrixType A;
        int cnt=0;
        for(int vdim=0;vdim<3;vdim++)
        {
            for(int idim=0;idim<3;idim++)
            {
                A(vdim,idim)= grad_dev_img[cnt]->GetPixel(ind3);
                cnt++;
            }
        }
        A1_physical_L_img->SetPixel(ind3,A);
    }




    DisplacementFieldType::Pointer disp_field=DisplacementFieldType::New();
    disp_field= DisplacementFieldType::New();
    disp_field->SetRegions(grad_dev_img[0]->GetLargestPossibleRegion());
    disp_field->Allocate();
    disp_field->SetDirection(grad_dev_img[0]->GetDirection());
    disp_field->SetSpacing(grad_dev_img[0]->GetSpacing());
    disp_field->SetOrigin(grad_dev_img[0]->GetOrigin());
    DisplacementFieldType::PixelType zero; zero.Fill(0);
    disp_field->FillBuffer(zero);

    ImageType3D::Pointer filled_img= ImageType3D::New();
    filled_img->SetRegions(grad_dev_img[0]->GetLargestPossibleRegion());
    filled_img->Allocate();
    filled_img->SetOrigin(grad_dev_img[0]->GetOrigin());
    filled_img->SetSpacing(grad_dev_img[0]->GetSpacing());
    filled_img->SetDirection(grad_dev_img[0]->GetDirection());
    filled_img->FillBuffer(0);


    ImageType3D::PointType zero_point; zero_point.Fill(0);
    itk::ContinuousIndex<double,3> zero_indexc;
    disp_field->TransformPhysicalPointToContinuousIndex(zero_point,zero_indexc);
    ImageType3D::IndexType zero_index;
    zero_index[0]= (int)std::round(zero_indexc[0]);
    zero_index[1]= (int)std::round(zero_indexc[1]);
    zero_index[2]= (int)std::round(zero_indexc[2]);

    ImageType3D::SizeType sz= disp_field->GetLargestPossibleRegion().GetSize();


    filled_img->SetPixel(zero_index,1);
    for(int x=zero_index[0];x<sz[0]-1;x++)
    {
        DisplacementFieldType::IndexType center_index =zero_index;
        center_index[0]=x;

        FillLine(center_index,0,0,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,1,0,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,2,0,disp_field,A1_physical_L_img,filled_img);

        for(int y=0;y<sz[1];y++)
        {
            DisplacementFieldType::IndexType curr_index =center_index;
            curr_index[1]=y;
            FillLine(curr_index,2,0,disp_field,A1_physical_L_img,filled_img);
        }
    }
    for(int x=zero_index[0]-1;x>0;x--)
    {
        DisplacementFieldType::IndexType center_index =zero_index;
        center_index[0]=x;

        FillLine(center_index,0,0,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,1,0,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,2,0,disp_field,A1_physical_L_img,filled_img);

        for(int y=0;y<sz[1];y++)
        {
            DisplacementFieldType::IndexType curr_index =center_index;
            curr_index[1]=y;
            FillLine(curr_index,2,0,disp_field,A1_physical_L_img,filled_img);
        }
    }

    filled_img->FillBuffer(0);
    filled_img->SetPixel(zero_index,1);
    for(int y=zero_index[1];y<sz[1]-1;y++)
    {
        DisplacementFieldType::IndexType center_index =zero_index;
        center_index[1]=y;

        FillLine(center_index,0,1,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,1,1,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,2,1,disp_field,A1_physical_L_img,filled_img);

        for(int x=0;x<sz[0];x++)
        {
            DisplacementFieldType::IndexType curr_index =center_index;
            curr_index[0]=x;
            FillLine(curr_index,2,1,disp_field,A1_physical_L_img,filled_img);
        }
    }
    for(int y=zero_index[1]-1;y>0;y--)
    {
        DisplacementFieldType::IndexType center_index =zero_index;
        center_index[1]=y;

        FillLine(center_index,0,1,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,1,1,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,2,1,disp_field,A1_physical_L_img,filled_img);

        for(int x=0;x<sz[0];x++)
        {
            DisplacementFieldType::IndexType curr_index =center_index;
            curr_index[0]=x;
            FillLine(curr_index,2,1,disp_field,A1_physical_L_img,filled_img);
        }
    }

    filled_img->FillBuffer(0);
    filled_img->SetPixel(zero_index,1);
    for(int y=zero_index[1];y<sz[1]-1;y++)
    {
        DisplacementFieldType::IndexType center_index =zero_index;
        center_index[1]=y;

        FillLine(center_index,0,2,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,1,2,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,2,2,disp_field,A1_physical_L_img,filled_img);

        for(int x=0;x<sz[0];x++)
        {
            DisplacementFieldType::IndexType curr_index =center_index;
            curr_index[0]=x;
            FillLine(curr_index,2,2,disp_field,A1_physical_L_img,filled_img);
        }
    }
    for(int y=zero_index[1]-1;y>0;y--)
    {
        DisplacementFieldType::IndexType center_index =zero_index;
        center_index[1]=y;

        FillLine(center_index,0,2,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,1,2,disp_field,A1_physical_L_img,filled_img);
        FillLine(center_index,2,2,disp_field,A1_physical_L_img,filled_img);

        for(int x=0;x<sz[0];x++)
        {
            DisplacementFieldType::IndexType curr_index =center_index;
            curr_index[0]=x;
            FillLine(curr_index,2,2,disp_field,A1_physical_L_img,filled_img);
        }
    }




    vnl_matrix_fixed<double,3,3> L_to_DICOM_FlipMat;L_to_DICOM_FlipMat.set_identity();
    L_to_DICOM_FlipMat(0,0)=-1;
    vnl_matrix_fixed<double,3,3> itk_to_DICOM_FlipMat;itk_to_DICOM_FlipMat.set_identity();
    itk_to_DICOM_FlipMat(0,0)=-1;
    itk_to_DICOM_FlipMat(1,1)=-1;


    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it2(disp_field,disp_field->GetLargestPossibleRegion());
    it2.GoToBegin();
    while(!it2.IsAtEnd())
    {
        vnl_vector<double> disp_vec_Lvoxelspace= it2.Get().GetVnlVector();
        vnl_vector<double> disp_vec_Lvoxelspace_flipped= L_to_DICOM_FlipMat * disp_vec_Lvoxelspace;
        vnl_vector<double> disp_vec_ITK_space=   grad_dev_img[0]->GetDirection().GetVnlMatrix() * disp_vec_Lvoxelspace_flipped;
        //vnl_vector<double> disp_vec_ITK_space_on_b0 = rigid_trans->GetMatrix().GetVnlMatrix() * disp_vec_ITK_space;
        //vnl_vector<double> trans_vec= disp_vec_ITK_space_on_b0;
        vnl_vector<double> trans_vec= disp_vec_ITK_space;


        DisplacementFieldType::PixelType disp_vec;
        if(grad3d)
        {
            disp_vec[0]=trans_vec[0];
            disp_vec[1]=trans_vec[1];
            disp_vec[2]=trans_vec[2];
        }
        else
        {
            disp_vec[0]=trans_vec[0];
            disp_vec[1]=trans_vec[1];
            disp_vec[2]=0;
        }


        it2.Set(disp_vec);

        ++it2;
    }





    std::string basename= std::string(argv[1]).substr(0,std::string(argv[1]).rfind(".nii"));
    std::string fname = basename + std::string("_field.nii");


       typedef itk::ImageFileWriter<DisplacementFieldType> WrType;              
       WrType::Pointer wr2= WrType::New();
       wr2->SetFileName(fname);
       wr2->SetInput(disp_field);
       wr2->Update();




}
