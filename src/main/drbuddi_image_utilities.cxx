#ifndef _DRBUDDIIMAGEUTILITIES_CXX
#define _DRBUDDIIMAGEUTILITIES_CXX

#include "drbuddi_image_utilities.h"

#include "itkImageToHistogramFilter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkDisplacementFieldTransform.h"
#include "itkResampleImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkGaussianOperator.h"
#include "itkVectorNeighborhoodOperatorImageFilter.h"
#include "itkComposeDisplacementFieldsImageFilter.h"
#include "itkInvertDisplacementFieldImageFilterOkan.h"
#include "itkDiscreteGaussianImageFilter.h"


ImageType3D::Pointer PreprocessImage( ImageType3D::Pointer  inputImage,ImageType3D::PixelType lowerScaleValue,ImageType3D::PixelType upperScaleValue)
{

    ImageType3D::Pointer output_img=ImageType3D::New();
    output_img->SetRegions(inputImage->GetLargestPossibleRegion());
    output_img->Allocate();
    output_img->SetSpacing(inputImage->GetSpacing());
    output_img->SetOrigin(inputImage->GetOrigin());
    output_img->SetDirection(inputImage->GetDirection());

    double img_max= - 1E100;
    double img_min= 1E100;

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(inputImage,inputImage->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        if(it.Get()<0)
            it.Set(0);

        if(it.Get()>img_max)
            img_max=it.Get();

        if(it.Get()<img_min)
            img_min=it.Get();

        ++it;
    }


    double scl= (upperScaleValue -lowerScaleValue)/(img_max-img_min);
    double added=  - img_min*scl + lowerScaleValue;

    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind3= it.GetIndex();

        float val=  scl* it.Get() + added;
        output_img->SetPixel(ind3,val);

        ++it;
    }
    return output_img;





    /*
  typedef itk::Statistics::ImageToHistogramFilter<ImageType3D>   HistogramFilterType;
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

  float lowerValue = histogramFilter->GetOutput()->Quantile( 0, 0);
  float upperValue = histogramFilter->GetOutput()->Quantile( 0, 1);

  typedef itk::IntensityWindowingImageFilter<ImageType3D, ImageType3D> IntensityWindowingImageFilterType;

  typename IntensityWindowingImageFilterType::Pointer windowingFilter = IntensityWindowingImageFilterType::New();
  windowingFilter->SetInput( inputImage );
  windowingFilter->SetWindowMinimum( lowerValue );
  windowingFilter->SetWindowMaximum( upperValue );
  windowingFilter->SetOutputMinimum( lowerScaleValue );
  windowingFilter->SetOutputMaximum( upperScaleValue );
  windowingFilter->Update();

  typename ImageType3D::Pointer outputImage = nullptr;
    outputImage = windowingFilter->GetOutput();
    outputImage->Update();
    outputImage->DisconnectPipeline();

  return outputImage;
  */
}

ImageType3D::Pointer WarpImage(ImageType3D::Pointer img, DisplacementFieldType::Pointer field)
{
    DisplacementFieldTransformType::Pointer trans= DisplacementFieldTransformType::New();
    trans->SetDisplacementField(field);

    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
    resampleFilter2->SetOutputParametersFromImage(img);
    resampleFilter2->SetInput(img);
    resampleFilter2->SetTransform(trans);
    resampleFilter2->SetNumberOfWorkUnits(TORTOISE::GetNAvailableCores());
    resampleFilter2->Update();
    return resampleFilter2->GetOutput();
}


DisplacementFieldType::PixelType ComputeImageGradient(ImageType3D::Pointer img,  ImageType3D::IndexType &index)
{
    DisplacementFieldType::PixelType grad,grad2;
    grad[0]=0;
    grad[1]=0;
    grad[2]=0;

    if(index[0]==0 || index[0]== img->GetLargestPossibleRegion().GetSize()[0]-1 || index[1]==0 || index[1]== img->GetLargestPossibleRegion().GetSize()[1]-1 || index[2]==0 || index[2]== img->GetLargestPossibleRegion().GetSize()[2]-1 )
        return grad;

    const ImageType3D::SpacingType &spc=img->GetSpacing();
    const ImageType3D::DirectionType &dir=img->GetDirection();

    ImageType3D::IndexType nIndex=index;
        
    for(int d=0;d<3;d++)
    {
        nIndex[d]++;
        grad[d]= img->GetPixel(nIndex);
        nIndex[d]-=2;
        grad[d]-= img->GetPixel(nIndex);
        grad[d]*= 0.5/spc[d];
        nIndex[d]++;
    }

    grad2[0]= dir(0,0)*grad[0] + dir(0,1)*grad[1] + dir(0,2)*grad[2];
    grad2[1]= dir(1,0)*grad[0] + dir(1,1)*grad[1] + dir(1,2)*grad[2];
    grad2[2]= dir(2,0)*grad[0] + dir(2,1)*grad[1] + dir(2,2)*grad[2];
    return grad2; 
}

std::vector<ImageType3D::Pointer> ComputeImageGradientImg(ImageType3D::Pointer img)
{
    std::vector<ImageType3D::Pointer> grad_imgs; grad_imgs.resize(3);

    for(int i=0;i<3;i++)
    {
        grad_imgs[i] = ImageType3D::New();
        grad_imgs[i]->SetRegions(img->GetLargestPossibleRegion());
        grad_imgs[i]->Allocate();
        grad_imgs[i]->SetSpacing(img->GetSpacing());
        grad_imgs[i]->SetDirection(img->GetDirection());
        grad_imgs[i]->SetOrigin(img->GetOrigin());
    }



    itk::ImageRegionIteratorWithIndex<ImageType3D> it(img,img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        DisplacementFieldType::PixelType grad = ComputeImageGradient(img,ind3);
        grad_imgs[0]->SetPixel(ind3,grad[0]);
        grad_imgs[1]->SetPixel(ind3,grad[1]);
        grad_imgs[2]->SetPixel(ind3,grad[2]);
    }
    return grad_imgs;
}


void AddToUpdateField(DisplacementFieldType::Pointer updateField,DisplacementFieldType::Pointer  updateField_temp,double weight)
{       
    DisplacementFieldType::SizeType sz= updateField_temp->GetLargestPossibleRegion().GetSize();

    std::vector<double> mags;
    mags.resize(sz[2]);

    #pragma omp parallel for
    for(int k=0;k<sz[2];k++)
    {
        mags[k]=0;
        ImageType3D::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;
                DisplacementFieldType::PixelType vec= updateField_temp->GetPixel(ind3);
                mags[k]+= vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2];
            }
        }
    }
    double magnitude=0;
    for(int k=0;k<sz[2];k++)
        magnitude+=mags[k];
    magnitude= sqrt(magnitude);

    #pragma omp parallel for
    for(int k=0;k<sz[2];k++)
    {
        ImageType3D::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;
                updateField->SetPixel(ind3, updateField->GetPixel(ind3)+ weight/magnitude* updateField_temp->GetPixel(ind3) );
            }
        }
    }
}

void ScaleUpdateField(DisplacementFieldType::Pointer  field,float scale_factor)
{        
    DisplacementFieldType::SpacingType spc = field->GetSpacing();
    DisplacementFieldType::SizeType sz= field->GetLargestPossibleRegion().GetSize();


    std::vector<double> mags;
    mags.resize(sz[2]);

    #pragma omp parallel for
    for(int k=0;k<sz[2];k++)
    {
        mags[k]=-1;
        ImageType3D::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;
                DisplacementFieldType::PixelType vec= field->GetPixel(ind3);
                double nrm = (vec[0]/spc[0])*(vec[0]/spc[0]) + (vec[1]/spc[1])*(vec[1]/spc[1]) + (vec[2]/spc[2])*(vec[2]/spc[2]) ;
                nrm=sqrt(nrm);
                if(nrm>mags[k])
                    mags[k]=nrm;
            }
        }
    }
    double mxnrm=-1;
    for(int k=0;k<sz[2];k++)
        if(mags[k]>mxnrm)
            mxnrm=mags[k];


    #pragma omp parallel for
    for(int k=0;k<sz[2];k++)
    {
        mags[k]=-1;
        ImageType3D::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;
                field->SetPixel(ind3,field->GetPixel(ind3)*scale_factor/mxnrm);
            }
        }
    }
}

void RestrictPhase(DisplacementFieldType::Pointer  field, vnl_vector<double> phase_vector)
{
    DisplacementFieldType::SizeType sz= field->GetLargestPossibleRegion().GetSize();
    #pragma omp parallel for
    for(int k=0;k<sz[2];k++)
    {
        ImageType3D::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;
                DisplacementFieldType::PixelType vec=field->GetPixel(ind3);
                double nrm= sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
                if(nrm !=0)
                {
                    vec/=nrm;
                    double dot= vec[0]*phase_vector[0] + vec[1]*phase_vector[1] +vec[2]*phase_vector[2];
                    vec[0]=  phase_vector[0]*nrm*dot;
                    vec[1]=  phase_vector[1]*nrm*dot;
                    vec[2]=  phase_vector[2]*nrm*dot;
                    field->SetPixel(ind3,vec);
                }
            }
        }
    }
}

DisplacementFieldType::Pointer NegateField( const DisplacementFieldType::Pointer field)
{
    DisplacementFieldType::Pointer nfield = DisplacementFieldType::New();
    nfield->SetRegions(field->GetLargestPossibleRegion());
    nfield->Allocate();
    nfield->SetSpacing(field->GetSpacing());
    nfield->SetDirection(field->GetDirection());
    nfield->SetOrigin(field->GetOrigin());

    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(nfield,nfield->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        DisplacementFieldType::IndexType ind3= it.GetIndex();
        it.Set(-1.* field->GetPixel(ind3));
    }
    return nfield;
}

DisplacementFieldType::Pointer InvertField( const DisplacementFieldType * field, const DisplacementFieldType * inverseFieldEstimate )
{
    using InverterType = itk::InvertDisplacementFieldImageFilterOkan<DisplacementFieldType>;

    typename InverterType::Pointer inverter = InverterType::New();
    inverter->SetInput( field );
    inverter->SetInverseFieldInitialEstimate( inverseFieldEstimate );
    inverter->SetMaximumNumberOfIterations( 20 );
    inverter->SetMeanErrorToleranceThreshold( 0.001 );
    inverter->SetMaxErrorToleranceThreshold( 0.1 );
    inverter->SetNumberOfWorkUnits(TORTOISE::GetNAvailableCores());
    inverter->Update();

    DisplacementFieldType::Pointer inverseField = inverter->GetOutput();

    return inverseField;
}

void ContrainDefFields(DisplacementFieldType::Pointer  field1, DisplacementFieldType::Pointer  field2)
{
    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> ItF( field1, field1->GetLargestPossibleRegion() );
    for( ItF.GoToBegin(); !ItF.IsAtEnd(); ++ItF )
    {
        ItF.Set( (ItF.Get() - field2->GetPixel( ItF.GetIndex() ))*0.5 );
        field2->SetPixel( ItF.GetIndex(), -ItF.Get() );
    }
}

DisplacementFieldType::Pointer ComposeFields(DisplacementFieldType::Pointer  field,DisplacementFieldType::Pointer  updateField)
{   
    using ComposerType = itk::ComposeDisplacementFieldsImageFilter<DisplacementFieldType>;

    typename ComposerType::Pointer fixedComposer = ComposerType::New();
    fixedComposer->SetDisplacementField( updateField);
    fixedComposer->SetWarpingField(field );
    fixedComposer->SetNumberOfWorkUnits(TORTOISE::GetNAvailableCores());
    fixedComposer->Update();
    return fixedComposer->GetOutput();
}


ImageType3D::Pointer GaussianSmoothImage(ImageType3D::Pointer img,double variance)
{
    if(variance==0)
        return img;

    if(img==nullptr)
        return nullptr;

    using SmootherType= itk::DiscreteGaussianImageFilter<ImageType3D, ImageType3D> ;
    SmootherType::Pointer smoother = SmootherType::New();
    smoother->SetUseImageSpacing(false);
    smoother->SetVariance(  variance );
    smoother->SetMaximumError( 0.01 );
    smoother->SetInput( img );
    smoother->SetMaximumKernelWidth(32);
    smoother->SetNumberOfWorkUnits(TORTOISE::GetNAvailableCores());
    smoother->Update();
    return smoother->GetOutput();
}


DisplacementFieldType::Pointer GaussianSmoothImage(DisplacementFieldType::Pointer field,double variance)
{

    if(variance==0)
        return field;

    if(field==nullptr)
        return nullptr;


 using DuplicatorType = itk::ImageDuplicator<DisplacementFieldType>;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage( field );
  duplicator->Update();

  DisplacementFieldType::Pointer smoothField = duplicator->GetOutput();

  if( variance <= 0.0 )
    {
    return smoothField;
    }

  using GaussianSmoothingOperatorType = itk::GaussianOperator<double, 3>;
  GaussianSmoothingOperatorType gaussianSmoothingOperator;

  using GaussianSmoothingSmootherType = itk::VectorNeighborhoodOperatorImageFilter<DisplacementFieldType, DisplacementFieldType>;
  typename GaussianSmoothingSmootherType::Pointer smoother = GaussianSmoothingSmootherType::New();

  for( int d = 0; d < 3; d++ )
    {
    // smooth along this dimension
    gaussianSmoothingOperator.SetDirection( d );
    gaussianSmoothingOperator.SetVariance( variance );
    gaussianSmoothingOperator.SetMaximumError( 0.001 );
//    gaussianSmoothingOperator.SetMaximumKernelWidth( smoothField->GetRequestedRegion().GetSize()[d] );
    gaussianSmoothingOperator.SetMaximumKernelWidth( 32 );
    gaussianSmoothingOperator.CreateDirectional();



    // todo: make sure we only smooth within the buffered region
    smoother->SetOperator( gaussianSmoothingOperator );
    smoother->SetInput( smoothField );
    smoother->SetNumberOfWorkUnits(TORTOISE::GetNAvailableCores());
    smoother->Update();
    smoothField = smoother->GetOutput();
    smoothField->Update();
    smoothField->DisconnectPipeline();
  }

  const DisplacementFieldType::PixelType  zeroVector( 0.0 );

  //make sure boundary does not move
  double weight1 = 1.0;
  if( variance < 0.5 )
    {
    weight1 = 1.0 - 1.0 * ( variance / 0.5 );
    }
  double weight2 = 1.0 - weight1;

  const typename DisplacementFieldType::RegionType region = field->GetLargestPossibleRegion();
  const typename DisplacementFieldType::SizeType size = region.GetSize();
  const typename DisplacementFieldType::IndexType startIndex = region.GetIndex();

  itk::ImageRegionConstIteratorWithIndex<DisplacementFieldType> ItF( field, field->GetLargestPossibleRegion() );
  itk::ImageRegionIteratorWithIndex<DisplacementFieldType> ItS( smoothField, smoothField->GetLargestPossibleRegion() );
  for( ItF.GoToBegin(), ItS.GoToBegin(); !ItF.IsAtEnd(); ++ItF, ++ItS )
    {
    typename DisplacementFieldType::IndexType index = ItF.GetIndex();
    bool isOnBoundary = false;
    for ( unsigned int d = 0; d <3; d++ )
      {
      if( index[d] == startIndex[d] || index[d] == static_cast<ImageType3D::IndexValueType>( size[d] ) - startIndex[d] - 1 )
        {
        isOnBoundary = true;
        break;
        }
      }
    if( isOnBoundary )
      {
      ItS.Set( zeroVector );
      }
    else
      {
      ItS.Set( ItS.Get() * weight1 + ItF.Get() * weight2 );
      }
    }

  return smoothField;

}

DisplacementFieldType::Pointer ResampleImage(DisplacementFieldType::Pointer field, ImageType3D::Pointer ref_img)
{
    if(field==nullptr)
        return nullptr;

    itk::IdentityTransform<double,3>::Pointer id_trans= itk::IdentityTransform<double,3>::New();
    id_trans->SetIdentity();

    using ResampleImageFilterType= itk::ResampleImageFilter<DisplacementFieldType, DisplacementFieldType> ;
    ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
    resampleFilter->SetOutputParametersFromImage(ref_img);
    resampleFilter->SetInput(field);
    resampleFilter->SetTransform(id_trans);
    resampleFilter->SetNumberOfWorkUnits(TORTOISE::GetNAvailableCores());
    resampleFilter->Update();    
    auto resampled_img=resampleFilter->GetOutput();
    return resampled_img;
}

ImageType3D::Pointer ResampleImage(ImageType3D::Pointer img, ImageType3D::Pointer ref_img)
{
    if(img==nullptr)
        return nullptr;

    itk::IdentityTransform<double,3>::Pointer id_trans= itk::IdentityTransform<double,3>::New();
    id_trans->SetIdentity();

    using ResampleImageFilterType= itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
    resampleFilter->SetOutputParametersFromImage(ref_img);
    resampleFilter->SetInput(img);
    resampleFilter->SetTransform(id_trans);
    resampleFilter->SetDefaultPixelValue(0);
    resampleFilter->SetNumberOfWorkUnits(TORTOISE::GetNAvailableCores());
    resampleFilter->Update();
    auto resampled_img=resampleFilter->GetOutput();
    return resampled_img;

}



#endif
