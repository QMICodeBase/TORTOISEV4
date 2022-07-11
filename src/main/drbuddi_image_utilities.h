#ifndef _DRBUDDIIMAGEUTILITIES_H
#define _DRBUDDIIMAGEUTILITIES_H

#include "defines.h"
#include "TORTOISE.h"


using DisplacementFieldTransformType= TORTOISE::DisplacementFieldTransformType;

ImageType3D::Pointer PreprocessImage( ImageType3D::Pointer  inputImage,ImageType3D::PixelType lowerScaleValue,ImageType3D::PixelType upperScaleValue);

ImageType3D::Pointer WarpImage(ImageType3D::Pointer img, DisplacementFieldType::Pointer field);

DisplacementFieldType::PixelType ComputeImageGradient(ImageType3D::Pointer img,ImageType3D::IndexType &index);

void AddToUpdateField(DisplacementFieldType::Pointer updateField,DisplacementFieldType::Pointer  updateField_temp,double weight);

void ScaleUpdateField(DisplacementFieldType::Pointer  field,float scale_factor);

void RestrictPhase(DisplacementFieldType::Pointer  field,vnl_vector<double> phase_vector);

DisplacementFieldType::Pointer InvertField( const DisplacementFieldType * field, const DisplacementFieldType * inverseFieldEstimate=nullptr );

void ContrainDefFields(DisplacementFieldType::Pointer  field1, DisplacementFieldType::Pointer  field2);

DisplacementFieldType::Pointer ComposeFields(DisplacementFieldType::Pointer  field,DisplacementFieldType::Pointer  updateField);

DisplacementFieldType::Pointer GaussianSmoothImage(DisplacementFieldType::Pointer field,double variance);
ImageType3D::Pointer GaussianSmoothImage(ImageType3D::Pointer img,double variance);

DisplacementFieldType::Pointer ResampleImage(DisplacementFieldType::Pointer field, ImageType3D::Pointer ref_img);
ImageType3D::Pointer ResampleImage(ImageType3D::Pointer img, ImageType3D::Pointer ref_img);


#endif
