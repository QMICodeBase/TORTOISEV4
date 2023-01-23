#ifndef _CONVERTEDDYTANSTOFIELD_H
#define _CONVERTEDDYTANSTOFIELD_H

#include "defines.h"
#include "TORTOISE.h"

using CompositeTransformType= TORTOISE::CompositeTransformType;
using OkanQuadraticTransformType= TORTOISE::OkanQuadraticTransformType;
using DisplacementFieldTransformType= TORTOISE::DisplacementFieldTransformType;




DisplacementFieldType::Pointer  ConvertEddyTransformToField(CompositeTransformType::Pointer eddy_trans,ImageType3D::Pointer ref_img,ImageType3D::Pointer ref_img_DP)
{
    DisplacementFieldType::Pointer field=DisplacementFieldType::New();
    field->SetRegions(ref_img->GetLargestPossibleRegion());
    field->Allocate();
    DisplacementFieldType::PixelType zero;    zero.Fill(0);
    field->FillBuffer(zero);
    field->SetDirection(ref_img->GetDirection());
    field->SetOrigin(ref_img->GetOrigin());
    field->SetSpacing(ref_img->GetSpacing());


    itk::ImageRegionIteratorWithIndex<ImageType3D> it(ref_img_DP,ref_img_DP->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        ImageType3D::PointType pt,pt_trans;
        ref_img_DP->TransformIndexToPhysicalPoint(ind3,pt);
        pt_trans= eddy_trans->TransformPoint(pt);

        itk::ContinuousIndex<double,3> cind;
        ref_img_DP->TransformPhysicalPointToContinuousIndex(pt_trans,cind);

        ref_img->TransformIndexToPhysicalPoint(ind3,pt);
        ref_img->TransformContinuousIndexToPhysicalPoint(cind,pt_trans);

        DisplacementFieldType::PixelType vec;
        vec[0]= pt_trans[0]- pt[0];
        vec[1]= pt_trans[1]- pt[1];
        vec[2]= pt_trans[2]- pt[2];
        field->SetPixel(ind3,vec);


        ++it;
    }
    return field;
}

#endif
