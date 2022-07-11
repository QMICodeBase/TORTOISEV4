#ifndef _CONVERTEDDYTANSTOFIELD_H
#define _CONVERTEDDYTANSTOFIELD_H

#include "defines.h"
#include "TORTOISE.h"

using CompositeTransformType= TORTOISE::CompositeTransformType;
using OkanQuadraticTransformType= TORTOISE::OkanQuadraticTransformType;
using DisplacementFieldTransformType= TORTOISE::DisplacementFieldTransformType;



ImageType3D::Pointer ChangeImageHeaderToDP(ImageType3D::Pointer img)
{
    // do not want to touch the original image so we duplicate it
    using DupType= itk::ImageDuplicator<ImageType3D>;
    DupType::Pointer dup= DupType::New();
    dup->SetInputImage(img);
    dup->Update();
    ImageType3D::Pointer nimg= dup->GetOutput();


    ImageType3D::DirectionType orig_dir = img->GetDirection();
    ImageType3D::PointType orig_org = img->GetOrigin();

    ImageType3D::DirectionType id_dir;
    id_dir.SetIdentity();
    typename ImageType3D::PointType id_org;
    id_org.Fill(0);

    //Make the rotation and eddy center the image center voxel.
    id_org[0]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[0])-1)/2. * img->GetSpacing()[0];
    id_org[1]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[1])-1)/2. * img->GetSpacing()[1];
    id_org[2]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[2])-1)/2. * img->GetSpacing()[2];

    nimg->SetOrigin(id_org);
    nimg->SetDirection(id_dir);


    return nimg;
}


DisplacementFieldType::Pointer  ConvertEddyTransformToField(CompositeTransformType::Pointer eddy_trans,ImageType3D::Pointer ref_img)
{
    DisplacementFieldType::Pointer field=DisplacementFieldType::New();
    field->SetRegions(ref_img->GetLargestPossibleRegion());
    field->Allocate();
    DisplacementFieldType::PixelType zero;    zero.Fill(0);
    field->FillBuffer(zero);
    field->SetDirection(ref_img->GetDirection());
    field->SetOrigin(ref_img->GetOrigin());
    field->SetSpacing(ref_img->GetSpacing());

    ImageType3D::Pointer ref_img_DP= ChangeImageHeaderToDP(ref_img);

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
