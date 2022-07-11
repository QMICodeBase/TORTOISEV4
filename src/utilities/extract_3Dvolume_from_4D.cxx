#ifndef _EXTRACT3DVOLUMEFROM4D_CXX
#define _EXTRACT3DVOLUMEFROM4D_CXX


#include "defines.h"


ImageType3D::Pointer extract_3D_volume_from_4D(ImageType4D::Pointer img4d, int vol_id)
{
    ImageType3D::SizeType sz;
    sz[0]=img4d->GetLargestPossibleRegion().GetSize()[0];
    sz[1]=img4d->GetLargestPossibleRegion().GetSize()[1];
    sz[2]=img4d->GetLargestPossibleRegion().GetSize()[2];
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,sz);

    ImageType3D::SpacingType spc;
    spc[0]=img4d->GetSpacing()[0];
    spc[1]=img4d->GetSpacing()[1];
    spc[2]=img4d->GetSpacing()[2];

    ImageType3D::PointType orig;
    orig[0]=img4d->GetOrigin()[0];
    orig[1]=img4d->GetOrigin()[1];
    orig[2]=img4d->GetOrigin()[2];

    ImageType3D::DirectionType dir;
    dir(0,0)=img4d->GetDirection()(0,0);dir(0,1)=img4d->GetDirection()(0,1);dir(0,2)=img4d->GetDirection()(0,2);
    dir(1,0)=img4d->GetDirection()(1,0);dir(1,1)=img4d->GetDirection()(1,1);dir(1,2)=img4d->GetDirection()(1,2);
    dir(2,0)=img4d->GetDirection()(2,0);dir(2,1)=img4d->GetDirection()(2,1);dir(2,2)=img4d->GetDirection()(2,2);


    ImageType3D::Pointer img3d = ImageType3D::New();
    img3d->SetRegions(reg);
    img3d->Allocate();
    img3d->SetOrigin(orig);
    img3d->SetSpacing(spc);
    img3d->SetDirection(dir);

    ImageType4D::IndexType ind4;
    ind4[3]=vol_id;

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(img3d,img3d->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        ind4[2]=ind3[2];
        ind4[1]=ind3[1];
        ind4[0]=ind3[0];

        it.Set(img4d->GetPixel(ind4));
        ++it;
    }

    return img3d;
}





#endif
