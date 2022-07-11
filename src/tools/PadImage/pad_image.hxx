#ifndef PADIMAGE_HXX
#define PADIMAGE_HXX

#include "defines.h"

template<typename ImageType>
typename ImageType::Pointer PadImage(typename ImageType::Pointer img, int pl, int pr, int pt,int pb, int pf, int pback, bool change_header)
{


    int pad_left= pl;
    int pad_right= pr;
    int pad_top= pt;
    int pad_bottom= pb;
    int pad_low= pf;
    int pad_high= pback;

    typename ImageType::SizeType orig_size= img->GetLargestPossibleRegion().GetSize();

    typename ImageType::SizeType new_size=orig_size;
    new_size[0]=orig_size[0]+pad_left+pad_right;
    new_size[1]=orig_size[1]+pad_top+pad_bottom;
    new_size[2]=orig_size[2]+pad_low+pad_high;


    typename ImageType::Pointer new_image=ImageType::New();
    typename ImageType::IndexType start; start.Fill(0);
    typename ImageType::RegionType reg(start,new_size);
    new_image->SetRegions(reg);
    new_image->Allocate();
    new_image->SetOrigin(img->GetOrigin());
    new_image->SetSpacing(img->GetSpacing());
    new_image->SetDirection(img->GetDirection());
    new_image->FillBuffer(0.);


    itk::ImageRegionIteratorWithIndex<ImageType> it(img,img->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        typename ImageType::IndexType old_ind= it.GetIndex();
        typename ImageType::IndexType new_ind=old_ind;

        new_ind[2]= old_ind[2] + pad_low;
        new_ind[1]= old_ind[1] + pad_top;
        new_ind[0]= old_ind[0] + pad_left;
        new_image->SetPixel(new_ind,it.Get());
        ++it;
    }

    if(change_header)
    {
        itk::ContinuousIndex<double, ImageType::ImageDimension> cint;
        cint.Fill(0);
        cint[0]=-pad_left;
        cint[1]=-pad_top;
        cint[2]=-pad_low;

        typename ImageType::PointType pt;
        img->TransformContinuousIndexToPhysicalPoint(cint,pt);
        new_image->SetOrigin(pt);
    }

    return new_image;
}




template ImageType3D::Pointer PadImage<ImageType3D>(ImageType3D::Pointer, int, int , int ,int ,int ,int, bool) ;
template ImageType4D::Pointer PadImage<ImageType4D>(ImageType4D::Pointer, int, int , int ,int ,int ,int, bool) ;



#endif
