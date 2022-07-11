#ifndef _RESAMPLE_DWIS_CXX
#define _RESAMPLE_DWIS_CXX

#include <string>
#include "../utilities/read_3Dvolume_from_4D.h"
#include "itkResampleImageFilter.h"
#include "itkIdentityTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "resample_dwis.h"

ImageType3D::Pointer resample_3D_image(ImageType3D::Pointer img,std::vector<float> new_res,std::vector<float> up_factors,std::string method)
{
    typedef itk::IdentityTransform<double, 3> TransformType;
    typedef itk::ResampleImageFilter<ImageType3D, ImageType3D> ResamplerType;


    TransformType::Pointer id_trans= TransformType::New();
    id_trans->SetIdentity();

    ResamplerType::Pointer resampler= ResamplerType::New();
    resampler->SetTransform(id_trans);
    resampler->SetInput(img);

    if(method=="NN")
    {
        typedef itk::NearestNeighborInterpolateImageFunction<ImageType3D,double> InterpolatorType;
        InterpolatorType::Pointer interp = InterpolatorType::New();
        resampler->SetInterpolator(interp);
    }

    if(method=="Linear")
    {
        typedef itk::LinearInterpolateImageFunction<ImageType3D,double> InterpolatorType;
        InterpolatorType::Pointer interp = InterpolatorType::New();
        resampler->SetInterpolator(interp);
    }

    if(method=="BSPCubic")
    {
        typedef itk::BSplineInterpolateImageFunction<ImageType3D, double, double> InterpolatorType;
        InterpolatorType::Pointer interp = InterpolatorType::New();
        interp->SetSplineOrder(3);
        resampler->SetInterpolator(interp);
    }

    resampler->SetOutputDirection(img->GetDirection());


    ImageType3D::PointType old_orig= img->GetOrigin();
    ImageType3D::IndexType one_one;
    one_one.Fill(1);
    ImageType3D::PointType oneone_pt;
    img->TransformIndexToPhysicalPoint(one_one,oneone_pt);
    ImageType3D::PointType diff=  old_orig - oneone_pt;
    ImageType3D::PointType new_orig;
    ImageType3D::SpacingType new_spc;
    ImageType3D::SizeType new_sz;

    if(up_factors.size())
    {
        new_orig[0]= old_orig[0] + diff[0]*(0.5-0.5/up_factors[0]);
        new_orig[1]= old_orig[1] + diff[1]*(0.5-0.5/up_factors[1]);
        new_orig[2]= old_orig[2] + diff[2]*(0.5-0.5/up_factors[2]);

        new_sz[0]= (int)round((double)img->GetLargestPossibleRegion().GetSize()[0] * up_factors[0]);
        new_sz[1]= (int)round((double)img->GetLargestPossibleRegion().GetSize()[1] * up_factors[1]);
        new_sz[2]= (int)round((double)img->GetLargestPossibleRegion().GetSize()[2] * up_factors[2]);

        new_spc[0]= img->GetSpacing()[0] * img->GetLargestPossibleRegion().GetSize()[0]  / (double) new_sz[0];
        new_spc[1]= img->GetSpacing()[1] * img->GetLargestPossibleRegion().GetSize()[1]  / (double) new_sz[1];
        new_spc[2]= img->GetSpacing()[2] * img->GetLargestPossibleRegion().GetSize()[2]  / (double) new_sz[2];

    }

    if(new_res.size())
    {
        new_spc[0]=new_res[0];
        new_spc[1]=new_res[1];
        new_spc[2]=new_res[2];

        new_sz[0]= (int)round((double)img->GetLargestPossibleRegion().GetSize()[0] * img->GetSpacing()[0]/new_spc[0]);
        new_sz[1]= (int)round((double)img->GetLargestPossibleRegion().GetSize()[1] * img->GetSpacing()[1]/new_spc[1]);
        new_sz[2]= (int)round((double)img->GetLargestPossibleRegion().GetSize()[2] * img->GetSpacing()[2]/new_spc[2]);

        double up_factor = img->GetSpacing()[0]/ new_spc[0];
        new_orig[0]= old_orig[0] + diff[0]*(0.5-0.5/up_factor);
        up_factor = img->GetSpacing()[1]/ new_spc[1];
        new_orig[1]= old_orig[1] + diff[1]*(0.5-0.5/up_factor);
        up_factor = img->GetSpacing()[2]/ new_spc[2];
        new_orig[2]= old_orig[2] + diff[2]*(0.5-0.5/up_factor);
    }

    resampler->SetOutputOrigin(new_orig);
    resampler->SetOutputSpacing(new_spc);
    resampler->SetSize(new_sz);
    resampler->Update();

    ImageType3D::Pointer nimg= resampler->GetOutput();
    itk::ImageRegionIteratorWithIndex<ImageType3D> it(nimg,nimg->GetLargestPossibleRegion());
    while(!it.IsAtEnd())
    {
           float val= it.Get();
           if(val<0)
               it.Set(0);
           ++it;
    }

    return nimg;
}




#endif

