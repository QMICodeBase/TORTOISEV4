#ifndef _UNOBLIQUEIMAGE_CXX
#define _UNOBLIQUEIMAGE_CXX


#include "defines.h"
#include "../utilities/math_utilities.h"

#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"


ImageType3D::Pointer UnObliqueImage(ImageType3D::Pointer img, bool keep_orig)
{
    ImageType3D::SizeType sz= img->GetLargestPossibleRegion().GetSize();
    ImageType3D::DirectionType orig_dir = img->GetDirection();
    ImageType3D::SpacingType spc = img->GetSpacing();
    ImageType3D::PointType orig = img->GetOrigin();

    ImageType3D::DirectionType new_dir;
    ImageType3D::SizeType new_sz;
    ImageType3D::SpacingType new_spc=spc;
    ImageType3D::PointType new_orig ;

    for(int r=0;r<3;r++)
    {
        for(int c=0;c<3;c++)
        {
            if(fabs(orig_dir(r,c))>0.5)
                new_dir(r,c)  =  sgn<double>(orig_dir(r,c));
            else
                new_dir(r,c)  =0;
        }
    }

    if(keep_orig)
    {
        itk::ContinuousIndex<double,3> mid_ind;
        mid_ind[0]= 1.*(sz[0]-1)/2;
        mid_ind[1]= 1.*(sz[1]-1)/2;
        mid_ind[2]= 1.*(sz[2]-1)/2;

        ImageType3D::PointType mid_pt;
        img->TransformContinuousIndexToPhysicalPoint(mid_ind,mid_pt);

        vnl_matrix_fixed<double,3,3> S; S.fill(0);
        S(0,0)= img->GetSpacing()[0];
        S(1,1)= img->GetSpacing()[1];
        S(2,2)= img->GetSpacing()[2];

        vnl_vector<double> orig_pt_new = mid_pt.GetVnlVector() - new_dir.GetVnlMatrix()* S * mid_ind.GetVnlVector();

        new_orig[0]=orig_pt_new[0];
        new_orig[1]=orig_pt_new[1];
        new_orig[2]=orig_pt_new[2];

        new_sz=sz;
    }
    else
    {
        double mn[3],mx[3];
        int is[2]={0,(int)sz[0]-1};
        int js[2]={0,(int)sz[1]-1};
        int ks[2]={0,(int)sz[2]-1};

        for(int d=0;d<3;d++)
        {
            mx[d]=-1E10;
            mn[d]=1E10;
            for(int kk=0;kk<2;kk++)
            {
                ImageType3D::IndexType ind3;
                ind3[2]= ks[kk];
                for(int jj=0;jj<2;jj++)
                {
                    ind3[1]= js[jj];
                    for(int ii=0;ii<2;ii++)
                    {
                        ind3[0]= is[ii];
                        ImageType3D::PointType pt;
                        img->TransformIndexToPhysicalPoint(ind3,pt);

                        if(pt[d]<mn[d])
                            mn[d]=pt[d];
                        if(pt[d]>mx[d])
                            mx[d]=pt[d];
                    }
                }
            }
        }
        vnl_vector<double> FOV(3,0);
        FOV[0]= fabs(mx[0]-mn[0]);
        FOV[1]= fabs(mx[1]-mn[1]);
        FOV[2]= fabs(mx[2]-mn[2]);
        vnl_matrix_fixed<double,3,3> Sinv; Sinv.fill(0);
        Sinv(0,0)= 1./img->GetSpacing()[0];
        Sinv(1,1)= 1./img->GetSpacing()[1];
        Sinv(2,2)= 1./img->GetSpacing()[2];
        vnl_matrix_fixed<double,3,3> S; S.fill(0);
        S(0,0)= img->GetSpacing()[0];
        S(1,1)= img->GetSpacing()[1];
        S(2,2)= img->GetSpacing()[2];

        vnl_vector< double> new_sizes_v= Sinv * new_dir.GetTranspose()* FOV;
        new_sz[0] = (unsigned int)(std::ceil(fabs(new_sizes_v[0])));
        new_sz[1] = (unsigned int)(std::ceil(fabs(new_sizes_v[1])));
        new_sz[2] = (unsigned int)(std::ceil(fabs(new_sizes_v[2])));


        itk::ContinuousIndex<double,3> mid_ind;
        mid_ind[0]= ((double)(sz[0])-1)/2.;
        mid_ind[1]= ((double)(sz[1])-1)/2.;
        mid_ind[2]= ((double)(sz[2])-1)/2.;
        ImageType3D::PointType mid_pt;
        img->TransformContinuousIndexToPhysicalPoint(mid_ind,mid_pt);
        mid_ind[0]= ((double)(new_sz[0])-1)/2.;
        mid_ind[1]= ((double)(new_sz[1])-1)/2.;
        mid_ind[2]= ((double)(new_sz[2])-1)/2.;

        vnl_vector<double> new_orig_v = mid_pt.GetVnlVector() - new_dir.GetVnlMatrix() * S * mid_ind.GetVnlVector();
        ImageType3D::PointType new_orig;
        new_orig[0]=new_orig_v[0];
        new_orig[1]=new_orig_v[1];
        new_orig[2]=new_orig_v[2];
    }


    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,new_sz);

    ImageType3D::Pointer ref_img= ImageType3D::New();
    ref_img->SetRegions(reg);
    ref_img->SetSpacing(new_spc);
    ref_img->SetDirection(new_dir);
    ref_img->SetOrigin(new_orig);

    using BSPInterPolatorType = itk::BSplineInterpolateImageFunction<ImageType3D,double>;
    BSPInterPolatorType::Pointer interp = BSPInterPolatorType::New();
    interp->SetSplineOrder(3);
    interp->SetInputImage(img);

    using IdTransformType = itk::IdentityTransform<double,3>;
    IdTransformType::Pointer id_trans= IdTransformType::New();
    id_trans->SetIdentity();

    using ResamplerType = itk::ResampleImageFilter<ImageType3D,ImageType3D>;
    ResamplerType::Pointer resampler= ResamplerType::New();
    resampler->SetDefaultPixelValue(0);
    resampler->SetInput(img);
    resampler->SetInterpolator(interp);
    resampler->SetOutputParametersFromImage(ref_img);
    resampler->SetTransform(id_trans);
    resampler->Update();
    ImageType3D::Pointer new_img= resampler->GetOutput();

    return new_img;
}


int main(int argc, char* argv[])
{
    if(argc==1)
       {
           std::cout<<"Usage: UnObliqueImage input_img output_img keep_orig_size (0/1)"<<std::endl;
           exit(EXIT_FAILURE);
       }

    bool keep_orig = (bool) (atoi(argv[3]));

    ImageType3D::Pointer img = readImageD<ImageType3D>(argv[1]);

    ImageType3D::Pointer out_img = UnObliqueImage( img,keep_orig);

    writeImageD<ImageType3D>(out_img,argv[2]);



}


#endif
