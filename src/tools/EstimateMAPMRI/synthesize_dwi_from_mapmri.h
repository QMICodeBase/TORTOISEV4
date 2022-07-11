#ifndef _SYNTHESIZEDWIFROMMAPMRI_H
#define _SYNTHESIZEDWIFROMMAPMRI_H


#include "defines.h"
#include "itkImportImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"


ImageType3D::Pointer SynthesizeDWIFromMAPMRI(MAPImageType::Pointer mapmri_image,ImageType3D::Pointer A0_image,EVecImageType::Pointer evec_image,EValImageType::Pointer eval_image, vnl_matrix<double> bmat_vec, double small_delta,double big_delta )
{
    MAPImageType::IndexType ind_temp; ind_temp.Fill(0);
    MAPImageType::PixelType vec= mapmri_image->GetPixel(ind_temp);
    int ncoeffs= vec.Size();
    int MAP_ORDER;
    switch(ncoeffs)
    {
        case 7:
            MAP_ORDER=2;
            break;
        case 22:
            MAP_ORDER=4;
            break;
        case 50:
            MAP_ORDER=6;
            break;
        case 95:
            MAP_ORDER=8;
            break;
        case 161:
            MAP_ORDER=10;
            break;
        default:
            std::cout<<"MAPMRI number of coefficients do not match any order. Exiting..."<<std::endl;
            return nullptr;
    }




    ImageType3D::Pointer synth_image =  ImageType3D::New();
    synth_image->SetRegions(eval_image->GetLargestPossibleRegion());
    synth_image->Allocate();
    synth_image->SetOrigin(eval_image->GetOrigin());
    synth_image->SetDirection(eval_image->GetDirection());
    synth_image->SetSpacing(eval_image->GetSpacing());

    synth_image->FillBuffer(0.);

    ImageType3D::SizeType  size;
    size[0]= eval_image->GetLargestPossibleRegion().GetSize()[0];
    size[1]= eval_image->GetLargestPossibleRegion().GetSize()[1];
    size[2]= eval_image->GetLargestPossibleRegion().GetSize()[2];

    double tdiff= big_delta - small_delta/3;
    std::vector<int> all_indices;
    all_indices.push_back(0);
    vnl_matrix<double> qq= bmat2q(bmat_vec,all_indices,small_delta,big_delta);

    for(int k=0;k<size[2];k++)
    {
        ImageType3D::IndexType index3;
        index3[2]=k;
        for(int j=0;j<size[1];j++)
        {
            index3[1]=j;
            for(int i=0;i<size[0];i++)
            {
                index3[0]=i;

                MAPType coeffs= mapmri_image->GetPixel(index3);
                EVecType R= evec_image->GetPixel(index3);
                EValType uvec= eval_image->GetPixel(index3);


                vnl_vector<double> uu(3);
                uu[0]= sqrt(uvec[0]*2000.*tdiff);
                uu[1]= sqrt(uvec[1]*2000.*tdiff);
                uu[2]= sqrt(uvec[2]*2000.*tdiff);

                vnl_matrix<double> qqtmp= R * qq;

                MatrixXd qmtrx=mk_ashore_basis(MAP_ORDER,uu,qqtmp,1);

                double sm=0;
                for(int i=0;i<qmtrx.rows();i++)
                    sm+= coeffs[i] * qmtrx(i,0);


                //sm= sm* A0_image->GetPixel(index3);

                if(!isnan(sm) && isfinite(sm))
                    synth_image->SetPixel(index3,sm);
            }
        }
    }


    float hist_min = 1E-5;

    std::vector<float> vals;
    itk::ImageRegionIteratorWithIndex<ImageType3D> it(synth_image,synth_image->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind = it.GetIndex();
        if(A0_image->GetPixel(ind) > 0)
        {
            float val= it.Get();
            if(val > hist_min)
                vals.push_back(val);
        }
        ++it;
    }

    float median_val=median(vals);
    for(int s=0;s<vals.size();s++)
    {
        vals[s]=fabs(vals[s]-median_val);
    }
    float MAD = median(vals);



    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind = it.GetIndex();
        if(A0_image->GetPixel(ind) > 0)
        {
            float val= it.Get();
            if(   (val > median_val + 20*MAD) || (val < median_val - 20*MAD) )
                it.Set(0);
        }
        ++it;
    }



    return synth_image;
}









#endif
