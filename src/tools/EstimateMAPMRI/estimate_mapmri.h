#ifndef _ESTIMATEMAPMRISUB_H
#define _ESTIMATEMAPMRISUB_H

#include "TORTOISE.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "bmat2q.h"
#include "fit_mapmri.h"
#include "../utilities/math_utilities.h"




MAPImageType::Pointer EstimateMAPMRI_sub_nomm(int MAP_DEGREE,std::vector<ImageType3D::Pointer> dwis,vnl_matrix<double> Bmatrix, ImageType3D::Pointer A0_image,EVecImageType::Pointer evec_image, EValImageType::Pointer eval_image, std::vector<int> & DT_indices, double small_delta,double big_delta, ImageType3D::Pointer mask_image,std::vector<ImageType3DBool::Pointer> inclusion_img)
{
    auto stream= TORTOISE::stream;        //for logging

    int N_MAPMRI_COEFFS =((((MAP_DEGREE/2)+1)*((MAP_DEGREE/2)+2)*(4*(MAP_DEGREE/2)+3))/6);

    int Nvols=Bmatrix.rows();

    MAPImageType::Pointer mapmri_image =  MAPImageType::New();
    mapmri_image->SetRegions(eval_image->GetLargestPossibleRegion());
    mapmri_image->SetNumberOfComponentsPerPixel(N_MAPMRI_COEFFS) ;
    mapmri_image->Allocate();
    mapmri_image->SetOrigin(eval_image->GetOrigin());
    mapmri_image->SetDirection(eval_image->GetDirection());
    mapmri_image->SetSpacing(eval_image->GetSpacing());
    MAPType zero;     zero.SetSize(N_MAPMRI_COEFFS);     zero.Fill(0);
    mapmri_image->FillBuffer(zero);

    EValImageType::SizeType size = eval_image->GetLargestPossibleRegion().GetSize();

    std::vector<int> all_indices;
    if(DT_indices.size()>0)
        all_indices= DT_indices;
    else
    {
        for(int ma=0;ma<Nvols;ma++)
            all_indices.push_back(ma);
    }

    (*stream)<<"Computing MAPMRI with degree: "<<MAP_DEGREE<<std::endl;

    #pragma omp parallel for
    for(int k=0;k<size[2];k++)
    {
        TORTOISE::EnableOMPThread();

        vnl_matrix<double> qq_orig= bmat2q(Bmatrix,all_indices,small_delta,big_delta);

        ImageType3D::IndexType index3;
        index3[2]=k;
        for(int j=0;j<size[1];j++)
        {
            index3[1]=j;
            for(int i=0;i<size[0];i++)
            {
                index3[0]=i;

                std::vector<float> curr_signal;
                curr_signal.resize(all_indices.size());
                std::vector<int> curr_all_indices= all_indices;
                vnl_matrix<double> curr_qq=qq_orig;


                 EValType eval = eval_image->GetPixel(index3);
                 if( A0_image->GetPixel(index3)!=0)
                 {
                     if((mask_image==nullptr) || mask_image->GetPixel(index3)!=0 )
                     {
                         if(inclusion_img.size())
                         {
                             curr_all_indices.resize(0);
                             int Nincludes=0;
                             for(int vol=0;vol<all_indices.size();vol++)
                             {
                                 int vol_id= all_indices[vol];
                                 float inc= inclusion_img[vol_id]->GetPixel(index3);
                                 if(inc!=0)
                                 {
                                     curr_all_indices.push_back(all_indices[vol]);
                                     Nincludes++;
                                 }
                             }
                             curr_signal.resize(Nincludes);
                             curr_qq.set_size(3,Nincludes);
                         }


                         for(int vol=0;vol<curr_all_indices.size();vol++)
                         {
                             int vol_id= curr_all_indices[vol];
                             double nval = dwis[vol_id]->GetPixel(index3);
                             if(nval <=0)
                             {
                                 std::vector<float> data_for_median;

                                 ImageType3D::IndexType newind;
                                 newind[2]=k;
                                 for(int i2= std::max(i-1,0);i2<std::min(i+1,(int)size[0]-1);i2++)
                                 {
                                     newind[0]=i2;
                                     for(int j2= std::max(j-1,0);j2<std::min(j+1,(int)size[1]-1);j2++)
                                     {
                                         newind[1]=j2;

                                         float newval= dwis[vol_id]->GetPixel(newind);
                                         if(newval >0)
                                             data_for_median.push_back(newval);
                                     }
                                 }

                                 if(data_for_median.size())
                                 {
                                     nval=median(data_for_median);

                                 }
                                 else
                                 {
                                     nval=0.0001;
                                 }
                             }
                             curr_signal[vol]=nval;
                             curr_qq(0,vol)= qq_orig(0,curr_all_indices[vol]);
                             curr_qq(1,vol)= qq_orig(1,curr_all_indices[vol]);
                             curr_qq(2,vol)= qq_orig(2,curr_all_indices[vol]);
                         }

                         EVecType R = evec_image->GetPixel(index3);
                         vnl_matrix<double> qqtmp= R * curr_qq;


                         MAPType coeffs= FitMAPMRI(curr_signal, A0_image->GetPixel(index3), MAP_DEGREE, eval_image->GetPixel(index3), qqtmp,big_delta-small_delta/3. ,0);

                         mapmri_image->SetPixel(index3,coeffs);
                     }
                 }
             }
         }
         TORTOISE::DisableOMPThread();
     }


    return mapmri_image;


}




#endif
