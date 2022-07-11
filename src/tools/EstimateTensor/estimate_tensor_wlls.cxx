#ifndef _ESTIMATETENSORWLLSSUB_CXX
#define _ESTIMATETENSORWLLSSUB_CXX


#include "estimate_tensor_wlls.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "TORTOISE.h"
#include "../utilities/math_utilities.h"
#include "itkImportImageFilter.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include <numeric>



DTImageType::Pointer   EstimateTensorWLLS_sub_nomm(std::vector<ImageType3D::Pointer> dwis, vnl_matrix<double> Bmatrix,std::vector<int> &DT_indices, ImageType3D::Pointer & A0_image,ImageType3D::Pointer mask_image,std::vector<ImageType3DBool::Pointer> inclusion_img )
{
    auto stream= TORTOISE::stream;        //for logging

    int Nvols=Bmatrix.rows();


    std::vector<int> all_indices;
    if(DT_indices.size()>0)
        all_indices= DT_indices;
    else
    {
        for(int ma=0;ma<Nvols;ma++)
            all_indices.push_back(ma);
    }


    vnl_matrix<double> design_matrix(all_indices.size(),7);
    for(int i=0;i<all_indices.size();i++)
    {
        design_matrix(i,0)=1;
        for(int j=0;j<6;j++)
        {
            design_matrix(i,j+1)= -Bmatrix(all_indices[i],j)/1000.;
        }
    }

     DTImageType::Pointer dt_image =  DTImageType::New();
     dt_image->SetRegions(dwis[0]->GetLargestPossibleRegion());
     dt_image->Allocate();
     dt_image->SetSpacing(dwis[0]->GetSpacing());
     dt_image->SetOrigin(dwis[0]->GetOrigin());
     dt_image->SetDirection(dwis[0]->GetDirection());
     DTType zerodt; zerodt.Fill(0);
     dt_image->FillBuffer(zerodt);

     A0_image= ImageType3D::New();
     A0_image->SetRegions(dwis[0]->GetLargestPossibleRegion());
     A0_image->Allocate();
     A0_image->SetOrigin(dwis[0]->GetOrigin());
     A0_image->SetSpacing(dwis[0]->GetSpacing());
     A0_image->SetDirection(dwis[0]->GetDirection());
     A0_image->FillBuffer(0);

     if(stream)
         (*stream)<<"Computing Tensors..."<<std::endl;
     else
         std::cout<<"Computing Tensors..."<<std::endl;

     ImageType3D::SizeType size = dwis[0]->GetLargestPossibleRegion().GetSize();

     #pragma omp parallel for
     for(int k=0;k<size[2];k++)
     {
         TORTOISE::EnableOMPThread();
         ImageType3D::IndexType ind3;
         ind3[2]=k;

         vnl_matrix<double> logS(all_indices.size(),1);
         vnl_diag_matrix<double> weights(all_indices.size(),0.0);

         vnl_matrix<double> curr_logS=logS;
         vnl_diag_matrix<double> curr_weights=weights;
         std::vector<int> curr_all_indices= all_indices;
         vnl_matrix<double> curr_design_matrix=design_matrix;

         for(int j=0;j<size[1];j++)
         {
             ind3[1]=j;
             for(int i=0;i<size[0];i++)
             {
                 ind3[0]=i;

                 if(mask_image && mask_image->GetPixel(ind3)==0)
                     continue;

                 if(inclusion_img.size())
                 {
                     curr_all_indices.resize(0);
                     int Nincludes=0;
                     for(int vol=0;vol<all_indices.size();vol++)
                     {
                         int vol_id= all_indices[vol];
                         bool inc= inclusion_img[vol_id]->GetPixel(ind3);
                         if(inc)
                         {
                             curr_all_indices.push_back(vol_id);
                             Nincludes++;
                         }
                     }

                     if(Nincludes <=10)
                     {
                         curr_logS.set_size(all_indices.size(),1);
                         curr_weights.set_size(all_indices.size());
                         curr_all_indices= all_indices;
                         curr_design_matrix=design_matrix;
                     }
                     else
                     {
                         curr_logS.set_size(Nincludes,1);
                         curr_weights.set_size(Nincludes);
                         curr_design_matrix.set_size(Nincludes,7);
                     }
                 }

                 for(int vol=0;vol<curr_all_indices.size();vol++)
                 {
                     int vol_id= curr_all_indices[vol];
                     double nval = dwis[vol_id]->GetPixel(ind3);
                     if(nval <=0)
                     {
                         std::vector<float> data_for_median;
                         std::vector<float> noise_for_median;

                         ImageType3D::IndexType newind;
                         newind[2]=k;

                         for(int i2= std::max(i-1,0);i2<=std::min(i+1,(int)size[0]-1);i2++)
                         {
                             newind[0]=i2;
                             for(int j2= std::max(j-1,0);j2<=std::min(j+1,(int)size[1]-1);j2++)
                             {
                                 newind[1]=j2;

                                 float newval= dwis[vol_id]->GetPixel(newind);
                                 if(newval >0)
                                 {
                                     data_for_median.push_back(newval);
                                 }
                             }
                         }

                         if(data_for_median.size())
                         {
                             nval=median(data_for_median);
                         }
                         else
                         {
                             nval= 1E-3;
                         }
                     }

                     curr_logS(vol,0)= log(nval);
                     curr_weights[vol]= nval*nval;
                     curr_design_matrix(vol,0)=1;
                     for(int jj=0;jj<6;jj++)
                     {
                         curr_design_matrix(vol,jj+1)= -Bmatrix(curr_all_indices[vol],jj)/1000.;
                     }
                 }

                 vnl_matrix<double> mid= curr_design_matrix.transpose()* curr_weights * curr_design_matrix;
                 vnl_matrix<double> D= vnl_svd<double>(mid).solve(curr_design_matrix.transpose()*curr_weights*curr_logS);

                 if(D(0,0)==D(0,0))
                     A0_image->SetPixel(ind3,  exp(D(0,0)));
                 else
                     A0_image->SetPixel(ind3,  -1);

                 if(D(1,0)!=D(1,0))
                     D(1,0)=0;
                 if(D(2,0)!=D(2,0))
                     D(2,0)=0;
                 if(D(3,0)!=D(3,0))
                     D(3,0)=0;
                 if(D(4,0)!=D(4,0))
                     D(4,0)=0;
                 if(D(5,0)!=D(5,0))
                     D(5,0)=0;
                 if(D(6,0)!=D(6,0))
                     D(6,0)=0;

                 vnl_matrix_fixed<double,3,3> Dmat;
                 Dmat(0,0)=D(1,0)/1000.;
                 Dmat(1,0)=D(2,0)/1000.;
                 Dmat(0,1)=D(2,0)/1000.;
                 Dmat(2,0)=D(3,0)/1000.;
                 Dmat(0,2)=D(3,0)/1000.;
                 Dmat(1,1)=D(4,0)/1000.;
                 Dmat(1,2)=D(5,0)/1000.;
                 Dmat(2,1)=D(5,0)/1000.;
                 Dmat(2,2)=D(6,0)/1000.;

                 vnl_symmetric_eigensystem<double> eig(Dmat);
                 if(eig.D(0,0)<0)
                     eig.D(0,0)=0;
                 if(eig.D(1,1)<0)
                     eig.D(1,1)=0;
                 if(eig.D(2,2)<0)
                     eig.D(2,2)=0;
                 vnl_matrix_fixed<double,3,3> Dmat_corr= eig.recompose();

                 if(Dmat_corr(0,0)+Dmat_corr(1,1)+Dmat_corr(2,2) >0.2)    //bad fit
                 {
                     Dmat_corr.fill(0);
                     A0_image->SetPixel(ind3,  0);
                 }

                 double mx = exp(curr_logS.max_value());
                 if(A0_image->GetPixel(ind3)>=3*mx)
                 {
                     A0_image->SetPixel(ind3,  0);
                 }


                 DTImageType::PixelType dt;
                 dt[0]=Dmat_corr(0,0);
                 dt[1]=Dmat_corr(0,1);
                 dt[2]=Dmat_corr(0,2);
                 dt[3]=Dmat_corr(1,1);
                 dt[4]=Dmat_corr(1,2);
                 dt[5]=Dmat_corr(2,2);

                 dt_image->SetPixel(ind3,dt);
             }
         }
         TORTOISE::DisableOMPThread();
      }


     if(!mask_image)
     {
         for(int k=0;k<size[2];k++)
         {
              ImageType3D::IndexType ind3;
             ind3[2]=k;
             for(int j=0;j<size[1];j++)
             {
                 ind3[1]=j;
                 for(int i=0;i<size[0];i++)
                 {
                     ind3[0]=i;

                     std::vector<float> data_for_median;
                      ImageType3D::IndexType newind;

                     for(int k2= std::max(k-1,0);k2<=std::min(k+1,(int)size[2]-1);k2++)
                     {
                         newind[2]=k2;
                         for(int i2= std::max(i-1,0);i2<=std::min(i+1,(int)size[0]-1);i2++)
                         {
                             newind[0]=i2;
                             for(int j2= std::max(j-1,0);j2<=std::min(j+1,(int)size[1]-1);j2++)
                             {
                                 newind[1]=j2;

                                 float newval= A0_image->GetPixel(newind);
                                 if(newval >=0)
                                     data_for_median.push_back(newval);

                             }
                         }
                     }

                     if(data_for_median.size())
                     {
                         float med=median(data_for_median);

                         std::transform(data_for_median.begin(), data_for_median.end(), data_for_median.begin(), bind2nd(std::plus<double>(), -med));

                         for(unsigned int mi = 0; mi < data_for_median.size(); mi++)
                             if(data_for_median[mi] < 0)
                                 data_for_median[mi] *= -1;

                         float abs_med= median(data_for_median);
                         float stdev= abs_med *  1.4826;



                         float val = A0_image->GetPixel(ind3);
                         if( val < med-10*stdev  || val > med+10*stdev)
                             A0_image->SetPixel(ind3,0);
                     }
                     else
                     {
                        A0_image->SetPixel(ind3,0);
                     }
                 }
             }
         }
     }
    return dt_image;
}



#endif
