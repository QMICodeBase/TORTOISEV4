#ifndef _DTIModel_CXX
#define _DTIModel_CXX


#include "DTIModel.h"
#include "../utilities/math_utilities.h"
#include <math.h>


void DTIModel::PerformFitting()
{
    if(fitting_mode=="")
    {
        (*stream)<<"DTI fitting mode not set. Defaulting to WLLS"<<std::endl;
        fitting_mode="WLLS";
    }

    if(fitting_mode=="WLLS")
    {
        EstimateTensorWLLS();
    }

}

void DTIModel::EstimateTensorWLLS()
{
    int Nvols=Bmatrix.rows();


    std::vector<int> all_indices;
    if(indices_fitting.size()>0)
        all_indices= indices_fitting;
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
     dt_image->SetRegions(dwi_data[0]->GetLargestPossibleRegion());
     dt_image->Allocate();
     dt_image->SetSpacing(dwi_data[0]->GetSpacing());
     dt_image->SetOrigin(dwi_data[0]->GetOrigin());
     dt_image->SetDirection(dwi_data[0]->GetDirection());
     DTType zerodt; zerodt.Fill(0);
     dt_image->FillBuffer(zerodt);

     A0_img= ImageType3D::New();
     A0_img->SetRegions(dwi_data[0]->GetLargestPossibleRegion());
     A0_img->Allocate();
     A0_img->SetOrigin(dwi_data[0]->GetOrigin());
     A0_img->SetSpacing(dwi_data[0]->GetSpacing());
     A0_img->SetDirection(dwi_data[0]->GetDirection());
     A0_img->FillBuffer(0);

     if(stream)
         (*stream)<<"Computing Tensors..."<<std::endl;
     else
         std::cout<<"Computing Tensors..."<<std::endl;

     ImageType3D::SizeType size = dwi_data[0]->GetLargestPossibleRegion().GetSize();

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

                 if(mask_img && mask_img->GetPixel(ind3)==0)
                     continue;

                 for(int vol=0;vol<curr_all_indices.size();vol++)
                 {
                     int vol_id= curr_all_indices[vol];
                     double nval = dwi_data[vol_id]->GetPixel(ind3);
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

                                 float newval= dwi_data[vol_id]->GetPixel(newind);
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
                     if(this->weight_imgs.size())
                     {
                         float weight= this->weight_imgs[vol_id]->GetPixel(ind3);
                         curr_weights[vol]*= weight*weight;
                     }
                     curr_design_matrix(vol,0)=1;
                     for(int jj=0;jj<6;jj++)
                     {
                         curr_design_matrix(vol,jj+1)= -Bmatrix(curr_all_indices[vol],jj)/1000.;
                     }
                 }

                 vnl_matrix<double> mid= curr_design_matrix.transpose()* curr_weights * curr_design_matrix;
                 vnl_matrix<double> D= vnl_svd<double>(mid).solve(curr_design_matrix.transpose()*curr_weights*curr_logS);

                 if(D(0,0)==D(0,0))
                     A0_img->SetPixel(ind3,  exp(D(0,0)));
                 else
                     A0_img->SetPixel(ind3,  -1);

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
                     A0_img->SetPixel(ind3,  0);
                 }

                 double mx = exp(curr_logS.max_value());
                 if(A0_img->GetPixel(ind3)>=3*mx)
                 {
                     A0_img->SetPixel(ind3,  0);
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


     if(!mask_img)
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

                                 float newval= A0_img->GetPixel(newind);
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



                         float val = A0_img->GetPixel(ind3);
                         if( val < med-10*stdev  || val > med+10*stdev)
                             A0_img->SetPixel(ind3,0);
                     }
                     else
                     {
                        A0_img->SetPixel(ind3,0);
                     }
                 }
             }
         }
     }
     output_img=dt_image;
}



ImageType3D::Pointer DTIModel::SynthesizeDWI(vnl_vector<double> bmatrix_vec)
{
    ImageType3D::Pointer synth_image = ImageType3D::New();
    synth_image->SetRegions(A0_img->GetLargestPossibleRegion());
    synth_image->Allocate();
    synth_image->SetOrigin(A0_img->GetOrigin());
    synth_image->SetDirection(A0_img->GetDirection());
    synth_image->SetSpacing(A0_img->GetSpacing());
    synth_image->FillBuffer(0.);



    itk::ImageRegionIteratorWithIndex<ImageType3D> it(synth_image,synth_image->GetLargestPossibleRegion());
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind=it.GetIndex();
        DTType tensor= output_img->GetPixel(ind);
        double exp_term=0;
        for(int i=0;i<6;i++)
            exp_term += tensor[i] * bmatrix_vec[i];


        if(exp_term < 0)
            exp_term=0;

         float A0val= A0_img->GetPixel(ind);

        float signal = A0val * exp(-exp_term);
        if(!isnan(signal) && isfinite(signal))
            it.Set(signal);
        ++it;
    }

    return synth_image;

}





#endif
