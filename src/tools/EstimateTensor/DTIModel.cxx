#ifndef _DTIModel_CXX
#define _DTIModel_CXX


#include "DTIModel.h"
#include "../utilities/math_utilities.h"
#include <math.h>
#include "../external_src/cmpfit-1.3a/mpfit.h"

int myNLLS_with_derivs(int m, int n, double *p, double *deviates,   double **derivs, void *vars)
{
    int i;
    // m points, n params

    struct vars_struct *v = (struct vars_struct *) vars;
    vnl_matrix<double> *Bmat= v->Bmat;
    vnl_vector<double> *signal= v->signal;
    vnl_vector<double> *weights=v->weights;


    for (i=0; i<m; i++)
    {
        double expon = (*Bmat)[i][1]* p[1] + (*Bmat)[i][2]* p[2]+(*Bmat)[i][3]* p[3]+(*Bmat)[i][4]* p[4]+(*Bmat)[i][5]* p[5]+(*Bmat)[i][6]* p[6];

        double est=  exp(expon);
        if(v->useWeights)
            deviates[i] =  ((*signal)[i]- p[0]*est) * (*weights)[i];
        else
            deviates[i] =(*signal)[i]- p[0]*est;

        if (derivs)
        {
            if(v->useWeights)
            {
                derivs[0][i]= -est* (*weights)[i];
                est= -est*p[0]* (*weights)[i];

                derivs[1][i]=  est*  (*Bmat)[i][1];
                derivs[2][i]=  est*  (*Bmat)[i][2];
                derivs[3][i]=  est*  (*Bmat)[i][3];
                derivs[4][i]=  est*  (*Bmat)[i][4];
                derivs[5][i]=  est*  (*Bmat)[i][5];
                derivs[6][i]=  est*  (*Bmat)[i][6];
            }
            else
            {
                derivs[0][i]= -est;
                est= -est*p[0];

                derivs[1][i]=  est*  (*Bmat)[i][1];
                derivs[2][i]=  est*  (*Bmat)[i][2];
                derivs[3][i]=  est*  (*Bmat)[i][3];
                derivs[4][i]=  est*  (*Bmat)[i][4];
                derivs[5][i]=  est*  (*Bmat)[i][5];
                derivs[6][i]=  est*  (*Bmat)[i][6];

            }
        }
    }

    return 0;
}



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
    if(fitting_mode=="NLLS")
    {
        EstimateTensorNLLS();
    }
}


void DTIModel::EstimateTensorNLLS()
{
    EstimateTensorWLLS();

    int Nvols=Bmatrix.rows();

    std::vector<int> all_indices;
    if(indices_fitting.size()>0)
        all_indices= indices_fitting;
    else
    {
        for(int ma=0;ma<Nvols;ma++)
            all_indices.push_back(ma);
    }

    if(stream)
        (*stream)<<"Computing Tensors NLLS..."<<std::endl;
    else
        std::cout<<"Computing Tensors NLLS..."<<std::endl;

    ImageType3D::SizeType size = dwi_data[0]->GetLargestPossibleRegion().GetSize();

    mp_config_struct config;
    config.maxiter=500;
    config.ftol=1E-10;
    config.xtol=1E-10;
    config.gtol=1E-10;
    config.epsfcn=MP_MACHEP0;
    config.stepfactor=100;
    config.covtol=1E-14;
    config.maxfev=0;
    config.nprint=1;
    config.douserscale=0;
    config.nofinitecheck=0;

    CS_img=ImageType3D::New();
    CS_img->SetRegions(A0_img->GetLargestPossibleRegion());
    CS_img->Allocate();
    CS_img->SetSpacing(A0_img->GetSpacing());
    CS_img->SetOrigin(A0_img->GetOrigin());
    CS_img->SetDirection(A0_img->GetDirection());
    CS_img->FillBuffer(0.);



    #pragma omp parallel for
    for(int k=0;k<size[2];k++)
    {
        #ifndef NOTORTOISE
        TORTOISE::EnableOMPThread();
        #endif
        ImageType3D::IndexType ind3;
        ind3[2]=k;

        mp_par pars[7];
        memset(&pars[0], 0, sizeof(pars));
        pars[0].side=3;
        pars[1].side=3;
        pars[2].side=3;
        pars[3].side=3;
        pars[4].side=3;
        pars[5].side=3;
        pars[6].side=3;



        for(int j=0;j<size[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<size[0];i++)
            {
                ind3[0]=i;

                if(mask_img && mask_img->GetPixel(ind3)==0)
                    continue;

                std::vector<int> curr_all_indices;
                if(this->weight_imgs.size())
                {
                    for(int v=0;v<all_indices.size();v++)
                    {
                        int vol_id= curr_all_indices[v];
                        if(this->weight_imgs[vol_id]->GetPixel(ind3)>0)
                            curr_all_indices.push_back(vol_id);
                    }
                }
                else
                {
                    curr_all_indices=all_indices;
                }

                vnl_vector<double> signal(curr_all_indices.size());
                vnl_vector<double> weights(curr_all_indices.size(),1.);

                vnl_matrix<double> curr_design_matrix(curr_all_indices.size(),7);
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
                    signal[vol] = nval;
                    if(this->weight_imgs.size())
                    {
                        weights[vol] =this->weight_imgs[vol_id]->GetPixel(ind3);
                    }

                    curr_design_matrix(vol,0)=1;
                    if(graddev_img.size())
                    {
                        vnl_matrix_fixed<double,3,3> L, B;
                        L(0,0)= 1+graddev_img[0]->GetPixel(ind3); L(0,1)= graddev_img[1]->GetPixel(ind3); L(0,2)= graddev_img[2]->GetPixel(ind3);
                        L(1,0)= graddev_img[3]->GetPixel(ind3); L(1,1)= 1+graddev_img[4]->GetPixel(ind3); L(1,2)= graddev_img[5]->GetPixel(ind3);
                        L(2,0)= graddev_img[6]->GetPixel(ind3); L(2,1)= graddev_img[7]->GetPixel(ind3); L(2,2)= 1+graddev_img[8]->GetPixel(ind3);

                        B(0,0)= Bmatrix(vol_id,0); B(0,1)= Bmatrix(vol_id,1)/2;  B(0,2)= Bmatrix(vol_id,2)/2;
                        B(1,0)= Bmatrix(vol_id,1)/2; B(1,1)= Bmatrix(vol_id,3);  B(1,2)= Bmatrix(vol_id,4)/2;
                        B(2,0)= Bmatrix(vol_id,2)/2; B(2,1)= Bmatrix(vol_id,4)/2;  B(2,2)= Bmatrix(vol_id,5);

                        B= L * B * L.transpose();
                        curr_design_matrix(vol,1)= -B(0,0)/1000;
                        curr_design_matrix(vol,2)= -B(0,1)/1000;
                        curr_design_matrix(vol,3)= -B(0,2)/1000;
                        curr_design_matrix(vol,4)= -B(1,1)/1000;
                        curr_design_matrix(vol,5)= -B(1,2)/1000;
                        curr_design_matrix(vol,6)= -B(2,2)/1000;
                    }
                    if(voxelwise_Bmatrix.size())
                    {
                        curr_design_matrix(vol,1)= -voxelwise_Bmatrix[vol_id][0]->GetPixel(ind3)/1000.;
                        curr_design_matrix(vol,2)= -voxelwise_Bmatrix[vol_id][1]->GetPixel(ind3)/1000.;
                        curr_design_matrix(vol,3)= -voxelwise_Bmatrix[vol_id][2]->GetPixel(ind3)/1000.;
                        curr_design_matrix(vol,4)= -voxelwise_Bmatrix[vol_id][3]->GetPixel(ind3)/1000.;
                        curr_design_matrix(vol,5)= -voxelwise_Bmatrix[vol_id][4]->GetPixel(ind3)/1000.;
                        curr_design_matrix(vol,6)= -voxelwise_Bmatrix[vol_id][5]->GetPixel(ind3)/1000.;
                    }
                } //for vol

                DTImageType::PixelType dt_vec= this->output_img->GetPixel(ind3);
                double p[7];
                p[0]= this->A0_img->GetPixel(ind3);
                p[1]= dt_vec[0]*1000.;
                p[2]= dt_vec[1]*1000.;
                p[3]= dt_vec[2]*1000.;
                p[4]= dt_vec[3]*1000.;
                p[5]= dt_vec[4]*1000.;
                p[6]= dt_vec[5]*1000.;

                vars_struct my_struct;
                my_struct.useWeights=true;

                mp_result_struct my_results_struct;
                vnl_vector<double> my_resids(curr_all_indices.size());
                my_results_struct.resid= my_resids.data_block();
                my_results_struct.xerror=nullptr;
                my_results_struct.covar=nullptr;
                my_struct.signal= &signal;
                my_struct.weights=&weights;
                my_struct.Bmat= &curr_design_matrix;

                int status = mpfit(myNLLS_with_derivs, curr_design_matrix.rows(), 7, p, pars, &config, (void *) &my_struct, &my_results_struct);

                double degrees_of_freedom= curr_all_indices.size()-7;
                CS_img->SetPixel(ind3,my_results_struct.bestnorm/degrees_of_freedom);

                dt_vec[0]= p[1]/1000.;
                dt_vec[1]= p[2]/1000.;
                dt_vec[2]= p[3]/1000.;
                dt_vec[3]= p[4]/1000.;
                dt_vec[4]= p[5]/1000.;
                dt_vec[5]= p[6]/1000.;


                if(p[1]> 3.2 || p[4] > 3.2 || p[6]>3.2)  //FLOW ARTIFACT
                {
                    dt_vec= output_img->GetPixel(ind3);
                }

                 if(p[2]>1.1 || p[3]> 1.1 || p[5]>1.1)  //FLOW ARTIFACT
                 {
                     dt_vec= output_img->GetPixel(ind3);
                 }

                 output_img->SetPixel(ind3,dt_vec);
                 A0_img->SetPixel(ind3,p[0]);
            } //for i
        } //for j
    } //for k


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
         (*stream)<<"Computing Tensors WLLS..."<<std::endl;
     else
         std::cout<<"Computing Tensors WLLS..."<<std::endl;

     ImageType3D::SizeType size = dwi_data[0]->GetLargestPossibleRegion().GetSize();

     #pragma omp parallel for
     for(int k=0;k<size[2];k++)
     {
         #ifndef NOTORTOISE
         TORTOISE::EnableOMPThread();
         #endif
         ImageType3D::IndexType ind3;
         ind3[2]=k;


         for(int j=0;j<size[1];j++)
         {
             ind3[1]=j;
             for(int i=0;i<size[0];i++)
             {
                 ind3[0]=i;

                 if(mask_img && mask_img->GetPixel(ind3)==0)
                     continue;

                 std::vector<int> curr_all_indices;
                 if(this->weight_imgs.size())
                 {
                     for(int v=0;v<all_indices.size();v++)
                     {
                         int vol_id= curr_all_indices[v];
                         if(this->weight_imgs[vol_id]->GetPixel(ind3)>0)
                             curr_all_indices.push_back(vol_id);
                     }
                 }
                 else
                 {
                     curr_all_indices=all_indices;
                 }


                 vnl_matrix<double> curr_logS(curr_all_indices.size(),1);
                 vnl_diag_matrix<double> curr_weights(curr_all_indices.size(),curr_all_indices.size());
                 vnl_matrix<double> curr_design_matrix(curr_all_indices.size(),7,0);

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
                     if(graddev_img.size())
                     {
                         vnl_matrix_fixed<double,3,3> L, B;
                         L(0,0)= 1+graddev_img[0]->GetPixel(ind3); L(0,1)= graddev_img[1]->GetPixel(ind3); L(0,2)= graddev_img[2]->GetPixel(ind3);
                         L(1,0)= graddev_img[3]->GetPixel(ind3); L(1,1)= 1+graddev_img[4]->GetPixel(ind3); L(1,2)= graddev_img[5]->GetPixel(ind3);
                         L(2,0)= graddev_img[6]->GetPixel(ind3); L(2,1)= graddev_img[7]->GetPixel(ind3); L(2,2)= 1+graddev_img[8]->GetPixel(ind3);

                         B(0,0)= Bmatrix(vol_id,0); B(0,1)= Bmatrix(vol_id,1)/2;  B(0,2)= Bmatrix(vol_id,2)/2;
                         B(1,0)= Bmatrix(vol_id,1)/2; B(1,1)= Bmatrix(vol_id,3);  B(1,2)= Bmatrix(vol_id,4)/2;
                         B(2,0)= Bmatrix(vol_id,2)/2; B(2,1)= Bmatrix(vol_id,4)/2;  B(2,2)= Bmatrix(vol_id,5);

                         B= L * B * L.transpose();
                         curr_design_matrix(vol,1)= -B(0,0)/1000;
                         curr_design_matrix(vol,2)= -B(0,1)/1000;
                         curr_design_matrix(vol,3)= -B(0,2)/1000;
                         curr_design_matrix(vol,4)= -B(1,1)/1000;
                         curr_design_matrix(vol,5)= -B(1,2)/1000;
                         curr_design_matrix(vol,6)= -B(2,2)/1000;
                     }
                     if(voxelwise_Bmatrix.size())
                     {
                         curr_design_matrix(vol,1)= -voxelwise_Bmatrix[vol_id][0]->GetPixel(ind3)/1000.;
                         curr_design_matrix(vol,2)= -voxelwise_Bmatrix[vol_id][1]->GetPixel(ind3)/1000.;
                         curr_design_matrix(vol,3)= -voxelwise_Bmatrix[vol_id][2]->GetPixel(ind3)/1000.;
                         curr_design_matrix(vol,4)= -voxelwise_Bmatrix[vol_id][3]->GetPixel(ind3)/1000.;
                         curr_design_matrix(vol,5)= -voxelwise_Bmatrix[vol_id][4]->GetPixel(ind3)/1000.;
                         curr_design_matrix(vol,6)= -voxelwise_Bmatrix[vol_id][5]->GetPixel(ind3)/1000.;
                     }
                 } //for vol

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
         #ifndef NOTORTOISE
         TORTOISE::DisableOMPThread();
         #endif
      } //for k


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
