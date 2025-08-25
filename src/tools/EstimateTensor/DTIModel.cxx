#ifndef _DTIModel_CXX
#define _DTIModel_CXX


#include "DTIModel.h"
#include "../utilities/math_utilities.h"
#include <math.h>
#include "../external_src/cmpfit-1.3a/mpfit.h"
#include "itkImageDuplicator.h"
#include "vnl/algo/vnl_cholesky.h"

void DTIModel::PerformFitting()
{
    if(fitting_mode!="SPD" &&  fitting_mode!="WLLS" &&  fitting_mode!="NLLS" && fitting_mode!="N2" && fitting_mode!="RESTORE" && fitting_mode!="DIAG"  && fitting_mode!="NT2")
    {
        std::cout<<"DTI fitting mode not set correctly. Defaulting to WLLS"<<std::endl;
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
    if(fitting_mode=="SPD")
    {
        EstimateTensorSPD();
    }
    if(fitting_mode=="N2")
    {
        EstimateTensorN2();
    }
    if(fitting_mode=="DIAG")
    {
        EstimateTensorWLLSDiagonal();
    }
    if(fitting_mode=="RESTORE")
    {
        EstimateTensorRESTORE();
    }
    if(fitting_mode=="NT2")
    {
        EstimateTensorNT2();
    }
}


int myNLLSSPD_with_derivs(int m, int n, double *p, double *deviates,   double **derivs, void *vars)
{
    int i;
    // m points, n params

    struct vars_struct *v = (struct vars_struct *) vars;
    vnl_matrix<double> *Bmat= v->Bmat;
    vnl_vector<double> *signal= v->signal;
    vnl_vector<double> *weights=v->weights;

    vnl_matrix_fixed<double,3,3> lta; lta.fill(0);
    lta(0,0)=p[1];
    lta(1,0)=p[2];
    lta(2,0)=p[3];
    lta(1,1)=p[4];
    lta(1,2)=p[5];
    lta(2,2)=p[6];

    vnl_matrix<double> lt= lta* lta.transpose();


    for (i=0; i<m; i++)
    {
        double expon = (*Bmat)[i][1]* lt(0,0) + (*Bmat)[i][2]* lt(0,1)+(*Bmat)[i][3]* lt(0,2)+(*Bmat)[i][4]* lt(1,1)+(*Bmat)[i][5]* lt(1,2)+(*Bmat)[i][6]* lt(2,2);

        double est=  exp(expon);
        if(v->useWeights)
            deviates[i] =  ((*signal)[i]- p[0]*est) * (*weights)[i];
        else
            deviates[i] =(*signal)[i]- p[0]*est;

        if (derivs)
        {
            derivs[0][i]= -est;
            est= -est*p[0];

            if(v->useWeights)
            {
                derivs[0][i]*=  (*weights)[i];
                est*=  (*weights)[i];
            }
            derivs[1][i]=  est*  (*Bmat)[i][1] *  2* lta(0,0)  ;
            derivs[2][i]=  est*  (2* (*Bmat)[i][1]*lta(1,0) +  (*Bmat)[i][2]*lta(1,1)   );
            derivs[3][i]=  est*  (2* (*Bmat)[i][1]*lta(2,0) +  (*Bmat)[i][2]*lta(2,1)  +  (*Bmat)[i][3]*lta(2,2)   );
            derivs[4][i]=  est*  ( (*Bmat)[i][2]*lta(1,0) +  2*(*Bmat)[i][4]*lta(1,1)   );
            derivs[5][i]=  est*  ( (*Bmat)[i][2]*lta(2,0) +  2*(*Bmat)[i][4]*lta(2,1)  +  (*Bmat)[i][5]*lta(2,2)   );
            derivs[6][i]=  est*  ( (*Bmat)[i][3]*lta(2,0) +  (*Bmat)[i][5]*lta(2,1)  +  2*(*Bmat)[i][6]*lta(2,2)   );
        }
    }

    return 0;
}


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
            derivs[0][i]= -est;
            est= -est*p[0];

            if(v->useWeights)
            {
                derivs[0][i]*=  (*weights)[i];
                est*=  (*weights)[i];
            }
            derivs[1][i]=  est*  (*Bmat)[i][1];
            derivs[2][i]=  est*  (*Bmat)[i][2];
            derivs[3][i]=  est*  (*Bmat)[i][3];
            derivs[4][i]=  est*  (*Bmat)[i][4];
            derivs[5][i]=  est*  (*Bmat)[i][5];
            derivs[6][i]=  est*  (*Bmat)[i][6];
        }
    }

    return 0;
}



int myNLLS2(int m, int n, double *p, double *deviates,   double **derivs, void *vars)
{
    int i;
    // m points, n params

    struct vars_struct *v = (struct vars_struct *) vars;
    vnl_matrix<double> *Bmat= v->Bmat;
    vnl_vector<double> *signal= v->signal;
    vnl_vector<double> *weights=v->weights;


    for (i=0; i<m; i++)
    {
        double expon1 = (*Bmat)[i][1]* p[0] + (*Bmat)[i][2]* p[1]+(*Bmat)[i][3]* p[2]+(*Bmat)[i][4]* p[3]+(*Bmat)[i][5]* p[4]+(*Bmat)[i][6]* p[5];
        double expon2 = p[8]*( (*Bmat)[i][1] + (*Bmat)[i][4] + (*Bmat)[i][6]);

        double est1=  exp(expon1);
        double est2=  exp(expon2);

        double est= p[6]*(p[7]*est1 + (1-p[7])*est2);


        if(v->useWeights)
            deviates[i] =  ((*signal)[i]- est) * (*weights)[i];
        else
            deviates[i] =(*signal)[i]- est;

        if(derivs)
        {
            if(v->useWeights)
            {
                derivs[7][i]=-p[6]*(est1-est2) * (*weights)[i];
                derivs[6][i]=-(p[7]*est1 + (1-p[7])*est2) * (*weights)[i];
                derivs[5][i]= -p[6]*p[7]*est1*(*Bmat)[i][6] * (*weights)[i];
                derivs[4][i]= -p[6]*p[7]*est1*(*Bmat)[i][5] * (*weights)[i];
                derivs[3][i]= -p[6]*p[7]*est1*(*Bmat)[i][4] * (*weights)[i];
                derivs[2][i]= -p[6]*p[7]*est1*(*Bmat)[i][3] * (*weights)[i];
                derivs[1][i]= -p[6]*p[7]*est1*(*Bmat)[i][2] * (*weights)[i];
                derivs[0][i]= -p[6]*p[7]*est1*(*Bmat)[i][1] * (*weights)[i];

            }
            else
            {
                derivs[7][i]=-p[6]*(est1-est2);
                derivs[6][i]=-(p[7]*est1 + (1-p[7])*est2);
                derivs[5][i]= -p[6]*p[7]*est1*(*Bmat)[i][6];
                derivs[4][i]= -p[6]*p[7]*est1*(*Bmat)[i][5];
                derivs[3][i]= -p[6]*p[7]*est1*(*Bmat)[i][4];
                derivs[2][i]= -p[6]*p[7]*est1*(*Bmat)[i][3];
                derivs[1][i]= -p[6]*p[7]*est1*(*Bmat)[i][2];
                derivs[0][i]= -p[6]*p[7]*est1*(*Bmat)[i][1];
            }

        }
     }
    return 0;
}


int myNLLS_t2(int m, int n, double *p, double *deviates,   double **derivs, void *vars)
{
    int i;
    // m points, n params

    struct vars_struct *v = (struct vars_struct *) vars;
    vnl_matrix<double> *Bmat= v->Bmat;
    vnl_vector<double> *signal= v->signal;
    vnl_vector<double> *weights=v->weights;

    //  S =  p[0] * p[1]* exp(- B : p[2-7])  +   p[0] * (1-p[1])* exp(- B : p[8-13])

    for (i=0; i<m; i++)
    {
        double expon1 = (*Bmat)[i][1]* p[2] + (*Bmat)[i][2]* p[3]+(*Bmat)[i][3]* p[4]+(*Bmat)[i][4]* p[5]+(*Bmat)[i][5]* p[6]+(*Bmat)[i][6]* p[7];
        double expon2 = (*Bmat)[i][1]* p[8] + (*Bmat)[i][2]* p[9]+(*Bmat)[i][3]* p[10]+(*Bmat)[i][4]* p[11]+(*Bmat)[i][5]* p[12]+(*Bmat)[i][6]* p[13];

        double atten_1=  exp(expon1);
        double atten_2=  exp(expon2);

        double est= p[0]*(p[1]*atten_1 + (1-p[1])*atten_2);

        if(v->useWeights)
            deviates[i] =  ((*signal)[i]- est) * (*weights)[i];
        else
            deviates[i] =(*signal)[i]- est;


        if(derivs)
        {
            if(v->useWeights)
            {
                derivs[0][i]=  -(p[1]*atten_1  + (1-p[1])*atten_2)* (*weights)[i];
                derivs[1][i]=  -(p[0]*(atten_1  -atten_2))* (*weights)[i];

                derivs[2][i]=  -(p[0]*p[1]*((*Bmat)[i][1]* atten_1))* (*weights)[i];
                derivs[3][i]=  -(p[0]*p[1]*((*Bmat)[i][2]* atten_1))* (*weights)[i];
                derivs[4][i]=  -(p[0]*p[1]*((*Bmat)[i][3]* atten_1))* (*weights)[i];
                derivs[5][i]=  -(p[0]*p[1]*((*Bmat)[i][4]* atten_1))* (*weights)[i];
                derivs[6][i]=  -(p[0]*p[1]*((*Bmat)[i][5]* atten_1))* (*weights)[i];
                derivs[7][i]=  -(p[0]*p[1]*((*Bmat)[i][6]* atten_1))* (*weights)[i];

                derivs[8][i]=   -(p[0]*(1-p[1])*((*Bmat)[i][1]* atten_2))* (*weights)[i];
                derivs[9][i]=   -(p[0]*(1-p[1])*((*Bmat)[i][2]* atten_2))* (*weights)[i];
                derivs[10][i]=  -(p[0]*(1-p[1])*((*Bmat)[i][3]* atten_2))* (*weights)[i];
                derivs[11][i]=  -(p[0]*(1-p[1])*((*Bmat)[i][4]* atten_2))* (*weights)[i];
                derivs[12][i]=  -(p[0]*(1-p[1])*((*Bmat)[i][5]* atten_2))* (*weights)[i];
                derivs[13][i]=  -(p[0]*(1-p[1])*((*Bmat)[i][6]* atten_2))* (*weights)[i];


            }
            else
            {
                derivs[0][i]=  -(p[1]*atten_1  + (1-p[1])*atten_2);
                derivs[1][i]=  -(p[0]*(atten_1  -atten_2));

                derivs[2][i]=  -(p[0]*p[1]*((*Bmat)[i][1]* atten_1));
                derivs[3][i]=  -(p[0]*p[1]*((*Bmat)[i][2]* atten_1));
                derivs[4][i]=  -(p[0]*p[1]*((*Bmat)[i][3]* atten_1));
                derivs[5][i]=  -(p[0]*p[1]*((*Bmat)[i][4]* atten_1));
                derivs[6][i]=  -(p[0]*p[1]*((*Bmat)[i][5]* atten_1));
                derivs[7][i]=  -(p[0]*p[1]*((*Bmat)[i][6]* atten_1));

                derivs[8][i]=   -(p[0]*(1-p[1])*((*Bmat)[i][1]* atten_2));
                derivs[9][i]=   -(p[0]*(1-p[1])*((*Bmat)[i][2]* atten_2));
                derivs[10][i]=  -(p[0]*(1-p[1])*((*Bmat)[i][3]* atten_2));
                derivs[11][i]=  -(p[0]*(1-p[1])*((*Bmat)[i][4]* atten_2));
                derivs[12][i]=  -(p[0]*(1-p[1])*((*Bmat)[i][5]* atten_2));
                derivs[13][i]=  -(p[0]*(1-p[1])*((*Bmat)[i][6]* atten_2));

            }

        }

     }
    return 0;
}



void DTIModel::EstimateTensorSPD()
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
        (*stream)<<"Computing Tensors NLLS SPD ..."<<std::endl;
    else
        std::cout<<"Computing Tensors NLLS SPD ..."<<std::endl;

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
                        int vol_id= all_indices[v];
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
                auto curr_Bmatrix = getCurrentBmatrix(ind3,curr_all_indices);

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
                    curr_design_matrix(vol,1)= -curr_Bmatrix(vol,0)/1000.;
                    curr_design_matrix(vol,2)= -curr_Bmatrix(vol,1)/1000.;
                    curr_design_matrix(vol,3)= -curr_Bmatrix(vol,2)/1000.;
                    curr_design_matrix(vol,4)= -curr_Bmatrix(vol,3)/1000.;
                    curr_design_matrix(vol,5)= -curr_Bmatrix(vol,4)/1000.;
                    curr_design_matrix(vol,6)= -curr_Bmatrix(vol,5)/1000.;

                } //for vol


                DTImageType::PixelType dt_vec= this->output_img->GetPixel(ind3);
                vnl_matrix_fixed<double,3,3> Dmat;
                Dmat(0,0)=dt_vec[0]*1000;
                Dmat(1,0)=dt_vec[1]*1000;
                Dmat(0,1)=dt_vec[1]*1000;
                Dmat(2,0)=dt_vec[2]*1000;
                Dmat(0,2)=dt_vec[2]*1000;
                Dmat(1,1)=dt_vec[3]*1000;
                Dmat(1,2)=dt_vec[4]*1000;
                Dmat(2,1)=dt_vec[4]*1000;
                Dmat(2,2)=dt_vec[5]*1000;

                vnl_symmetric_eigensystem<double> eig(Dmat);
                if(eig.D(0,0)<=0)
                    eig.D(0,0)=1E-8;
                if(eig.D(1,1)<=0)
                    eig.D(1,1)=1E-8;
                if(eig.D(2,2)<=0)
                    eig.D(2,2)=1E-8;
                vnl_matrix_fixed<double,3,3> Dmat_corr= eig.recompose();

                vnl_cholesky mchol(Dmat_corr);
                vnl_matrix<double> lt= mchol.lower_triangle();


                double p[7];
                p[0]= this->A0_img->GetPixel(ind3);
                p[1]= lt(0,0);
                p[2]= lt(1,0);
                p[3]= lt(2,0);
                p[4]= lt(1,1);
                p[5]= lt(1,2);
                p[6]= lt(2,2);

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

                int status = mpfit(myNLLSSPD_with_derivs, curr_design_matrix.rows(), 7, p, pars, &config, (void *) &my_struct, &my_results_struct);

                double degrees_of_freedom= curr_all_indices.size()-7;
                CS_img->SetPixel(ind3,my_results_struct.bestnorm/degrees_of_freedom);

                vnl_matrix_fixed<double,3,3> mmat;
                mmat.fill(0);
                mmat(0,0)=p[1];
                mmat(1,0)=p[2];
                mmat(2,0)=p[3];
                mmat(1,1)=p[4];
                mmat(1,2)=p[5];
                mmat(2,2)=p[6];

                mmat=mmat * mmat.transpose();

                dt_vec[0]= mmat(0,0)/1000.;
                dt_vec[1]= mmat(1,0)/1000.;
                dt_vec[2]= mmat(2,0)/1000.;
                dt_vec[3]= mmat(1,1)/1000.;
                dt_vec[4]= mmat(2,1)/1000.;
                dt_vec[5]= mmat(2,2)/1000.;


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





void DTIModel::EstimateTensorNT2()
{
    EstimateTensorWLLS();

    {
        using DupType= itk::ImageDuplicator<DTImageType>;
        DupType::Pointer dup= DupType::New();
        dup->SetInputImage(this->output_img);
        dup->Update();
        this->flow_tensor_img=dup->GetOutput();
        DTImageType::PixelType zer;
        zer.Fill(0);
        this->flow_tensor_img->FillBuffer(zer);
    }


    VF_img=ImageType3D::New();
    VF_img->SetRegions(A0_img->GetLargestPossibleRegion());
    VF_img->Allocate();
    VF_img->SetSpacing(A0_img->GetSpacing());
    VF_img->SetOrigin(A0_img->GetOrigin());
    VF_img->SetDirection(A0_img->GetDirection());
    VF_img->FillBuffer(0.);


    float free_water_adc_value=this->nt2c1maxadc/1000.;
    double MAX_TR=-1;
    itk::ImageRegionIteratorWithIndex<DTImageType> it(output_img,output_img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        DTImageType::PixelType curr_tens= it.Get();
        double MD=( curr_tens[0]+curr_tens[3]+curr_tens[5])/3.*1000;
        if(MD>free_water_adc_value && MD > MAX_TR)
            MAX_TR=MD;
    }
    if(MAX_TR<0)
        MAX_TR=2*free_water_adc_value;

    double min_flow_diffusivity=this->nt2c2minadc/1000.;
    double flow_diffusivity=2*min_flow_diffusivity;


    DTImageType::PixelType free_tens;
    free_tens[0]=free_water_adc_value/1000.;
    free_tens[3]=free_water_adc_value/1000.;
    free_tens[5]=free_water_adc_value/1000.;
    free_tens[1]=0;
    free_tens[2]=0;
    free_tens[4]=0;

    DTImageType::PixelType flow_tens;
    flow_tens[0]=flow_diffusivity/1000.;
    flow_tens[3]=flow_diffusivity/1000.;
    flow_tens[5]=flow_diffusivity/1000.;
    flow_tens[1]=0;
    flow_tens[2]=0;
    flow_tens[4]=0;




    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType index=it.GetIndex();

        if(mask_img && mask_img->GetPixel(index)==0)
        {
            continue;
        }


        DTImageType::PixelType curr_tens= it.Get();
        double MD=( curr_tens[0]+curr_tens[3]+curr_tens[5])/3.;
        this->flow_tensor_img->SetPixel(index,flow_tens);
        if(MD*1000 > free_water_adc_value)
        {
            VF_img->SetPixel(index,0.2);
            it.Set(free_tens);
        }
        else
        {
            DTImageType::PixelType pix= it.Get();
            VF_img->SetPixel(index,0.99);
        }

    }


    int Nvols=Bmatrix.rows();

    std::vector<int> all_indices;
    if(indices_fitting.size()>0)
        all_indices= indices_fitting;
    else
    {
        for(int ma=0;ma<Nvols;ma++)
            all_indices.push_back(ma);
    }


    std::cout<<"Computing NT2 Dual Tensors ..."<<std::endl;

    ImageType3D::SizeType size = dwi_data[0]->GetLargestPossibleRegion().GetSize();

    mp_config_struct config;
    config.maxiter=200;
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

        double p[14];
        mp_par pars[14];
        memset(&pars[0], 0, sizeof(pars));

        //  S =  p[0] * p[1]* exp(- B : p[2-7])  +   p[0] * (1-p[1])* exp(- B : p[8-13])


   //   pars[0].parname="Am";
        pars[0].limited[0]=1;
        pars[0].limited[1]=0;
        pars[0].limits[0]=0.0;
        pars[0].limits[1]=0;
        pars[0].side=0;
        pars[0].fixed=1;

  //    pars[1].parname="vf";  this is vf  for parenchymal tensor
        pars[1].limited[0]=1;
        pars[1].limited[1]=1;
        pars[1].limits[0]=0.;
        pars[1].limits[1]=1.;
        pars[1].side=0;

      //pars[2].parname="D1_XX";
        pars[2].limited[0]=1;
        pars[2].limited[1]=1;
        pars[2].limits[0]=0.00001;
        pars[2].limits[1]=free_water_adc_value;
        pars[2].side=0;

        //pars[3].parname="D1_XY";
        pars[3].limited[0]=0;
        pars[3].limited[1]=0;
        pars[3].limits[0]=0;
        pars[3].limits[1]=0;
        pars[3].side=0;

        //pars[4].parname="D1_XZ";
        pars[4].limited[0]=0;
        pars[4].limited[1]=0;
        pars[4].limits[0]=0;
        pars[4].limits[1]=0;
        pars[4].side=0;

       //pars[5].parname="D1_YY";
        pars[5].limited[0]=1;
        pars[5].limited[1]=1;
        pars[5].limits[0]=0.00001;
        pars[5].limits[1]=free_water_adc_value;
        pars[5].side=0;

        //pars[6].parname="D1_YZ";
        pars[6].limited[0]=0;
        pars[6].limited[1]=0;
        pars[6].limits[0]=0;
        pars[6].limits[1]=0;
        pars[6].side=0;

        //pars[5].parname="D1_ZZ";
        pars[7].limited[0]=1;
        pars[7].limited[1]=1;
        pars[7].limits[0]=0.00001;
        pars[7].limits[1]=free_water_adc_value;
        pars[7].side=0;



        //pars[2].parname="D2_XX";
        pars[8].limited[0]=1;
        pars[8].limited[1]=0;
        pars[8].limits[0]=0.00001;
        pars[8].limits[0]=min_flow_diffusivity;
        pars[8].limits[1]=8.*flow_diffusivity;
        pars[8].side=0;

        //pars[3].parname="D2_XY";
        pars[9].limited[0]=0;
        pars[9].limited[1]=0;
        pars[9].limits[0]=0;
        pars[9].limits[1]=0;
        pars[9].side=0;

        //pars[4].parname="D2_XZ";
        pars[10].limited[0]=0;
        pars[10].limited[1]=0;
        pars[10].limits[0]=0;
        pars[10].limits[1]=0;
        pars[10].side=0;

        //pars[5].parname="D2_YY";
        pars[11].limited[0]=1;
        pars[11].limited[1]=0;
        pars[11].limits[0]=min_flow_diffusivity;
        pars[11].limits[1]=8.*flow_diffusivity;
        pars[11].side=0;

        //pars[6].parname="D2_YZ";
        pars[12].limited[0]=0;
        pars[12].limited[1]=0;
        pars[12].limits[0]=0;
        pars[12].limits[1]=0;
        pars[12].side=0;

        //pars[5].parname="D2_ZZ";
        pars[13].limited[0]=1;
        pars[13].limited[1]=0;
        pars[13].limits[0]=min_flow_diffusivity;
        pars[13].limits[1]=8.*flow_diffusivity;
        pars[13].side=0;


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
                        int vol_id= all_indices[v];
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
                auto curr_Bmatrix = getCurrentBmatrix(ind3,curr_all_indices);


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
               //     if(curr_Bmatrix(vol,0) +  curr_Bmatrix(vol,3) + curr_Bmatrix(vol,5) >10)
                    {
                        curr_design_matrix(vol,1)= -curr_Bmatrix(vol,0)/1000.;
                        curr_design_matrix(vol,2)= -curr_Bmatrix(vol,1)/1000.;
                        curr_design_matrix(vol,3)= -curr_Bmatrix(vol,2)/1000.;
                        curr_design_matrix(vol,4)= -curr_Bmatrix(vol,3)/1000.;
                        curr_design_matrix(vol,5)= -curr_Bmatrix(vol,4)/1000.;
                        curr_design_matrix(vol,6)= -curr_Bmatrix(vol,5)/1000.;
                    }
                //    else
                  //  {
                    //    curr_design_matrix(vol,1)= 0;
                  //      curr_design_matrix(vol,2)= 0;
                  //      curr_design_matrix(vol,3)= 0;
                  //      curr_design_matrix(vol,4)= 0;
                  //      curr_design_matrix(vol,5)= 0;
                  //      curr_design_matrix(vol,6)= 0;

               //     }

                } //for vol

                DTImageType::PixelType dt_vec= output_img->GetPixel(ind3);
                p[0]= A0_img->GetPixel(ind3);
                p[1]= VF_img->GetPixel(ind3);

                for(int pp=2;pp<8;pp++)
                {
                    p[pp]= dt_vec[pp-2]*1000.;
                    if(pars[pp].limited[0])
                    {
                        if(p[pp] < pars[pp].limits[0])
                            p[pp]=pars[pp].limits[0]+0.000001;
                    }
                    if(pars[pp].limited[1])
                    {
                        if(p[pp] > pars[pp].limits[1])
                            p[pp]=pars[pp].limits[1]-0.000001;
                    }
                }
                dt_vec=this->flow_tensor_img->GetPixel(ind3);
                for(int pp=8;pp<14;pp++)
                {
                    p[pp]= dt_vec[pp-8]*1000.;
                    if(pars[pp].limited[0])
                    {
                        if(p[pp] < pars[pp].limits[0])
                            p[pp]=pars[pp].limits[0]*1.5;
                    }
                    if(pars[pp].limited[1])
                    {
                        if(p[pp] > pars[pp].limits[1])
                            p[pp]=pars[pp].limits[1]*1.5;
                    }
                }


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


                int status = mpfit(myNLLS_t2, curr_design_matrix.rows(), 14, p, pars, &config, (void *) &my_struct, &my_results_struct);

                double degrees_of_freedom= curr_all_indices.size()-14;
                CS_img->SetPixel(ind3,my_results_struct.bestnorm/degrees_of_freedom);

                A0_img->SetPixel(ind3,p[0]);
                VF_img->SetPixel(ind3,p[1]);

                dt_vec[0]= p[2]/1000.;
                dt_vec[1]= p[3]/1000.;
                dt_vec[2]= p[4]/1000.;
                dt_vec[3]= p[5]/1000.;
                dt_vec[4]= p[6]/1000.;
                dt_vec[5]= p[7]/1000.;
                output_img->SetPixel(ind3,dt_vec);

                dt_vec[0]= p[8]/1000.;
                dt_vec[1]= p[9]/1000.;
                dt_vec[2]= p[10]/1000.;
                dt_vec[3]= p[11]/1000.;
                dt_vec[4]= p[12]/1000.;
                dt_vec[5]= p[13]/1000.;
                this->flow_tensor_img->SetPixel(ind3,dt_vec);

            } //for i
        } //for j
    } //for k


}


double DTIModel::check_condition_number(std::vector<int> &outlier_index, vnl_matrix<double> Bmatrix)
{
    vnl_svd<double> my_svd(Bmatrix);
    int N= my_svd.W().rows();


    double cn_original = my_svd.W()(0,0) / my_svd.W()(N-1,N-1);


    for(int i=0; i<outlier_index.size();i++)
    {
        int vol = outlier_index[i];
        Bmatrix(vol,0)=0;
        Bmatrix(vol,1)=0;
        Bmatrix(vol,2)=0;
        Bmatrix(vol,3)=0;
        Bmatrix(vol,4)=0;
        Bmatrix(vol,5)=0;
    }

    vnl_svd<double> my_svd2(Bmatrix);
    double cn = my_svd2.W()(0,0) / my_svd2.W()(N-1,N-1);

    if(cn / cn_original > 2)
    {
        std::vector<int> dummy;
        outlier_index =dummy;
    }

    return cn;
}



std::vector<int> DTIModel::check_gradient_direction(std::vector<int> outlier_index_original,vnl_matrix<double> Bmatrix)
{

    vnl_matrix<double> dir6(6,3);
    dir6(0,0)=  0.79546387; dir6(0,1)= -0.10374812; dir6(0,2)= 0.59705407 ;
    dir6(1,0)= -0.86270267; dir6(1,1)= -0.37933154; dir6(1,2)=  0.33444235;
    dir6(2,0)=  0.05833067; dir6(2,1)= -0.8499489 ; dir6(2,2)=  0.5236262;
    dir6(3,0)= -0.16712488; dir6(3,1)=  0.06830884; dir6(3,2)=  0.98356656;
    dir6(4,0)=  0.33000462; dir6(4,1)=  0.82804632; dir6(4,2)=  0.45325075;
    dir6(5,0)= -0.6947986 ; dir6(5,1)=  0.65772637; dir6(5,2)=  0.29094833;


    int Nvols = Bmatrix.rows();

    vnl_matrix<double> bvecs(Nvols,3);
    vnl_matrix<double> bvals(Nvols,1);


    for(int i=0;i<Nvols;i++)
    {
           vnl_vector<double> bmat_vec= Bmatrix.get_row(i);

           vnl_matrix_fixed<double,3,3> bmat;
           bmat(0,0)= bmat_vec[0];
           bmat(0,1)= bmat_vec[1]/2;
           bmat(1,0)= bmat_vec[1]/2;
           bmat(0,2)= bmat_vec[2]/2;
           bmat(2,0)= bmat_vec[2]/2;
           bmat(1,1)= bmat_vec[3];
           bmat(1,2)= bmat_vec[4]/2;
           bmat(2,1)= bmat_vec[4]/2;
           bmat(2,2)= bmat_vec[5];

           vnl_symmetric_eigensystem<double> eig(bmat);

           bvals(i,0)= eig.D(2,2);
           vnl_vector<double> cbvec= eig.V.get_column(2);
           bvecs(i,0)= -cbvec[0];
           bvecs(i,1)= cbvec[1];
           bvecs(i,2)= cbvec[2];

           if(bvecs(i,2)<0)
           {
               bvecs(i,0)=-bvecs(i,0);
               bvecs(i,1)=-bvecs(i,1);
               bvecs(i,2)=-bvecs(i,2);
           }
    }


    int nd=0;

    vnl_matrix<double> direction_temp = bvecs;
    std::vector<int> outlier_index_temp = outlier_index_original;

    for(int i=0;i<outlier_index_original.size();i++)
    {
        direction_temp(outlier_index_original[i],0)=0;
        direction_temp(outlier_index_original[i],1)=0;
        direction_temp(outlier_index_original[i],2)=0;

    }

    vnl_vector<double> sum_of_vector(6,0);


    for(int i=0;i<6;i++)
    {
        for(int j=0;j<Nvols;j++)
        {
            vnl_vector<double> vec1 =  dir6.get_row(i);
            vnl_vector<double> vec2=  direction_temp.get_row(j);

            if((vec1.magnitude()!=0) && (vec2.magnitude()!=0) )
            {
                double cos_theta = fabs(vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]    );
                sum_of_vector[i]+= cos_theta;
                nd++;
            }
        }
    }


    std::vector<int> dummy;

    if(sum_of_vector.min_value() < 3.23)
    {
        return dummy;
    }

    return outlier_index_original;

}





DTImageType::PixelType  DTIModel::RobustFit(vnl_matrix<double> design_matrix,vnl_vector<double> signal, double initial_A0_estimate,
                                  DTImageType::PixelType initial_dt_estimate1,  vnl_vector<double> sigstdev,
                                  std::vector<int> &b0_indices,double &A0_estimate,std::vector<int> &outlier_index,double &CS_val,float THR)
{
    vnl_vector<double> initial_dt_estimate(6);
    initial_dt_estimate[0]=initial_dt_estimate1[0];
    initial_dt_estimate[1]=initial_dt_estimate1[1];
    initial_dt_estimate[2]=initial_dt_estimate1[2];
    initial_dt_estimate[3]=initial_dt_estimate1[3];
    initial_dt_estimate[4]=initial_dt_estimate1[4];
    initial_dt_estimate[5]=initial_dt_estimate1[5];

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

    DTImageType::PixelType new_dt ;

    vars_struct my_struct;
    my_struct.useWeights=true;


    int Nvols = design_matrix.rows();

    mp_result_struct my_results_struct;
    vnl_vector<double> my_resids(Nvols);
    my_results_struct.resid= my_resids.data_block();
    my_results_struct.xerror=NULL;
    my_results_struct.covar=NULL;


    //Use analytical derivatives in LM optimizer
    double p[7];
    mp_par pars[7];
    memset(&pars[0], 0, sizeof(pars));
    pars[0].side=3;
    pars[1].side=3;
    pars[2].side=3;
    pars[3].side=3;
    pars[4].side=3;
    pars[5].side=3;
    pars[6].side=3;


    int degrees_of_freedom = design_matrix.rows() -7;
    double robust_thresh_chi_confidence=3.0;
    double robust_thresh_chi = 1.0 + robust_thresh_chi_confidence*sqrt(2.0/degrees_of_freedom);


    double tensor_improvement = 0.005;
    int max_iteration=40;
    int weightfun=1;
    double threshold = THR;


    vnl_vector<double> signal_estimate(Nvols);
    vnl_vector<double> init_weight(Nvols);
    vnl_vector<double> idxyfit(Nvols);


    for(int v=0;v<Nvols;v++)
    {
      double exp_term = design_matrix(v,1)*initial_dt_estimate[0]*1000.+
                        design_matrix(v,2)*initial_dt_estimate[1]*1000.+
                        design_matrix(v,3)*initial_dt_estimate[2]*1000.+
                        design_matrix(v,4)*initial_dt_estimate[3]*1000.+
                        design_matrix(v,5)*initial_dt_estimate[4]*1000.+
                        design_matrix(v,6)*initial_dt_estimate[5]*1000.;

      signal_estimate[v] = initial_A0_estimate * exp(exp_term);
      init_weight[v]= 1./sigstdev[v]/sigstdev[v];
    }
    vnl_vector<double> weight= init_weight;

    vnl_vector<double> yresidue = signal - signal_estimate;
    std::vector<float> yres_std;
    for(int v=0;v<Nvols;v++)
    {
      yres_std.push_back(yresidue[v]);
    }
    double med_yres= median(yres_std);
    std::vector<float> abs_diff;
    for(int v=0;v<Nvols;v++)
    {
      abs_diff.push_back(fabs(yresidue[v]-med_yres));
    }

    double c= 1.4826 * median(abs_diff);

    double original_df =  degrees_of_freedom;
    double df =  degrees_of_freedom;

    int rcounter=1;
    bool converged=0;

    double total_weight=0,total_init_weight=0;

    vnl_vector<double> result = initial_dt_estimate;

    double curr_A0_estimate= initial_A0_estimate;

      while(!converged)
      {
          vnl_vector<double> tensor_in=result;
          double total_yres = 0;
          for(int v=0;v<Nvols;v++)
          {
              total_yres +=   (1./(yresidue[v]*yresidue[v] + sigstdev[v]*sigstdev[v]));
          }


          for(int v=0;v<Nvols;v++)
          {
              weight[v] = (1./(yresidue[v]*yresidue[v] + sigstdev[v]*sigstdev[v]))* original_df/total_yres;
          }

         for(int v=0;v<b0_indices.size();v++)
         {
             weight[ b0_indices[v] ] =1;
         }


          //SET the weights

          total_weight = weight.sum();
          total_init_weight = init_weight.sum();

          for(int v=0;v<Nvols;v++)
          {
              weight[v] = weight[v] * (total_init_weight/total_weight);
          }



          for(int mama=0;mama<weight.size();mama++)
          {
              weight[mama]=sqrt(weight[mama]);
          }

          my_struct.signal= &signal;
          my_struct.weights=&weight;
          my_struct.Bmat= &design_matrix;

          p[0]= curr_A0_estimate;
          p[1]= tensor_in[0]*1000.;
          p[2]= tensor_in[1]*1000.;
          p[3]= tensor_in[2]*1000.;
          p[4]= tensor_in[3]*1000.;
          p[5]= tensor_in[4]*1000.;
          p[6]= tensor_in[5]*1000.;


          // do a new fitting with new weights
          int status = mpfit(myNLLS_with_derivs, design_matrix.rows(), 7, p, pars, &config, (void *) &my_struct, &my_results_struct);



          double new_A0_estimate = p[0];
          for(int v=0;v<Nvols;v++)
          {
              double exp_term = design_matrix(v,1)*p[1]+
                                design_matrix(v,2)*p[2]+
                                design_matrix(v,3)*p[3]+
                                design_matrix(v,4)*p[4]+
                                design_matrix(v,5)*p[5]+
                                design_matrix(v,6)*p[6];

              signal_estimate[v] = new_A0_estimate * exp(exp_term);

          }



          result[0]=p[1]/1000.;
          result[1]=p[2]/1000.;
          result[2]=p[3]/1000.;
          result[3]=p[4]/1000.;
          result[4]=p[5]/1000.;
          result[5]=p[6]/1000.;



          yresidue = signal - signal_estimate;
          vnl_vector<double> tensor_out=result;

          vnl_vector<double> din(7,0),dout(7,0);
          din[0]=curr_A0_estimate;
          din[1]=tensor_in[0]*1000.;
          din[2]=tensor_in[1]*1000.;
          din[3]=tensor_in[2]*1000.;
          din[4]=tensor_in[3]*1000.;
          din[5]=tensor_in[4]*1000.;
          din[6]=tensor_in[5]*1000.;

          dout[0]=new_A0_estimate;
          dout[1]=tensor_out[0]*1000.;
          dout[2]=tensor_out[1]*1000.;
          dout[3]=tensor_out[2]*1000.;
          dout[4]=tensor_out[3]*1000.;
          dout[5]=tensor_out[4]*1000.;
          dout[6]=tensor_out[5]*1000.;

          vnl_vector<double> diff= din-dout;
          double improvement = diff.magnitude()/din.magnitude();



          curr_A0_estimate=new_A0_estimate;

          ++rcounter;

          if(improvement < tensor_improvement)
              converged = true;

      } //while converged


      double current_threshold = threshold;
      vnl_vector<double> check_array(Nvols);

      for(int v=0;v<Nvols;v++)
      {
          check_array[v] = fabs(yresidue[v]) -current_threshold*sigstdev[v];
      }

      for(int v=0;v<b0_indices.size();v++)
      {
          check_array[b0_indices[v]] = -1;
      }



      int Nb0s = b0_indices.size();
      int nn = 0;



      std::vector<int> outlier_index_original;

      int Noutliers=0;
      for(int v=0;v<Nvols;v++)
      {
          if(check_array[v] > 0)
          {
              outlier_index_original.push_back(v);
              Noutliers++;
          }
      }


      if(Noutliers>0)
      {
          vnl_matrix<double> Bmatrix = design_matrix.extract(design_matrix.rows(),6,0,1);
          Bmatrix = -Bmatrix;

          double cn = check_condition_number(outlier_index_original,Bmatrix);

          int new_Noutliers = outlier_index_original.size();

          if(new_Noutliers>0)
          {
              outlier_index_original = check_gradient_direction(outlier_index_original,Bmatrix);
          }
      }



      if(outlier_index_original.size()!=0)
      {
          vnl_vector<double> signal2(Nvols-outlier_index_original.size()), weight2(Nvols-outlier_index_original.size());
          vnl_matrix<double> design_matrix2(Nvols-outlier_index_original.size(),7);


          int cnt = -1;
          for(int v=0;v<Nvols;v++)
          {
              bool is_outlier = false;
              for(int v2=0;v2<outlier_index_original.size();v2++)
              {
                  if(v == outlier_index_original[v2])
                  {
                      is_outlier = true;
                      break;
                  }
              }

              if(!is_outlier)
              {
                  cnt++;

                  signal2[cnt] = signal[v];
                  design_matrix2.set_row(cnt,design_matrix.get_row(v));
                  weight2[cnt] = 1./ sigstdev[v]/sigstdev[v];
              }
          }


          for(int mama=0;mama<weight2.size();mama++)
          {
              weight2[mama] = sqrt(weight2[mama]);
          }

          my_struct.signal = &signal2;
          my_struct.weights = &weight2;
          my_struct.Bmat = &design_matrix2;

          p[0]= curr_A0_estimate;
          p[1]= result[0]*1000.;
          p[2]= result[1]*1000.;
          p[3]= result[2]*1000.;
          p[4]= result[3]*1000.;
          p[5]= result[4]*1000.;
          p[6]= result[5]*1000.;


          double  df= design_matrix2.rows()-7;

          int status = mpfit(myNLLS_with_derivs, design_matrix2.rows(), 7, p, pars, &config, (void *) &my_struct, &my_results_struct);

          CS_val = my_results_struct.bestnorm/df;

          A0_estimate=p[0];

          new_dt[0]=p[1]/1000.;
          new_dt[1]=p[2]/1000.;
          new_dt[2]=p[3]/1000.;
          new_dt[3]=p[4]/1000.;
          new_dt[4]=p[5]/1000.;
          new_dt[5]=p[6]/1000.;

      }
      else
      {
          CS_val = -1;
          A0_estimate=initial_A0_estimate;
          new_dt[0]= initial_dt_estimate[0];
          new_dt[1]= initial_dt_estimate[1];
          new_dt[2]= initial_dt_estimate[2];
          new_dt[3]= initial_dt_estimate[3];
          new_dt[4]= initial_dt_estimate[4];
          new_dt[5]= initial_dt_estimate[5];
      }

      outlier_index = outlier_index_original;

      return new_dt;


}





float DTIModel::ComputeMedianB0MeanStDev(std::vector<int> b0_indices, float &median_signal_b0_std)
{
    std::vector<ImageType3D::Pointer> b0_images;
    for(int v=0;v<b0_indices.size();v++)
    {
        ImageType3D::Pointer curr_b0_image = dwi_data[b0_indices[v]];
        b0_images.push_back(curr_b0_image);
    }


    ImageType3D::Pointer b0_image = b0_images[0];

    itk::ImageRegionIteratorWithIndex <ImageType3D> it(b0_image,b0_image->GetLargestPossibleRegion());
    it.GoToBegin();
    std::vector<float> b0vals,b0stdevs;
    while(!it.IsAtEnd())
    {
       ImageType3D::IndexType index = it.GetIndex();
       float mask_val = mask_img->GetPixel(index);
       if(mask_val !=0)
       {
           double avg_b0_val=0;
           for(int v=0;v<b0_indices.size();v++)
           {
               avg_b0_val += b0_images[v]-> GetPixel(index);
           }
           avg_b0_val/= b0_indices.size();
           b0vals.push_back(avg_b0_val);

           double var=0;
           for(int v=0;v<b0_indices.size();v++)
           {
                double val =b0_images[v]-> GetPixel(index);
                var+=(val - avg_b0_val)*(val - avg_b0_val);
           }

           double avg_stdev= sqrt(var/b0_indices.size());
           b0stdevs.push_back(avg_stdev);

       }

       ++it;
   }

   float median_signal_b0 = median(b0vals);
   median_signal_b0_std= median(b0stdevs);

   return median_signal_b0;

}



void DTIModel::EstimateTensorRESTORE()
{

    if(!noise_img)
    {
        std::cerr<<"RESTORE fitting requires a noise image to be present. Currently, it does not exist..."<<std::endl;
        std::cerr<<"Exiting..."<<std::endl;
        exit(EXIT_FAILURE);
    }
    if(!mask_img)
    {
        std::cerr<<"RESTORE fitting requires a mask image to be present. Currently, it is not provided..."<<std::endl;
        std::cerr<<"Exiting..."<<std::endl;
        exit(EXIT_FAILURE);
    }



    // compute the median noise value computed by MPPCA  within a brain mask
    float median_noise_val=0;
    {
        std::vector<float> noise_vals;
        itk::ImageRegionIteratorWithIndex<ImageType3D> it2(noise_img,noise_img->GetLargestPossibleRegion());
        for(it2.GoToBegin();!it2.IsAtEnd();++it2)
        {
            ImageType3D::IndexType ind3= it2.GetIndex();
            if(mask_img->GetPixel(ind3))
                noise_vals.push_back(it2.Get());
        }
        median_noise_val= median(noise_vals);
    }


    float THR=5.;   //how many sigmas is considered outlier


    // Estimate the tensor with nonlinear regression first
    EstimateTensorNLLS();



    //depending on which bvalues to use we might not need all volumes
    int Nvols=Bmatrix.rows();
    std::vector<int> all_indices;
    if(indices_fitting.size()>0)
        all_indices= indices_fitting;
    else
    {
        for(int ma=0;ma<Nvols;ma++)
            all_indices.push_back(ma);
    }



    std::cout<<"Computing Tensors NLLS RESTORE..."<<std::endl;

    //Create the weight image if it is not provided
    if(this->weight_imgs.size()==0)
    {
        this->weight_imgs.resize(Nvols);
        for(int v=0;v<Nvols;v++)
        {
            this->weight_imgs[v]=ImageType3D::New();
            this->weight_imgs[v]->SetRegions(A0_img->GetLargestPossibleRegion());
            this->weight_imgs[v]->Allocate();
            this->weight_imgs[v]->SetSpacing(A0_img->GetSpacing());
            this->weight_imgs[v]->SetOrigin(A0_img->GetOrigin());
            this->weight_imgs[v]->SetDirection(A0_img->GetDirection());
            this->weight_imgs[v]->FillBuffer(1.);
        }
    }



    ImageType3D::SizeType size = dwi_data[0]->GetLargestPossibleRegion().GetSize();


    //Optimization parameters
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


    //treat b=0s a little differently.  compute the median standard deviation of b0s
    vnl_vector<double> bvals= Bmatrix.get_column(0)+Bmatrix.get_column(3)+Bmatrix.get_column(5);
    double min_bval= bvals.min_value();
    std::vector<int> b0_indices;
    for(int v=0;v<all_indices.size();v++)
    {
        int vol = all_indices[v];
        double bval = bvals[vol];
        if ( fabs(bval-min_bval) < (0.05*min_bval+2)  )
            b0_indices.push_back(v);
    }
    float median_signal_b0_std=1;
    float median_signal_b0 = ComputeMedianB0MeanStDev(b0_indices, median_signal_b0_std);
    float noise_kappa=1.;




    #pragma omp parallel for
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

                if(mask_img && mask_img->GetPixel(ind3)==0)
                    continue;

                //compute the chi-squared value of the initial fit.
                //nominator comes from the residuals
                //denominator comes from the estimated noise by MPPCA
                double CS_val = CS_img->GetPixel(ind3);
                double curr_noise_val=noise_img->GetPixel(ind3);
                if(curr_noise_val<median_noise_val/3.)
                    curr_noise_val=median_noise_val;
                CS_val/= curr_noise_val * curr_noise_val;



                std::vector<int> curr_all_indices;
                if(this->weight_imgs.size())
                {
                    for(int v=0;v<all_indices.size();v++)
                    {
                        int vol_id= all_indices[v];
                        if(this->weight_imgs[vol_id]->GetPixel(ind3)>0)
                            curr_all_indices.push_back(vol_id);
                    }
                }
                else
                {
                    curr_all_indices=all_indices;
                }

                auto curr_Bmatrix = getCurrentBmatrix(ind3,curr_all_indices);


                vnl_vector<double> signal(curr_all_indices.size());
                vnl_vector<double> weights(curr_all_indices.size(),1.);
                vnl_matrix<double> curr_design_matrix(curr_all_indices.size(),7);
                vnl_vector<double> curr_sigstdev(curr_all_indices.size(),1.);

                int degrees_of_freedom= curr_all_indices.size()-7;
                double robust_thresh_chi_confidence=3.0;
                double robust_thresh_chi = 1.0 + robust_thresh_chi_confidence*sqrt(2.0/degrees_of_freedom) ;

                if(CS_val <= robust_thresh_chi )
                    continue;


                bool broken_voxel=false;
                vnl_vector<double> sigstdev(curr_all_indices.size(),1.);
                for(int vol=0;vol<curr_all_indices.size();vol++)
                {
                    curr_sigstdev[vol]=curr_noise_val;

                    int vol_id= curr_all_indices[vol];
                    double nval = dwi_data[vol_id]->GetPixel(ind3);


                    //fix artifactual signals
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
                            broken_voxel=true;
                        }
                    }


                    signal[vol] = nval;
                    curr_design_matrix(vol,0)=1;
                    if(this->weight_imgs.size())
                    {
                        weights[vol] =this->weight_imgs[vol_id]->GetPixel(ind3);
                    }

                    curr_design_matrix(vol,0)=1;
                    curr_design_matrix(vol,1)= -curr_Bmatrix(vol,0)/1000.;
                    curr_design_matrix(vol,2)= -curr_Bmatrix(vol,1)/1000.;
                    curr_design_matrix(vol,3)= -curr_Bmatrix(vol,2)/1000.;
                    curr_design_matrix(vol,4)= -curr_Bmatrix(vol,3)/1000.;
                    curr_design_matrix(vol,5)= -curr_Bmatrix(vol,4)/1000.;
                    curr_design_matrix(vol,6)= -curr_Bmatrix(vol,5)/1000.;

                } //for vol

                if(broken_voxel)
                    continue;

                double mn=0;
                for(int v=0;v<b0_indices.size();v++)
                {
                    mn+=dwi_data[b0_indices[v]]->GetPixel(ind3);
                }
                mn/=b0_indices.size();
                double signal_ratio = mn / median_signal_b0;


                if(b0_indices.size() !=1)
                {
                   if(signal_ratio >1)
                   {
                       for(int v=0;v<b0_indices.size();v++)
                       {
                           curr_sigstdev[b0_indices[v]] = noise_kappa * signal_ratio* median_signal_b0_std;
                       }
                   }
                }
                for(int mm=0;mm<curr_sigstdev.size();mm++)
                {
                    if(curr_sigstdev[mm]==0)
                    {
                        if(curr_noise_val!=0)
                            curr_sigstdev=curr_noise_val;
                        else
                            broken_voxel=true;
                    }
                }
                if(broken_voxel)
                    continue;


                //RESTORE fit a single voxel
                double A0_estimate;
                std::vector<int> outlier_index;
                double CS_value=0;
                DTImageType::PixelType dt_vec = RobustFit(curr_design_matrix,signal, A0_img->GetPixel(ind3), output_img->GetPixel(ind3),  curr_sigstdev,
                                                          b0_indices,A0_estimate,outlier_index,CS_value,THR);

                if(CS_value<0)
                    CS_value=CS_val;

                output_img->SetPixel(ind3,dt_vec);
                A0_img->SetPixel(ind3,A0_estimate);
                CS_img->SetPixel(ind3,CS_value);

                //At the end, have the weights binary. Either inlier or  outlier
                for(int v=0;v<outlier_index.size();v++)
                {
                    this->weight_imgs[outlier_index[v]]->SetPixel(ind3,0.);
                }

            } //for i
        } //for j
    } //for k
}





void DTIModel::EstimateTensorWLLSDiagonal()
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
         (*stream)<<"Computing Tensors WLLS Diagonal..."<<std::endl;
     else
         std::cout<<"Computing Tensors WLLS Diagonal..."<<std::endl;

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
                         int vol_id= all_indices[v];
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
                 vnl_matrix<double> curr_design_matrix(curr_all_indices.size(),4,0);
                 auto curr_Bmatrix = getCurrentBmatrix(ind3,curr_all_indices);

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
                     curr_design_matrix(vol,1)= -curr_Bmatrix(vol,0)/1000.;
                     curr_design_matrix(vol,2)= -curr_Bmatrix(vol,3)/1000.;
                     curr_design_matrix(vol,3)= -curr_Bmatrix(vol,5)/1000.;
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

                 vnl_matrix_fixed<double,3,3> Dmat;
                 Dmat(0,0)=D(1,0)/1000.;
                 Dmat(1,0)=0;
                 Dmat(0,1)=0;
                 Dmat(2,0)=0;
                 Dmat(0,2)=0;
                 Dmat(1,1)=D(2,0)/1000.;
                 Dmat(1,2)=0;
                 Dmat(2,1)=0;
                 Dmat(2,2)=D(3,0)/1000.;

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
                 dt[1]=0;
                 dt[2]=0;
                 dt[3]=Dmat_corr(1,1);
                 dt[4]=0;
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


void DTIModel::EstimateTensorN2()
{
    EstimateTensorWLLS();

    VF_img=ImageType3D::New();
    VF_img->SetRegions(A0_img->GetLargestPossibleRegion());
    VF_img->Allocate();
    VF_img->SetSpacing(A0_img->GetSpacing());
    VF_img->SetOrigin(A0_img->GetOrigin());
    VF_img->SetDirection(A0_img->GetDirection());
    VF_img->FillBuffer(0.);

    float free_water_adc_value=this->free_water_diffusivity/1000.;
    constexpr float fraction_of_free_water_adc_value=0.8;

    float factor =  free_water_adc_value/3.;

    DTImageType::PixelType new_tens;
    new_tens[0]=0.0008*factor;
    new_tens[3]=0.0008*factor;
    new_tens[5]=0.0008*factor;
    new_tens[1]=0;
    new_tens[2]=0;
    new_tens[4]=0;

    itk::ImageRegionIteratorWithIndex<DTImageType> it(output_img,output_img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType index=it.GetIndex();

        if(mask_img && mask_img->GetPixel(index)==0)
        {
            continue;
        }

        DTImageType::PixelType curr_tens= it.Get();
        double TR= curr_tens[0]+curr_tens[3]+curr_tens[5];
        if(TR > 0.006)
        {
            VF_img->SetPixel(index,0.1);
            it.Set(new_tens);
        }
        else
        {
            VF_img->SetPixel(index,0.9);
        }
    }


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
        (*stream)<<"Computing Dual Tensors ..."<<std::endl;
    else
        std::cout<<"Computing Dual Tensors ..."<<std::endl;

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

        double p[9]={0};
        mp_par pars[9];
        memset(&pars[0], 0, sizeof(pars));
        pars[0].limited[0]=1;
        pars[0].limited[1]=1;
        pars[0].limits[0]=0.01;
        pars[0].limits[1]=free_water_adc_value * fraction_of_free_water_adc_value;
        pars[0].side=3;

        pars[1].limited[0]=0;
        pars[1].limited[1]=0;
        pars[1].limits[0]=0.0;
        pars[1].limits[1]=0;
        pars[1].side=3;

        pars[2].limited[0]=0;
        pars[2].limited[1]=0;
        pars[2].limits[0]=0.0;
        pars[2].limits[1]=0;
        pars[2].side=3;

        pars[3].limited[0]=1;
        pars[3].limited[1]=1;
        pars[3].limits[0]=0.01;
        pars[3].limits[1]=free_water_adc_value * fraction_of_free_water_adc_value;
        pars[3].side=3;

        pars[4].limited[0]=0;
        pars[4].limited[1]=0;
        pars[4].limits[0]=0.0;
        pars[4].limits[1]=0;
        pars[4].side=3;

        pars[5].limited[0]=1;
        pars[5].limited[1]=1;
        pars[5].limits[0]=0.01;
        pars[5].limits[1]=free_water_adc_value * fraction_of_free_water_adc_value;
        pars[5].side=3;

        pars[6].limited[0]=1;
        pars[6].limited[1]=0;
        pars[6].limits[0]=0.0;
        pars[6].limits[1]=0;
        pars[6].side=3;

        pars[7].limited[0]=1;
        pars[7].limited[1]=1;
        pars[7].limits[0]=0.;
        pars[7].limits[1]=1.;
        pars[7].side=3;

        pars[8].limited[0]=1;
        pars[8].limited[1]=1;
        pars[8].limits[0]=free_water_adc_value;
        pars[8].limits[1]=free_water_adc_value;
        pars[8].side=3;
        pars[8].fixed=1;


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
                        int vol_id= all_indices[v];
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
                auto curr_Bmatrix = getCurrentBmatrix(ind3,curr_all_indices);

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
                    curr_design_matrix(vol,1)= -curr_Bmatrix(vol,0)/1000.;
                    curr_design_matrix(vol,2)= -curr_Bmatrix(vol,1)/1000.;
                    curr_design_matrix(vol,3)= -curr_Bmatrix(vol,2)/1000.;
                    curr_design_matrix(vol,4)= -curr_Bmatrix(vol,3)/1000.;
                    curr_design_matrix(vol,5)= -curr_Bmatrix(vol,4)/1000.;
                    curr_design_matrix(vol,6)= -curr_Bmatrix(vol,5)/1000.;


                } //for vol

                DTImageType::PixelType dt_vec= output_img->GetPixel(ind3);
                for(int pp=0;pp<6;pp++)
                {
                    p[pp]= dt_vec[pp]*1000.;
                    if(pars[pp].limited[0])
                    {
                        if(p[pp] < pars[pp].limits[0])
                            p[pp]=pars[pp].limits[0]+0.000001;
                    }
                    if(pars[pp].limited[1])
                    {
                        if(p[pp] > pars[pp].limits[1])
                            p[pp]=pars[pp].limits[1]-0.000001;
                    }
                }
                p[6]= A0_img->GetPixel(ind3);
                p[7]= VF_img->GetPixel(ind3);
                p[8]=free_water_adc_value;


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

                int status = mpfit(myNLLS2, curr_design_matrix.rows(), 9, p, pars, &config, (void *) &my_struct, &my_results_struct);

                double degrees_of_freedom= curr_all_indices.size()-7;
                CS_img->SetPixel(ind3,my_results_struct.bestnorm/degrees_of_freedom);

                dt_vec[0]= p[0]/1000.;
                dt_vec[1]= p[1]/1000.;
                dt_vec[2]= p[2]/1000.;
                dt_vec[3]= p[3]/1000.;
                dt_vec[4]= p[4]/1000.;
                dt_vec[5]= p[5]/1000.;

                if(p[0]> 3.2 || p[3] > 3.2 || p[5]>3.2)  //FLOW ARTIFACT
                {
                    dt_vec= output_img->GetPixel(ind3);
                }

                if(p[1]>1.1 || p[2]> 1.1 || p[4]>1.1)  //FLOW ARTIFACT
                {
                    dt_vec= output_img->GetPixel(ind3);
                }

                 output_img->SetPixel(ind3,dt_vec);
                 A0_img->SetPixel(ind3,p[6]);
                 VF_img->SetPixel(ind3,p[7]);
            } //for i
        } //for j
    } //for k

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
                        int vol_id= all_indices[v];
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
                auto curr_Bmatrix = getCurrentBmatrix(ind3,curr_all_indices);

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
                    curr_design_matrix(vol,1)= -curr_Bmatrix(vol,0)/1000.;
                    curr_design_matrix(vol,2)= -curr_Bmatrix(vol,1)/1000.;
                    curr_design_matrix(vol,3)= -curr_Bmatrix(vol,2)/1000.;
                    curr_design_matrix(vol,4)= -curr_Bmatrix(vol,3)/1000.;
                    curr_design_matrix(vol,5)= -curr_Bmatrix(vol,4)/1000.;
                    curr_design_matrix(vol,6)= -curr_Bmatrix(vol,5)/1000.;

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
                         int vol_id= all_indices[v];
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

                 auto curr_Bmatrix = getCurrentBmatrix(ind3,curr_all_indices);

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
                     curr_design_matrix(vol,1)= -curr_Bmatrix(vol,0)/1000.;
                     curr_design_matrix(vol,2)= -curr_Bmatrix(vol,1)/1000.;
                     curr_design_matrix(vol,3)= -curr_Bmatrix(vol,2)/1000.;
                     curr_design_matrix(vol,4)= -curr_Bmatrix(vol,3)/1000.;
                     curr_design_matrix(vol,5)= -curr_Bmatrix(vol,4)/1000.;
                     curr_design_matrix(vol,6)= -curr_Bmatrix(vol,5)/1000.;
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

                 if( Dmat_corr(0,0)+Dmat_corr(1,1)+Dmat_corr(2,2) > 0.0085)  //flow
                 {
                     double mn = (eig.D(0,0)+ eig.D(1,1)+ eig.D(2,2))/3.;
                     double nom = (eig.D(0,0)-mn)*(eig.D(0,0)-mn)+ (eig.D(1,1)-mn)*(eig.D(1,1)-mn)+(eig.D(2,2)-mn)*(eig.D(2,2)-mn);
                     double denom= eig.D(0,0)*eig.D(0,0)+eig.D(1,1)*eig.D(1,1)+eig.D(2,2)*eig.D(2,2);


                     double FA=0;
                     if(denom!=0)
                         FA= sqrt( 1.5*nom/denom);

                     if(FA>0.5)
                     {
                         /*
                         Dmat_corr(0,0)=mn;
                         Dmat_corr(1,1)=mn;
                         Dmat_corr(2,2)=mn;

                         Dmat_corr(0,1)=Dmat_corr(1,0)=0;
                         Dmat_corr(0,2)=Dmat_corr(2,0)=0;
                         Dmat_corr(2,1)=Dmat_corr(1,2)=0;
*/
                         // Forcing FA to be 0.5
                         eig.D(0,0)= (1-sqrt(3./10))*mn;
                         eig.D(1,1)= mn;
                         eig.D(2,2)= (1+sqrt(3./10))*mn;
                         Dmat_corr= eig.recompose();


                     }
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



    double freeWater_ADC=3E-3;
    itk::ImageRegionIteratorWithIndex<ImageType3D> it(synth_image,synth_image->GetLargestPossibleRegion());
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind=it.GetIndex();
        DTType tensor= output_img->GetPixel(ind);
        double exp_term=0;
        for(int i=0;i<6;i++)
            exp_term += tensor[i] * bmatrix_vec[i];


        float A0val= A0_img->GetPixel(ind);

        float signal = A0val * exp(-exp_term);

        if(exp_term < 0)
            signal=0;


        if(VF_img)
        {
            double exp2= freeWater_ADC*(bmatrix_vec[0] + bmatrix_vec[3] + bmatrix_vec[5]);
            exp2=exp(-exp2);
            signal=VF_img->GetPixel(ind)* signal + A0val*(1-VF_img->GetPixel(ind))*exp2;
        }

        if(!isnan(signal) && isfinite(signal))
            it.Set(signal);
        ++it;
    }

    return synth_image;

}







#endif
