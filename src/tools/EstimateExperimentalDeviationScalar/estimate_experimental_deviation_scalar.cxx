#include "defines.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageDuplicator.h"


#include "../external_src/cmpfit-1.3a/mpfit.h"
#include "estimate_experimental_deviation_scalar_parser.h"
#include "vnl_matrix_inverse.h"
#include "vnl_trace.h"

#include "../utilities/read_bmatrix_file.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../tools/EstimateTensor/DTIModel.h"

#include "../utilities/math_utilities.h"


typedef  vnl_matrix_fixed< double, 3, 3 > InternalMatrixType;



DTImageType::Pointer   EstimateTensorNLLS_sub_new(std::vector<ImageType3D::Pointer> dwis, vnl_matrix<double> Bmatrix, std::vector<int> &DT_indices, ImageType3D::Pointer & A0_image,ImageType3D::Pointer mask_image, ImageType3D::Pointer &CS_img, ImageType4D::Pointer &resid_img, vnl_matrix<double> &W)
{
    bool useNoise=0;
    bool compute_resid=0;
    double noise_std=1;



    DTIModel dti_estimator;
    dti_estimator.SetBmatrix(Bmatrix);
    dti_estimator.SetDWIData(dwis);
    //dti_estimator.SetWeightImage(nullptr);
    //dti_estimator.SetVoxelwiseBmatrix(dummyv);
    dti_estimator.SetMaskImage(mask_image);
    dti_estimator.SetVolIndicesForFitting(DT_indices);
    dti_estimator.SetFittingMode("WLLS");
    dti_estimator.PerformFitting();
    DTImageType::Pointer dt_image=dti_estimator.GetOutput();
    A0_image=dti_estimator.GetA0Image();

    ImageType3D::SizeType size = dwis[0]->GetLargestPossibleRegion().GetSize();


    int Nvols=Bmatrix.rows();

    ImageType3D::Pointer vol3= dwis[0];


    CS_img=ImageType3D::New();
    CS_img->SetRegions(vol3->GetLargestPossibleRegion());
    CS_img->Allocate();
    CS_img->SetSpacing(vol3->GetSpacing());
    CS_img->SetOrigin(vol3->GetOrigin());    
    CS_img->SetDirection(vol3->GetDirection());    
    CS_img->FillBuffer(0.);


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



    {
        std::vector<int> all_indices;
        if(DT_indices.size()>0)
            all_indices= DT_indices;
        else
        {
            for(int ma=0;ma<Nvols;ma++)
                all_indices.push_back(ma);
        }


        vnl_matrix<double> design_matrix(all_indices.size(),7);
        W=design_matrix;
        for(int i=0;i<all_indices.size();i++)
        {
            design_matrix(i,0)=1;
            W(i,0)=1;
            for(int j=0;j<6;j++)
            {
                design_matrix(i,j+1)= -Bmatrix(all_indices[i],j)/1000.;
                W(i,j+1)= -Bmatrix(all_indices[i],j);
            }
        }




        std::cout<<"Computing N1 Tensors..."<<std::endl;


         #pragma omp parallel for
         for(int k=0;k<size[2];k++)
         {             
             ImageType3D::IndexType ind3;          
             ind3[2]=k;

             vars_struct my_struct;
             my_struct.useWeights=useNoise;


             mp_result_struct my_results_struct;
             vnl_vector<double> my_resids(all_indices.size());
             my_results_struct.resid= my_resids.data_block();
             my_results_struct.xerror=NULL;
             my_results_struct.covar=NULL;


             double p[7];
             mp_par pars[7];
             memset(&pars[0], 0, sizeof(pars));
             pars[0].side=0;
             pars[1].side=0;
             pars[2].side=0;
             pars[3].side=0;
             pars[4].side=0;
             pars[5].side=0;
             pars[6].side=0;

             vnl_vector<double> signal(all_indices.size());
             vnl_vector<double> weights(all_indices.size(),1.);

             vnl_vector<double> curr_signal=signal;
             vnl_vector<double> curr_weights=weights;
             std::vector<int> curr_all_indices= all_indices;


             double degrees_of_freedom= all_indices.size()-7;
             vnl_matrix<double> curr_design_matrix=design_matrix;


             for(int j=0;j<size[1];j++)
             {                 
                 ind3[1]=j;
                 for(int i=0;i<size[0];i++)
                 {              
                     ind3[0]=i;

                     if(mask_image && mask_image->GetPixel(ind3)==0)
                         continue;


                     for(int vol=0;vol<curr_all_indices.size();vol++)
                     {                         
                         double nval = dwis[curr_all_indices[vol]]->GetPixel(ind3);


                         if(nval <=0)
                         {
                             std::vector<float> data_for_median;
                             std::vector<float> noise_for_median;

                             ImageType3D::IndexType newind3;
                             newind3[2]=k;


                             for(int i2= std::max(i-1,0);i2<=std::min(i+1,(int)size[0]-1);i2++)
                             {
                                 newind3[0]=i2;
                                 for(int j2= std::max(j-1,0);j2<=std::min(j+1,(int)size[1]-1);j2++)
                                 {
                                     newind3[1]=j2;

                                     float newval= dwis[curr_all_indices[vol]]->GetPixel(newind3);
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
                                 nval=0.01;
                             }
                         }



                         //curr_signal[vol] = dwis->GetPixel(ind);
                         curr_signal[vol] = nval;
                         curr_design_matrix(vol,0)=1;
                         for(int jj=0;jj<6;jj++)
                         {
                             curr_design_matrix(vol,jj+1)= -Bmatrix(curr_all_indices[vol],jj)/1000.;
                         }

                     }

                     DTImageType::PixelType dt_vec= dt_image->GetPixel(ind3);



                     double p[7];
                     p[0]= A0_image->GetPixel(ind3);
                     p[1]= dt_vec[0]*1000.;
                     p[2]= dt_vec[1]*1000.;
                     p[3]= dt_vec[2]*1000.;
                     p[4]= dt_vec[3]*1000.;
                     p[5]= dt_vec[4]*1000.;
                     p[6]= dt_vec[5]*1000.;

                     my_struct.signal= &curr_signal;
                     my_struct.weights=&curr_weights;
                     my_struct.Bmat= &curr_design_matrix;


                     int status = mpfit(myNLLS_with_derivs, curr_design_matrix.rows(), 7, p, pars, &config, (void *) &my_struct, &my_results_struct);
                     //int status = mpfit(myNLLS_with_derivs, curr_design_matrix.rows(), 7, p, pars, NULL, (void *) &my_struct, &my_results_struct);

                     CS_img->SetPixel(ind3,2*my_results_struct.bestnorm/degrees_of_freedom);





                     dt_vec[0]= p[1]/1000.;
                     dt_vec[1]= p[2]/1000.;
                     dt_vec[2]= p[3]/1000.;
                     dt_vec[3]= p[4]/1000.;
                     dt_vec[4]= p[5]/1000.;
                     dt_vec[5]= p[6]/1000.;


                     if(p[1]> 3.2 || p[4] > 3.2 || p[6]>3.2)  //FLOW ARTIFACT
                     {
                         dt_vec= dt_image->GetPixel(ind3);
                     }

                     if(p[2]>1.1 || p[3]> 1.1 || p[5]>1.1)  //FLOW ARTIFACT
                     {
                         dt_vec= dt_image->GetPixel(ind3);
                     }

                     dt_image->SetPixel(ind3,dt_vec);
                     A0_image->SetPixel(ind3,p[0]);

                 }
             }
          }

    }    

    return dt_image;


}








int main(int argc, char *argv[])
{    
    EstimateExperimentalDeviationScalar_PARSER *parser= new EstimateExperimentalDeviationScalar_PARSER(argc,argv);


    std::string nii_name= parser->getInputImageName();
    std::string bmtxt_name = nii_name.substr(0,nii_name.rfind(".nii"))+".bmtxt";

    ImageType3D::Pointer mask_image=nullptr;
    if(parser->getMaskImageName()!="")
    {
        if(fs::exists(parser->getMaskImageName()))
        {
            mask_image=readImageD<ImageType3D>(parser->getMaskImageName());
        }
    }

    vnl_matrix<double> Bmatrix= read_bmatrix_file(bmtxt_name);
    int Nvols= Bmatrix.rows();
    std::vector<ImageType3D::Pointer> dwis; dwis.resize(Nvols);
    for(int v=0;v<Nvols;v++)
        dwis[v]=read_3D_volume_from_4D(nii_name,v);


    std::vector<int> dummy;
    {
        double bval_cut =parser->getBValCutoff();
        for(int i=0;i<Bmatrix.rows();i++)
        {
            double bval= Bmatrix[i][0]+ Bmatrix[i][3]+Bmatrix[i][5];
            if(bval<=1.05*bval_cut)
                dummy.push_back(i);
        }
    }


    ImageType3D::Pointer A0_image=nullptr;
    DTImageType::Pointer dt_image=nullptr;


    ImageType3D::Pointer CS_img=nullptr;
    ImageType4D::Pointer resid_img=nullptr;


    vnl_matrix<double> W;


    dt_image= EstimateTensorNLLS_sub_new(dwis,Bmatrix, dummy,A0_image,mask_image,CS_img,resid_img,W);




    ImageType3D::Pointer modvar_img=ImageType3D::New();
    modvar_img->SetRegions(dt_image->GetLargestPossibleRegion());
    modvar_img->Allocate();
    modvar_img->SetSpacing(dt_image->GetSpacing());
    modvar_img->SetOrigin(dt_image->GetOrigin());    
    modvar_img->SetDirection(dt_image->GetDirection());



    ImageType3D::SizeType size = modvar_img->GetLargestPossibleRegion().GetSize();



     #pragma omp parallel for
     for(int k=0;k<size[2];k++)
     {
         ImageType3D::IndexType ind3;
         ind3[2]=k;

         vnl_matrix<double> del_TR(7,1,0),del_FA(7,1,0);
         del_TR(1,0)=1;
         del_TR(2,0)=1;
         del_TR(3,0)=1;


         for(int j=0;j<size[1];j++)
         {
             ind3[1]=j;             
             for(int i=0;i<size[0];i++)
             {
                 ind3[0]=i;                 
                 vnl_diag_matrix<double> S_hat(Nvols);
                 vnl_diag_matrix<double> R(Nvols);

                 DTImageType::PixelType dt= dt_image->GetPixel(ind3);
                 double A0 = A0_image->GetPixel(ind3);
                 if(A0==0)
                     continue;

                 for(int v=0;v<Nvols;v++)
                 {                     
                     double S=dwis[v]->GetPixel(ind3);

                     double xpon = dt[0]*W(v,1) + dt[1]*W(v,2) + dt[2]*W(v,3) + dt[3]*W(v,4) + dt[4]*W(v,5) + dt[5]*W(v,6) ;
                     double est=   A0 * exp(xpon);
                     S_hat(v,v)= est;
                     R(v,v)= S - est;
                 }

                 vnl_matrix<double> hess = W.transpose() * (S_hat*S_hat - R *S_hat)*W;
                 vnl_matrix<double> inv_hess= vnl_matrix_inverse<double>(hess);


                 double CS= CS_img->GetPixel(ind3);
                 vnl_matrix<double> var= 1000000.*CS* del_TR.transpose() *inv_hess * del_TR;

                 if(parser->getModality()=="FA")
                 {
                     InternalMatrixType curr_tens;
                     curr_tens(0,0)=dt[0];
                     curr_tens(0,1)=dt[1];
                     curr_tens(1,0)=dt[1];
                     curr_tens(0,2)=dt[2];
                     curr_tens(0,2)=dt[2];
                     curr_tens(1,1)=dt[3];
                     curr_tens(1,2)=dt[4];
                     curr_tens(2,1)=dt[4];
                     curr_tens(2,2)=dt[5];

                     vnl_symmetric_eigensystem<double>  eig(curr_tens);

                     double mn = (eig.D(0,0)+ eig.D(1,1)+ eig.D(2,2))/3.;
                     double nom = (eig.D(0,0)-mn)*(eig.D(0,0)-mn)+ (eig.D(1,1)-mn)*(eig.D(1,1)-mn)+(eig.D(2,2)-mn)*(eig.D(2,2)-mn);
                     double denom= eig.D(0,0)*eig.D(0,0)+eig.D(1,1)*eig.D(1,1)+eig.D(2,2)*eig.D(2,2);

                     vnl_matrix<double> ct2= curr_tens *curr_tens;

                     double FA=0;
                     if(denom!=0)
                         FA= sqrt( 1.5*nom/denom);

                     double tr_ratio = vnl_trace<double>(curr_tens)/vnl_trace<double>(ct2);

                     del_FA(1,0)= 1./FA * tr_ratio *  ( (  1./2. *tr_ratio*dt[0]     ) - 0.5     );
                     del_FA(2,0)= 1./FA * tr_ratio *  ( (   tr_ratio*dt[1]     )      );
                     del_FA(3,0)= 1./FA * tr_ratio *  ( (   tr_ratio*dt[2]     )      );
                     del_FA(4,0)= 1./FA * tr_ratio *  ( (  1./2. *tr_ratio*dt[3]     ) - 0.5     );
                     del_FA(5,0)= 1./FA * tr_ratio *  ( (   tr_ratio*dt[4]     )      );
                     del_FA(6,0)= 1./FA * tr_ratio *  ( (  1./2. *tr_ratio*dt[5]     ) - 0.5     );

                     var= (double)(CS_img->GetPixel(ind3))* del_FA.transpose() *inv_hess * del_FA;
                 }


                 modvar_img->SetPixel(ind3,sqrt(0.5*var(0,0)));
             }
         }
     }





    std::string listname(parser->getInputImageName());
    std::string full_base_name= listname.substr(0, listname.find(".list"));

    std::string var_name= full_base_name + std::string("_N1_") + parser->getModality() + std::string("_std.nii");

    writeImageD<ImageType3D>(modvar_img,var_name);

}
