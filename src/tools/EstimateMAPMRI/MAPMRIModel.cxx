#ifndef _MAPMRIModel_CXX
#define _MAPMRIModel_CXX


#include "MAPMRIModel.h"
#include "../tools/EstimateTensor/DTIModel.h"
#include "../utilities/math_utilities.h"

#include "constrained_least_squares.h"
#include <math.h>
#include "itkImageDuplicator.h"

void MAPMRIModel::PerformFitting()
{
    if(!this->dt_image || ! this->A0_img)
    {
        DTIModel dti_estimator;
        dti_estimator.SetBmatrix(this->Bmatrix);
        dti_estimator.SetDWIData(this->dwi_data);
        dti_estimator.SetWeightImage(this->weight_imgs);
        dti_estimator.SetVoxelwiseBmatrix(this->voxelwise_Bmatrix);        
        dti_estimator.SetGradDev(this->graddev_img);
        dti_estimator.SetMaskImage(this->mask_img);
        dti_estimator.SetVolIndicesForFitting(this->DT_indices);
        dti_estimator.PerformFitting();
        this->dt_image= dti_estimator.GetOutput();
        this->A0_img= dti_estimator.GetA0Image();
    }

    ComputeEigenImages();


    const int N_MAPMRI_COEFFS =((((MAP_DEGREE/2)+1)*((MAP_DEGREE/2)+2)*(4*(MAP_DEGREE/2)+3))/6);

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
    if(indices_fitting.size()>0)
        all_indices= indices_fitting;
    else
    {
        for(int ma=0;ma<Nvols;ma++)
            all_indices.push_back(ma);
    }


    if(stream)
        (*stream)<<"Computing MAPMRI with degree: "<<MAP_DEGREE<<std::endl;
    else
        std::cout<<"Computing MAPMRI with degree: "<<MAP_DEGREE<<std::endl;



    #pragma omp parallel for
    for(int k=0;k<size[2];k++)
    {
        TORTOISE::EnableOMPThread();


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

                auto curr_Bmatrix = getCurrentBmatrix(index3,curr_all_indices);
                vnl_matrix<double> qq_orig= bmat2q(curr_Bmatrix,curr_all_indices);
                vnl_matrix<double> curr_qq=qq_orig;

                 EValType eval = eval_image->GetPixel(index3);
                 if( A0_img->GetPixel(index3)!=0)
                 {
                     if((mask_img==nullptr) || mask_img->GetPixel(index3)!=0 )
                     {
                         vnl_vector<double> weights(curr_all_indices.size(),1);

                         for(int vol=0;vol<curr_all_indices.size();vol++)
                         {                                                          
                             int vol_id= curr_all_indices[vol];


                             if(this->weight_imgs.size())
                                 weights[vol]= this->weight_imgs[vol_id]->GetPixel(index3);


                             double nval = dwi_data[vol_id]->GetPixel(index3);
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

                                         float newval= dwi_data[vol_id]->GetPixel(newind);
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


                         MAPType coeffs= FitMAPMRI(curr_signal, A0_img->GetPixel(index3), MAP_DEGREE, eval_image->GetPixel(index3), qqtmp,big_delta-small_delta/3.,0,weights);

                         mapmri_image->SetPixel(index3,coeffs);
                     }
                 }
             }
         }

         TORTOISE::DisableOMPThread();
     }

    output_img=mapmri_image;

}


vnl_matrix<double> MAPMRIModel::hermiteh(int nn, vnl_matrix<double> xx)
{
    vnl_matrix<double> result(nn+1,xx.columns());
    result.fill(1);
    if(nn==0)
        return result;

    for(int i=0;i<xx.columns();i++)
        result(1,i)= 2* xx(0,i);


    for(int order=1;order<nn;order++)
    {
        for(int i=0;i<xx.columns();i++)
        {
            result(order+1,i) = 2 *  (xx(0,i)* result(order,i)- order * result(order-1,i));
        }
    }

    return result;
}

MatrixXd MAPMRIModel::shore_car_phi(int nn, double uu, vnl_matrix<double> qarr)
{
    vnl_matrix<double> term1(1,qarr.columns());

    for(int c=0;c<qarr.columns();c++)
    {
       term1(0,c)=exp(-(2.* (DPI*uu*qarr(0,c))*(DPI*uu*qarr(0,c))));
    }


    vnl_vector<double> facts(nn+1);
    myfactorial(facts);

    vnl_matrix<double> term2(1,nn+1);


    for(int c=0;c<nn+1;c++)
    {
       term2(0,c)=  1./   sqrt(    std::pow(2.0, c)*facts[c]        );
    }

    vnl_matrix<double> term12=  term2.transpose()*term1;

    vnl_matrix<double> temp = 2* DPI *uu* qarr;
    vnl_matrix<double> term3 = hermiteh(nn, temp );

    MatrixXd final(term3.rows(),term3.columns());

    for(int r=0;r<term3.rows();r++)
        for(int c=0;c<term3.columns();c++)
            final(r,c)= term3(r,c)* term12(r,c);

    return final;
}


MatrixXd MAPMRIModel::mk_ashore_basis(int order, vnl_vector<double> & uvec,vnl_matrix<double> &qxyz, bool qsp)
{
    vnl_vector<double> uu=uvec;


    if(!qsp)
    {
        uu[0]=1./(2*DPI*uvec[0]);
        uu[1]=1./(2*DPI*uvec[1]);
        uu[2]=1./(2*DPI*uvec[2]);
    }


    MatrixXd basx= shore_car_phi(order, uu[0], qxyz.get_n_rows(0,1));
    MatrixXd basy= shore_car_phi(order, uu[1], qxyz.get_n_rows(1,1));
    MatrixXd basz= shore_car_phi(order, uu[2], qxyz.get_n_rows(2,1));

    if(!qsp)
    {
        basx*=sqrt(2*DPI)*uu[0];
        basy*=sqrt(2*DPI)*uu[1];
        basz*=sqrt(2*DPI)*uu[2];
    }


     int nnmax[]={0,6,21,49,94,160};

     int n1a[]={0,2,0,0,1,1,0,4,0,0,3,3,1,1,0,0,2,2,0,2,1,1,6,0,0,5,5,1,1,0,0,4,4,2,2,0,0,4,1,1,3,3,0,3,3,2,2,1,1,2,8,0,0,7,7,1,
         1,0,0,6,6,2,2,0,0,6,1,1,5,5,3,3,0,0,5,5,2,2,1,1,4,4,0,4,4,3,3,1,1,4,2,2,3,3,2,10,0,0,9,9,1,1,0,0,8,8,2,2,0,0,8,1,
         1,7,7,3,3,0,0,7,7,2,2,1,1,6,6,4,4,0,0,6,6,3,3,1,1,6,2,2,5,5,0,5,5,4,4,1,1,5,5,3,3,2,2,4,4,2,4,3,3};

    int n2a[]={0,0,2,0,1,0,1,0,4,0,1,0,3,0,3,1,2,0,2,1,2,1,0,6,0,1,0,5,0,5,1,2,0,4,0,4,2,1,4,1,3,0,3,2,1,3,1,3,2,2,0,8,0,1,0,7,0,
         7,1,2,0,6,0,6,2,1,6,1,3,0,5,0,5,3,2,1,5,1,5,2,4,0,4,3,1,4,1,4,3,2,4,2,3,2,3,0,10,0,1,0,9,0,9,1,2,0,8,0,8,2,1,8,1,3,
         0,7,0,7,3,2,1,7,1,7,2,4,0,6,0,6,4,3,1,6,1,6,3,2,6,2,5,0,5,4,1,5,1,5,4,3,2,5,2,5,3,4,2,4,3,4,3};

    int n3a[]={0,0,0,2,0,1,1,0,0,4,0,1,0,3,1,3,0,2,2,1,1,2,0,0,6,0,1,0,5,1,5,0,2,0,4,2,4,1,1,4,0,3,3,1,2,1,3,2,3,2,0,0,8,0,1,
         0,7,1,7,0,2,0,6,2,6,1,1,6,0,3,0,5,3,5,1,2,1,5,2,5,0,4,4,1,3,1,4,3,4,2,2,4,2,3,3,0,0,10,0,1,0,9,1,9,0,2,0,8,2,8,
         1,1,8,0,3,0,7,3,7,1,2,1,7,2,7,0,4,0,6,4,6,1,3,1,6,3,6,2,2,6,0,5,5,1,4,1,5,4,5,2,3,2,5,3,5,2,4,4,3,3,4};



    int lim= nnmax[order/2]+1;
    MatrixXd basx_sub(lim, basx.cols());
    MatrixXd basy_sub(lim, basx.cols());
    MatrixXd basz_sub(lim, basx.cols());

    for(int i=0;i<lim;i++)
    {
        basx_sub.row(i)= basx.row(n1a[i]);
        basy_sub.row(i)= basy.row(n2a[i]);
        basz_sub.row(i)= basz.row(n3a[i]);
    }

    MatrixXd result(lim,basx.cols());
    for(int r=0;r<result.rows();r++)
        for(int c=0;c<result.cols();c++)
            result(r,c)= basx_sub(r,c)* basy_sub(r,c)* basz_sub(r,c);

    if(qsp)
    {
        std::complex<double> aa(0,1);
        MatrixXcd  ns(lim,1);
        for(int i=0;i<lim;i++)
        {
            ns(i,0)= std::pow(aa,1.*(n1a[i]+n2a[i]+n3a[i]) );
        }


       MatrixXcd sgn(lim,basx.cols());
       for(int i=0;i<basx.cols();i++)
           sgn.col(i) = ns.col(0);

       for(int r=0;r<result.rows();r++)
           for(int c=0;c<result.cols();c++)
               result(r,c)= result(r,c) * (sgn(r,c).real());
    }

   return result;
}


vnl_matrix<double>  MAPMRIModel::shore_3d_reconstruction_domain(int order, vnl_vector<double>& uvec)
{
    int rec_zdim=11;
    double ratio=80;

    vnl_vector<double> max_radius=uvec*sqrt(order*log(ratio));


    int rec_xdim=2*rec_zdim-1;
    int rec_ydim=rec_xdim;

    int npts=long(rec_xdim)*rec_ydim*rec_zdim;


    vnl_matrix<double> rec_xyz_arr(npts,3);
    vnl_vector<double> delta_x=max_radius / (rec_zdim-1) ;

    long cnt=0;
    for( int z=0; z< rec_zdim;z++)
    {
        for( int y=-(rec_ydim-1)/2; y<= (rec_ydim-1)/2;y++)
        {
            for( int x=-(rec_xdim-1)/2; x<= (rec_xdim-1)/2;x++)
            {

                rec_xyz_arr(cnt,0)= delta_x[0]*x;
                rec_xyz_arr(cnt,1)= delta_x[1]*y;
                rec_xyz_arr(cnt,2)= delta_x[2]*z;

                cnt++;
            }
        }
    }

    return rec_xyz_arr.transpose();
}



MAPType MAPMRIModel::FitMAPMRI(std::vector<float> &signal, float A0val, int order, EValType uvec, vnl_matrix<double> &qxyz,double tdiff, double reg_weight,vnl_vector<double> weights_vector)
{

    vnl_vector<double> uu(3);
    uu[0]=sqrt(uvec[0]*2000.*tdiff);
    uu[1]=sqrt(uvec[1]*2000.*tdiff);
    uu[2]=sqrt(uvec[2]*2000.*tdiff);

    typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_type;
    vector_type sig_vec(signal.size());
    for(int i= 0; i<signal.size();i++)
        sig_vec[i]=signal[i];


    MatrixXd qmtrx=mk_ashore_basis(order,uu,qxyz,1);
    qmtrx.transposeInPlace();
    MatrixXd Hmtrx =qmtrx.transpose() * qmtrx;

    vnl_matrix<double> rmtrx2= shore_3d_reconstruction_domain(order, uu);
    MatrixXd rmtrx= mk_ashore_basis(order,uu,rmtrx2,0);
    rmtrx.transposeInPlace();


    Eigen::DiagonalMatrix<double,Eigen::Dynamic> W;
    if(weights_vector.size()!=0)
    {
        W.resize(weights_vector.size());
        for(int vv=0;vv<weights_vector.size();vv++)
            //W.diagonal()[vv]= sqrt(weights_vector[vv]);
            W.diagonal()[vv]= (weights_vector[vv]);

        qmtrx=W*qmtrx;
        sig_vec= W*sig_vec;
    }




    JacobiSVD<MatrixXd> svd(Hmtrx);
    double cond = svd.singularValues()(0)         / svd.singularValues()(svd.singularValues().size()-1);

    ICLS::Problem<double> problem;
    if(cond > 1E10)
    {
        if(weights_vector.size()==0)
            problem = ICLS::Problem<double>(qmtrx,rmtrx,1E-4,1E-4);
        else
            problem = ICLS::Problem<double>(qmtrx,rmtrx,1E-4,1E-4,5*qmtrx.cols());
    }
    else
    {
        if(weights_vector.size()==0)
            problem = ICLS::Problem<double>(qmtrx,rmtrx,1E-10,1E-10);
        else
            problem = ICLS::Problem<double>(qmtrx,rmtrx,1E-10,1E-10,5*qmtrx.cols());

    }




    vector_type x;

    ICLS::Solver<double> solver(problem);
    int niter =solver(x,sig_vec);

    MAPType res;
    res.SetSize(x.rows());
    for(int i=0;i<x.rows();i++)
    {
        res[i]=x[i];
    }

        return res;
}




ImageType3D::Pointer MAPMRIModel::SynthesizeDWI(vnl_vector<double> bmat_vec)
{
    MAPImageType::IndexType ind_temp; ind_temp.Fill(0);
    MAPImageType::PixelType vec= output_img->GetPixel(ind_temp);
    int ncoeffs= vec.Size();



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
    vnl_matrix<double> bmat_vec_mat(1,6);
    bmat_vec_mat.set_row(0,bmat_vec);
    vnl_matrix<double> qq= bmat2q(bmat_vec_mat,all_indices);

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
                if( (this->mask_img && this->mask_img->GetPixel(index3)) || (!this->mask_img) )
                {
                    MAPType coeffs= output_img->GetPixel(index3);
                    EVecType R= evec_image->GetPixel(index3);
                    EValType uvec= eval_image->GetPixel(index3);


                    vnl_vector<double> uu(3);
                    uu[0]= sqrt(uvec[0]*2000.*tdiff);
                    uu[1]= sqrt(uvec[1]*2000.*tdiff);
                    uu[2]= sqrt(uvec[2]*2000.*tdiff);

                    vnl_matrix<double> qqtmp= R * qq;

                    MatrixXd qmtrx=mk_ashore_basis(MAP_DEGREE,uu,qqtmp,1);

                    double sm=0;
                    for(int i=0;i<qmtrx.rows();i++)
                        sm+= coeffs[i] * qmtrx(i,0);


                    //sm= sm* A0_image->GetPixel(index3);

                    if(!isnan(sm) && isfinite(sm))
                        synth_image->SetPixel(index3,sm);
                }
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
        if(A0_img->GetPixel(ind) > 0)
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
        if(A0_img->GetPixel(ind) > 0)
        {
            float val= it.Get();
            if(   (val > median_val + 20*MAD) || (val < median_val - 20*MAD) )
                it.Set(0);
        }
        ++it;
    }



    using DupType = itk::ImageDuplicator<ImageType3D>;
    DupType::Pointer dup = DupType::New();
    dup->SetInputImage(synth_image);
    dup->Update();
    ImageType3D::Pointer synth_img2= dup->GetOutput();



    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind = it.GetIndex();
        if(A0_img->GetPixel(ind) > 0)
        {
            if(ind[0]>2 && ind[1]>2 && ind[2]>0 && ind[0]<size[0]-3 && ind[1]<size[1]-3 && ind[2]<size[2]-1)
            {
                std::vector<float> vals; vals.clear();

                ImageType3D::IndexType tind3=ind;
                for(int kk=-1;kk<=1;kk++)
                {
                    tind3[2]=ind[2]+kk;
                    for(int jj=-3;jj<=3;jj++)
                    {
                        tind3[1]=ind[1]+jj;
                        for(int ii=-3;ii<=3;ii++)
                        {
                            tind3[0]=ind[0]+ii;
                            if(A0_img->GetPixel(tind3))
                                vals.push_back(synth_img2->GetPixel(tind3));
                        }
                    }
                }
                if(vals.size()>0)
                {
                    median_val=median(vals);
                    for(int s=0;s<vals.size();s++)
                    {
                        vals[s]=fabs(vals[s]-median_val);
                    }
                    MAD = median(vals);

                    if(  (synth_img2->GetPixel(ind)-median_val)/MAD > 8.)
                        it.Set(median_val);
                }
            }

        }

    }
















    //Obviously buggy
/*
    it.GoToBegin();
    vals.resize(0);
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind = it.GetIndex();
        if(A0_img->GetPixel(ind) > 0)
        {
             if(ind[0]>0 && ind[1]>0 && ind[2]>0 && ind[0]<size[0]-1 && ind[1]<size[1]-1 && ind[2]<size[2]-1)
             {
                 ImageType3D::IndexType tind3=ind;
                 for(int kk=-1;kk<=1;kk++)
                 {
                     tind3[2]=ind[2]+kk;
                     for(int jj=-2;jj<=2;jj++)
                     {
                         tind3[1]=ind[1]+jj;
                         for(int ii=-2;ii<=2;ii++)
                         {
                             tind3[0]=ind[0]+ii;
                             vals.push_back(synth_image->GetPixel(tind3));
                         }
                     }
                 }
                 median_val=median(vals);
                 for(int s=0;s<vals.size();s++)
                 {
                     vals[s]=fabs(vals[s]-median_val);
                 }
                 MAD = median(vals)*1.4228;

                 if( fabs(it.Get()-median_val)/MAD > 10.)
                     it.Set(median_val);
             }
        }
        ++it;
    }
*/


    return synth_image;

}


vnl_matrix<double>  MAPMRIModel::bmat2q(vnl_matrix<double> cBMatrix, std::vector<int> all_indices,bool qspace)
{
    vnl_matrix<double> qmat;
    qmat.set_size(3,all_indices.size());

    double delta= big_delta-small_delta/3.;

    for(int v=0;v<all_indices.size();v++)
    {
        int vol =all_indices[v];
        vnl_vector<double> bmat_vec= cBMatrix.get_row(vol);

        vnl_matrix<double> bmat(3,3);
        bmat(0,0) =  bmat_vec[0];  bmat(0,1) =  bmat_vec[1]/2; bmat(0,2) =  bmat_vec[2]/2;
        bmat(1,0) =  bmat_vec[1]/2;  bmat(1,1) =  bmat_vec[3]; bmat(1,2) =  bmat_vec[4]/2;
        bmat(2,0) =  bmat_vec[2]/2;  bmat(2,1) =  bmat_vec[4]/2; bmat(2,2) =  bmat_vec[5];

        bmat= bmat / 1000.;

        vnl_symmetric_eigensystem<double> eig(bmat);
        if(eig.D(2,2)<0)
            eig.D(2,2)=0;

        double bval=eig.D(2,2);
        vnl_vector<double> bvec= eig.get_eigenvector(2);

        double qq;
        if(qspace)
            qq=sqrt(bval/delta)/(2*DPI)  ;
        else
        {
            qq=bmat_vec[0] + bmat_vec[3] + bmat_vec[5];
        }

        vnl_vector<double> qvec= qq * bvec;
        qmat.set_column(v,qvec);
    }

    return qmat;

}


void MAPMRIModel::ComputeEigenImages()
{
    if(!dt_image)
    {
        (*stream)<<"DTI Image not defined for MAPMRI fitting...Exiting..."<<std::endl;
        exit(EXIT_FAILURE);
    }


    this->evec_image = EVecImageType::New();
    evec_image->SetRegions(dt_image->GetLargestPossibleRegion());
    evec_image->Allocate();
    evec_image->SetOrigin(dt_image->GetOrigin());
    evec_image->SetDirection(dt_image->GetDirection());
    evec_image->SetSpacing(dt_image->GetSpacing());
    EVecType zerovec;
    zerovec.set_identity();
    evec_image->FillBuffer(zerovec);



    this->eval_image = EValImageType::New();
    eval_image->SetRegions(dt_image->GetLargestPossibleRegion());
    eval_image->Allocate();
    eval_image->SetOrigin(dt_image->GetOrigin());
    eval_image->SetDirection(dt_image->GetDirection());
    eval_image->SetSpacing(dt_image->GetSpacing());
    EValType zeroval; zeroval.Fill(0);
    eval_image->FillBuffer(zeroval);

    itk::ImageRegionIteratorWithIndex<EVecImageType> it(evec_image,evec_image->GetLargestPossibleRegion());
    while(!it.IsAtEnd())
    {
        EVecImageType::IndexType ind=it.GetIndex();
        DTType vec=  dt_image->GetPixel(ind);

        if(vec[0]+ vec[3]+ vec[5] !=0)
        {


            EVecType tensor;
            tensor(0,0)= vec[0]; tensor(0,1)= vec[1]; tensor(0,2)= vec[2];
            tensor(1,0)= vec[1]; tensor(1,1)= vec[3]; tensor(1,2)= vec[4];
            tensor(2,0)= vec[2]; tensor(2,1)= vec[4]; tensor(2,2)= vec[5];


            vnl_symmetric_eigensystem<double> eig(tensor);

            EVecType evec;
            evec.set_column(0, eig.V.get_column(2));
            evec.set_column(1, eig.V.get_column(1));
            evec.set_column(2, eig.V.get_column(0));

            double mdet= vnl_determinant<double>( evec);
            if(mdet<0)
            {
                evec.set_column(2, -1.* evec.get_column(2));
            }

            evec=evec.transpose();

            EValType eval;

            if(eig.D(0,0)< 0)
                eig.D(0,0)=0.000000000001;
            if(eig.D(1,1)< 0)
                eig.D(1,1)=0.000000000001;
            if(eig.D(2,2)< 0)
                eig.D(2,2)=0.000000000001;

            eval[2]= eig.D(0,0);
            eval[1]= eig.D(1,1);
            eval[0]= eig.D(2,2);

            it.Set(evec);
            eval_image->SetPixel(ind,eval);
        }
        else
        {
            eval_image->SetPixel(ind,zeroval);
            evec_image->SetPixel(ind,zerovec);
        }
        ++it;
    }

}



#endif
