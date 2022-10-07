#ifndef _DWIDENOISE_H
#define _DWIDENOISE_H

#include "defines.h"
#include "itkImageDuplicator.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "TORTOISE.h"

#include <Eigen/Dense>

using Eigen::MatrixXd;

ImageType4D::Pointer DWIDenoise(ImageType4D::Pointer input_img, ImageType3D::Pointer &noise_img,double &noise_mean,bool process_all_image=true,int desired_extent=0)
{

    ImageType4D::Pointer output_img=nullptr;


    if(process_all_image)
    {
        typedef itk::ImageDuplicator<ImageType4D> DuplicatorType;
        DuplicatorType::Pointer dup1= DuplicatorType::New();
        dup1->SetInputImage(input_img);
        dup1->Update();
        output_img=dup1->GetOutput();
        output_img->FillBuffer(0);
    }


    ImageType4D::SizeType sizes= input_img->GetLargestPossibleRegion().GetSize();
    int Nvols=sizes[3];

    double croot= pow((double)Nvols, (double)(1./3.));

    int curr_extent = std::max(5, (int)std::round(croot));
    if(curr_extent %2 ==0)
        curr_extent--;

    if(curr_extent>9)
        curr_extent=9;

    if(desired_extent!=0)
        curr_extent=desired_extent;

    int min_sz= std::min(std::min(sizes[0],sizes[1]),sizes[2]);



    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::SizeType sz;
    sz[0]=sizes[0];
    sz[1]=sizes[1];
    sz[2]=sizes[2];
    ImageType3D::RegionType reg(start,sz);
    ImageType3D::DirectionType dir,actual_dir;
    ImageType3D::PointType orig,actual_orig;
    dir(0,0)=input_img->GetDirection()(0,0);dir(0,1)=input_img->GetDirection()(0,1);dir(0,2)=input_img->GetDirection()(0,2);
    dir(1,0)=input_img->GetDirection()(1,0);dir(1,1)=input_img->GetDirection()(1,1);dir(1,2)=input_img->GetDirection()(1,2);
    dir(2,0)=input_img->GetDirection()(2,0);dir(2,1)=input_img->GetDirection()(2,1);dir(2,2)=input_img->GetDirection()(2,2);

    orig[0]=input_img->GetOrigin()[0];
    orig[1]=input_img->GetOrigin()[1];
    orig[2]=input_img->GetOrigin()[2];


    ImageType3D::SpacingType spc;
    spc[0]=input_img->GetSpacing()[0];
    spc[1]=input_img->GetSpacing()[1];
    spc[2]=input_img->GetSpacing()[2];


    noise_img = ImageType3D::New();
    noise_img->SetRegions(reg);
    noise_img->Allocate();
    noise_img->FillBuffer(0);
    noise_img->SetDirection(dir);
    noise_img->SetOrigin(orig);
    noise_img->SetSpacing(spc);


    if(min_sz<5)
    {
        noise_mean=0;
        return input_img;
    }



    std::vector<int> extent;
    extent.push_back(curr_extent/2);
    extent.push_back(curr_extent/2);
    extent.push_back(curr_extent/2);

    int m= input_img->GetLargestPossibleRegion().GetSize()[3];
    int n = curr_extent*curr_extent*curr_extent;
    int r=((m<n) ? m : n);


   int k_start=0;
   int k_end=sizes[2];

   if(process_all_image==false)
   {
       k_start= std::max(0,(int)sizes[2]/2 -3);
       k_end= std::min((int)sizes[2]/2 +3,(int)sizes[2]);
   }


   #ifdef NOTORTOISE
       std::cout<<"Denosing window radius: "<< extent[0]<<std::endl;
   #else
       auto stream = (TORTOISE::stream);
       (*stream)<<"Denosing window radius: "<< extent[0]<<std::endl;
   #endif

    #pragma omp parallel for
    for(int k=k_start;k<k_end;k++)
    {
        #ifndef NOTORTOISE
        TORTOISE::EnableOMPThread();
        #endif

        Eigen::MatrixXf X(m,n);

        ImageType4D::IndexType index;
        index[2]=k;

        for(int j=0;j<sizes[1];j++)
        {
            index[1]=j;


          for(int i=0;i<sizes[0];i++)
          {
                index[0]=i;

                X.fill(0);

                int cnt=-1;
                for(int kk=k-extent[2];kk<=k+extent[2];kk++)
                {
                    ImageType4D::IndexType temp_index;
                    temp_index[2]=kk;

                    for(int jj=j-extent[1];jj<=j+extent[1];jj++)
                    {
                        temp_index[1]=jj;

                        for(int ii=i-extent[0];ii<=i+extent[0];ii++)
                        {
                            temp_index[0]=ii;
                            cnt++;

                            if(ii>=0 && ii<sizes[0] && jj>=0 && jj<sizes[1] && kk>=0 && kk<sizes[2])
                            {
                                for(int vol=0;vol<sizes[3];vol++)
                                {
                                    temp_index[3]=vol;
                                    X(vol,cnt)= input_img->GetPixel(temp_index);
                                }
                            }
                        }
                    }
                }
                // Compute Eigendecomposition:
                Eigen::MatrixXf XtX (r,r);
                if (m <= n)
                  XtX.template triangularView<Eigen::Lower>() = X * X.transpose();
                else
                  XtX.template triangularView<Eigen::Lower>() = X.transpose() * X;
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig (XtX);
                // eigenvalues provide squared singular values:
                Eigen::VectorXf s = eig.eigenvalues();

                s=s.reverse().eval()/n;


                int end=r;
                for(int ii=0;ii<s.size();ii++)
                {
                    if(s[ii]<=0)
                    {
                        end=ii;
                        break;
                    }
                }
                if(end==0)
                    end=1;
                int m2=end;

                Eigen::VectorXf cum_s(s.size());
                cum_s[m2-1]=0;
                for(int v=m2-2;v>=0;v--)
                    cum_s[v]= cum_s[v+1] + s[v+1];


                int p;
                for(p=0;p<m2-1;p++)
                {                    
                    double sum_lambda= cum_s[p];
                    double gamma= 1.*(m2-p-1)/n;
                    double sigma_hat_sq2 = (s[p+1] - s[m2-1])/4./sqrt(gamma);
                    double RHS= (m2-p-1) * sigma_hat_sq2;                  

                    if(sum_lambda >= RHS)                    
                        break;                 
                }

               int p_hat= p;
                double sigma2;
                if(m2==p_hat+1)
                    sigma2=0;
                else
                    sigma2=cum_s[p_hat]/(m2-p_hat-1);

                s.tail(p_hat+1).setOnes();
                s.head (r-p_hat-1).setZero();


                if (p_hat!= m2-1 )
                {
                    if(m<=n)
                    {
                        Eigen::MatrixXf aa=eig.eigenvectors() * ( s.asDiagonal() * ( eig.eigenvectors().adjoint() * X.col(n/2) ));
                        X.col (n/2) = aa;
                    }
                    else
                    {
                        Eigen::MatrixXf aa=X * ( eig.eigenvectors() * ( s.asDiagonal() * eig.eigenvectors().adjoint().col(n/2) ));
                        X.col (n/2) = aa;
                    }

                }

                if(process_all_image)
                {
                    for(int v=0;v<sizes[3];v++)
                    {
                        index[3]=v;
                        output_img->SetPixel(index, X(v,n/2) );
                    }
                }

                ImageType3D::IndexType ind3;
                ind3[0]=i;
                ind3[1]=j;
                ind3[2]=k;
                noise_img->SetPixel(ind3,sqrt(sigma2));
            }
        }
        #ifndef NOTORTOISE
        TORTOISE::DisableOMPThread();
        #endif
    }

    long long cnt=0;
    itk::ImageRegionIteratorWithIndex<ImageType3D> it(noise_img,noise_img->GetLargestPossibleRegion());
    while(!it.IsAtEnd())
    {
        if(it.Get()!=0)
        {
            cnt++;
            noise_mean+=it.Get();
        }
        ++it;
    }
    if(cnt==0)
        noise_mean=0;
    else
        noise_mean/=cnt;


    if(!process_all_image)
    {
        ImageType3D::IndexType start; start.Fill(0);
        ImageType3D::SizeType sz;
        sz[0]=sizes[0];
        sz[1]=sizes[1];
        sz[2]= k_start;
        ImageType3D::RegionType reg(start,sz);

        itk::ImageRegionIteratorWithIndex<ImageType3D> it2(noise_img,reg);
        it2.GoToBegin();
        while(!it2.IsAtEnd())
        {
            ImageType3D::IndexType curr_ind = it2.GetIndex();
            curr_ind[2]= k_start;
            float val= noise_img->GetPixel(curr_ind);
            it2.Set(val);
            ++it2;
        }


        ImageType3D::IndexType start2;
        start2[0]=0;
        start2[1]=0;
        start2[2]= k_end;

        ImageType3D::SizeType sz2;
        sz2[0]=sizes[0];
        sz2[1]=sizes[1];
        sz2[2]= sizes[2]- k_end;
        ImageType3D::RegionType reg2(start2,sz2);

        itk::ImageRegionIteratorWithIndex<ImageType3D> it3(noise_img,reg2);
        it3.GoToBegin();
        while(!it3.IsAtEnd())
        {
            ImageType3D::IndexType curr_ind = it3.GetIndex();
            curr_ind[2]= k_end-1;
            float val= noise_img->GetPixel(curr_ind);

            it3.Set(val);
            ++it3;
        }
    }

    return output_img;
}







#endif
