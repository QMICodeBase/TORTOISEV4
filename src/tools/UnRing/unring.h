#ifndef _UNRING_H
#define _UNRING_H

#include "defines.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageDuplicator.h"
#include "fftw3.h"
#include <omp.h>

#include <iomanip>

#include "TORTOISE.h"
#include "../utilities/extract_3Dvolume_from_4D.h"
#include "../tools/ResampleDWIs/resample_dwis.h"

#include "itkConstantPadImageFilter.h"

struct my_plans_struct
{
    fftw_plan *p2d,*pinv2d,*p_tr2d,*pinv_tr2d;
    fftw_plan *p1d,*pinv1d,*ptr1d,*pinvtr1d;
};


#define PI  3.141592


void unring_1D(fftw_complex *data,int n, int numlines,int nsh,int minW, int maxW,my_plans_struct *my_plans,bool tr)
{
    fftw_complex *in, *out;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    fftw_complex *sh = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n *(2*nsh+1));
    fftw_complex *sh2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n *(2*nsh+1));

    double nfac = 1/double(n);

    int *shifts = (int*) malloc(sizeof(int)*(2*nsh+1));
    shifts[0] = 0;
    for (int j = 0; j < nsh;j++)
    {
        shifts[j+1] = j+1;
        shifts[1+nsh+j] = -(j+1);
    }

    double *TV1arr = new double[2*nsh+1];
    double *TV2arr = new double[2*nsh+1];

    for (int k = 0; k < numlines; k++)
    {        
        if(tr==0)
            fftw_execute_dft(*(my_plans->p1d),&(data[n*k]),sh);
        else
            fftw_execute_dft(*(my_plans->ptr1d),&(data[n*k]),sh);


        int maxn = (n%2 == 1)? (n-1)/2 : n/2 -1;

        for (int j = 1; j < 2*nsh+1; j++)
        {
            double phi = PI/double(n) * double(shifts[j])/double(nsh);
            fftw_complex u = {cos(phi),sin(phi)};
            fftw_complex e = {1,0};

            sh[j*n ][0] = sh[0][0];
            sh[j*n ][1] = sh[0][1];

            if (n%2 == 0)
            {
                sh[j*n + n/2][0] = 0;
                sh[j*n + n/2][1] = 0;
            }

            for (int l = 0; l < maxn; l++)
            {

                double tmp = e[0];
                e[0] = u[0]*e[0] - u[1]*e[1];
                e[1] = tmp*u[1] + u[0]*e[1];

                int L ;
                L = l+1;
                sh[j*n +L][0] = (e[0]*sh[L][0] - e[1]*sh[L][1]);
                sh[j*n +L][1] = (e[0]*sh[L][1] + e[1]*sh[L][0]);
                L = n-1-l;
                sh[j*n +L][0] = (e[0]*sh[L][0] + e[1]*sh[L][1]);
                sh[j*n +L][1] = (e[0]*sh[L][1] - e[1]*sh[L][0]);

            }
        }


        for (int j = 0; j < 2*nsh+1; j++)
        {
            if(tr==0)
                fftw_execute_dft(*(my_plans->pinv1d),&(sh[j*n]),&sh2[j*n]);
            else
                fftw_execute_dft(*(my_plans->pinvtr1d),&(sh[j*n]),&sh2[j*n]);

        }

        for (int j=0;j < 2*nsh+1;j++)
        {
            TV1arr[j] = 0;
            TV2arr[j] = 0;
            const int l = 0;
            for (int t = minW; t <= maxW;t++)
            {
                TV1arr[j] += fabs(sh2[j*n + (l-t+n)%n ][0] - sh2[j*n + (l-(t+1)+n)%n ][0]);
                TV1arr[j] += fabs(sh2[j*n + (l-t+n)%n ][1] - sh2[j*n + (l-(t+1)+n)%n ][1]);
                TV2arr[j] += fabs(sh2[j*n + (l+t+n)%n ][0] - sh2[j*n + (l+(t+1)+n)%n ][0]);
                TV2arr[j] += fabs(sh2[j*n + (l+t+n)%n ][1] - sh2[j*n + (l+(t+1)+n)%n ][1]);
            }
        }




        for(int l=0; l < n; l++)
        {
            double minTV = 999999999999;
            int minidx= 0;
            for (int j=0;j < 2*nsh+1;j++)
            {

                if (TV1arr[j] < minTV)
                {
                    minTV = TV1arr[j];
                    minidx = j;
                }
                if (TV2arr[j] < minTV)
                {
                    minTV = TV2arr[j];
                    minidx = j;
                }

                TV1arr[j] += fabs(sh2[j*n + (l-minW+1+n)%n ][0] - sh2[j*n + (l-(minW)+n)%n ][0]);
                TV1arr[j] -= fabs(sh2[j*n + (l-maxW+n)%n ][0] - sh2[j*n + (l-(maxW+1)+n)%n ][0]);
                TV2arr[j] += fabs(sh2[j*n + (l+maxW+1+n)%n ][0] - sh2[j*n + (l+(maxW+2)+n)%n ][0]);
                TV2arr[j] -= fabs(sh2[j*n + (l+minW+n)%n ][0] - sh2[j*n + (l+(minW+1)+n)%n ][0]);

                TV1arr[j] += fabs(sh2[j*n + (l-minW+1+n)%n ][1] - sh2[j*n +  (l-(minW)+n)%n ][1]);
                TV1arr[j] -= fabs(sh2[j*n + (l-maxW+n)%n ][1] - sh2[j*n + (l-(maxW+1)+n)%n ][1]);
                TV2arr[j] += fabs(sh2[j*n + (l+maxW+1+n)%n ][1] - sh2[j*n + (l+(maxW+2)+n)%n ][1]);
                TV2arr[j] -= fabs(sh2[j*n + (l+minW+n)%n ][1] - sh2[j*n + (l+(minW+1)+n)%n ][1]);

            }


            double a0r = sh2[minidx*n + (l-1+n)%n ][0];
            double a1r = sh2[minidx*n + l][0];
            double a2r = sh2[minidx*n + (l+1+n)%n ][0];
            double a0i = sh2[minidx*n + (l-1+n)%n ][1];
            double a1i = sh2[minidx*n + l][1];
            double a2i = sh2[minidx*n + (l+1+n)%n ][1];
            double s = double(shifts[minidx])/nsh/2;

            if (s>0)
            {
                data[k*n + l][0] =  (a1r*(1-s) + a0r*s)*nfac;
                data[k*n + l][1] =  (a1i*(1-s) + a0i*s)*nfac;
            }
            else
            {
                s = -s;
                data[k*n + l][0] =  (a1r*(1-s) + a2r*s)*nfac;
                data[k*n + l][1] =  (a1i*(1-s) + a2i*s)*nfac;
            }


        }
    }


     delete[] TV1arr;
     delete[] TV2arr;
     free(shifts);
     fftw_free(in);
     fftw_free(out);
     fftw_free(sh);
     fftw_free(sh2);
}


void unring_2d(fftw_complex *data1,fftw_complex *tmp2, const int *dim_sz, int nsh, int minW, int maxW, my_plans_struct *my_plans)
{
        double eps = 0;
        fftw_complex *tmp1 =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim_sz[0]*dim_sz[1]);
        fftw_complex *data2 =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim_sz[0]*dim_sz[1]);

        double nfac = 1./double(dim_sz[0]*dim_sz[1]);

        for (int k = 0 ; k < dim_sz[1];k++)
           for (int j = 0 ; j < dim_sz[0];j++)
           {
                data2[j*dim_sz[1]+k][0] = data1[k*dim_sz[0]+j][0];
                data2[j*dim_sz[1]+k][1] = data1[k*dim_sz[0]+j][1];
           }

        fftw_execute_dft(*(my_plans->p2d), data1,tmp1);   
        fftw_execute_dft(*(my_plans->p_tr2d), data2,tmp2);


        for (int k = 0 ; k < dim_sz[1];k++)
        {
            double ck = (1+cos(2*PI*(double(k)/dim_sz[1])))*0.5 +eps;
            for (int j = 0 ; j < dim_sz[0];j++)
            {
                double cj = (1+cos(2*PI*(double(j)/dim_sz[0])))*0.5 +eps;

                tmp1[k*dim_sz[0]+j][0] = nfac*(tmp1[k*dim_sz[0]+j][0] * ck) / (ck+cj);
                tmp1[k*dim_sz[0]+j][1] = nfac*(tmp1[k*dim_sz[0]+j][1] * ck) / (ck+cj);
                tmp2[j*dim_sz[1]+k][0] = nfac*(tmp2[j*dim_sz[1]+k][0] * cj) / (ck+cj);
                tmp2[j*dim_sz[1]+k][1] = nfac*(tmp2[j*dim_sz[1]+k][1] * cj) / (ck+cj);
            }
        }

        fftw_execute_dft(*(my_plans->pinv2d),tmp1,data1);
        fftw_execute_dft(*(my_plans->pinv_tr2d),tmp2,data2);


        unring_1D(data1,dim_sz[0],dim_sz[1],nsh,minW,maxW,my_plans,0);
        unring_1D(data2,dim_sz[1],dim_sz[0],nsh,minW,maxW,my_plans,1);


        fftw_execute_dft(*(my_plans->p2d),data1,tmp1);
        fftw_execute_dft(*(my_plans->p_tr2d),data2,tmp2);

        for (int k = 0 ; k < dim_sz[1];k++)
        {
            for (int j = 0 ; j < dim_sz[0];j++)
            {  
                tmp1[k*dim_sz[0]+j][0] = nfac*(tmp1[k*dim_sz[0]+j][0]  + tmp2[j*dim_sz[1]+k][0] ) ;
                tmp1[k*dim_sz[0]+j][1] = nfac*(tmp1[k*dim_sz[0]+j][1]  + tmp2[j*dim_sz[1]+k][1] ) ;     
            }
        }

        fftw_execute_dft(*(my_plans->pinv2d),tmp1,tmp2);

        fftw_free(data2);
        fftw_free(tmp1);
}





ImageType4D::Pointer UnRingFull(ImageType4D::Pointer input_img, int nsh=25, int minW=1,int maxW=3)
{
    typedef itk::ImageDuplicator<ImageType4D> DupType;
    DupType::Pointer dup= DupType::New();
    dup->SetInputImage(input_img);
    dup->Update();
    ImageType4D::Pointer output_img= dup->GetOutput();
    ImageType4D::SizeType sz= output_img->GetLargestPossibleRegion().GetSize();

    int dim_sz[4];
    dim_sz[0] = sz[0];
    dim_sz[1] = sz[1];
    dim_sz[2] = 1;
    dim_sz[3] = 1 ;


    auto stream = (TORTOISE::stream);
    if(stream)
        (*stream)<<  "Gibbs ringing correction of volume: " <<  std::flush;
    else
        std::cout<<  "Gibbs ringing correction of volume: " <<  std::flush;


    my_plans_struct my_plans;
    fftw_plan p = fftw_plan_dft_2d(dim_sz[1],dim_sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans.p2d = &p;
    fftw_plan pinv = fftw_plan_dft_2d(dim_sz[1],dim_sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.pinv2d= &pinv;
    fftw_plan p_tr = fftw_plan_dft_2d(dim_sz[0],dim_sz[1], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans.p_tr2d= &p_tr;
    fftw_plan pinv_tr = fftw_plan_dft_2d(dim_sz[0],dim_sz[1],  NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.pinv_tr2d= &pinv_tr;

    fftw_plan p1d = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1d = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan p1dtr = fftw_plan_dft_1d(sz[1], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1dtr = fftw_plan_dft_1d(sz[1], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.p1d=&p1d;
    my_plans.pinv1d=&pinv1d;
    my_plans.ptr1d=&p1dtr;
    my_plans.pinvtr1d= &pinv1dtr;


   #pragma omp parallel for
    for(int t=0;t<sz[3];t++)
    {
        TORTOISE::EnableOMPThread();
        #pragma omp critical
        {
            if(stream)
                (*stream)<<  t <<", "<< std::flush;
            else
                std::cout<<  t <<", "<< std::flush;

        }

        fftw_complex *data_complex =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sz[0]*sz[1]);
        fftw_complex *res_complex  =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sz[0]*sz[1]);

        ImageType4D::IndexType index;
        index[3]=t;

        for (int z=0; z<sz[2]; z++)
        {
            index[2]=z;
            for (int x = 0 ; x < sz[0];x++)
            {
                index[0]=x;
                for (int y = 0 ; y < sz[1];y++)
                {
                    index[1]=y;

                    data_complex[sz[0]*y+x][0] = (double) input_img->GetPixel(index);
                    data_complex[sz[0]*y+x][1] = 0;
                }
            }
            unring_2d(data_complex,res_complex, dim_sz,nsh,minW,maxW,&my_plans);

            for (int x = 0 ; x < sz[0];x++)
            {
                index[0]=x;
                for (int y = 0 ; y < sz[1];y++)
                {
                    index[1]=y;
                    float val = res_complex[sz[0]*y+x][0];
                    output_img->SetPixel(index,val);
                }
            }
        }

        fftw_free(data_complex);
        fftw_free(res_complex);
        TORTOISE::DisableOMPThread();
    }



    fftw_destroy_plan(p);
    fftw_destroy_plan(pinv);
    fftw_destroy_plan(p_tr);
    fftw_destroy_plan(pinv_tr);

    fftw_destroy_plan(p1d);
    fftw_destroy_plan(pinv1d);
    fftw_destroy_plan(p1dtr);
    fftw_destroy_plan(pinv1dtr);

    if(stream)
        (*stream)<< std::endl;
    else
        std::cout<< std::endl;

    return output_img;
}



ImageType3D::Pointer UnRingFull(ImageType3D::Pointer input_img, my_plans_struct *my_plans, int nsh=25, int minW=1,int maxW=3)
{
    typedef itk::ImageDuplicator<ImageType3D> DupType;
    DupType::Pointer dup= DupType::New();
    dup->SetInputImage(input_img);
    dup->Update();
    ImageType3D::Pointer output_img= dup->GetOutput();
    ImageType3D::SizeType sz= output_img->GetLargestPossibleRegion().GetSize();

    int dim_sz[4];
    dim_sz[0] = sz[0];
    dim_sz[1] = sz[1];
    dim_sz[2] = 1;
    dim_sz[3] = 1 ;



    {
        fftw_complex *data_complex =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sz[0]*sz[1]);
        fftw_complex *res_complex  =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sz[0]*sz[1]);

        ImageType3D::IndexType index;

        for (int z=0; z<sz[2]; z++)
        {
            index[2]=z;
            for (int x = 0 ; x < sz[0];x++)
            {
                index[0]=x;
                for (int y = 0 ; y < sz[1];y++)
                {
                    index[1]=y;

                    data_complex[sz[0]*y+x][0] = (double) input_img->GetPixel(index);
                    data_complex[sz[0]*y+x][1] = 0;
                }
            }
            unring_2d(data_complex,res_complex, dim_sz,nsh,minW,maxW,my_plans);

            for (int x = 0 ; x < sz[0];x++)
            {
                index[0]=x;
                for (int y = 0 ; y < sz[1];y++)
                {
                    index[1]=y;
                    float val = res_complex[sz[0]*y+x][0];
                    output_img->SetPixel(index,val);
                }
            }
        }

        fftw_free(data_complex);
        fftw_free(res_complex);
    }

    return output_img;
}




std::vector<ImageType3D::Pointer> SplitImageRows(ImageType3D::Pointer img3d, int split_factor)
{

    std::vector<ImageType3D::Pointer> imgs;

    ImageType3D::SizeType sz= img3d->GetLargestPossibleRegion().GetSize();

    int minsz= ((int)sz[1])/split_factor;
    int rem = sz[1] -minsz*split_factor;

    for(int s=0;s<split_factor;s++)
    {
        ImageType3D::SizeType new_sz;
        new_sz[0]=sz[0];
        new_sz[2]=sz[2];
        new_sz[1]= minsz + (int)(s<rem);

        ImageType3D::IndexType start; start.Fill(0);
        ImageType3D::RegionType reg(start,new_sz);
        ImageType3D::Pointer new_img= ImageType3D::New();
        new_img->SetRegions(reg);
        new_img->Allocate();
        new_img->FillBuffer(0);
        imgs.push_back(new_img);
    }


    ImageType3D::IndexType ind3, nind3;
    for(int k=0;k<sz[2];k++)
    {
        ind3[2]=k;
        nind3[2]=k;
        for(int i=0;i<sz[0];i++)
        {
            ind3[0]=i;
            nind3[0]=i;
            for(int j=0;j<sz[1];j++)
            {
                ind3[1]=j;

                int img_id = j % split_factor;
                nind3[1]= (int)(j / split_factor);

                imgs[img_id]->SetPixel(nind3, img3d->GetPixel(ind3));
            }
        }
    }
    return imgs;
}



ImageType3D::Pointer CombineImageRows(std::vector<ImageType3D::Pointer> imgs)
{
    int cf= imgs.size();

    ImageType3D::SizeType sz;
    sz[0] = imgs[0]->GetLargestPossibleRegion().GetSize()[0];
    sz[2] = imgs[0]->GetLargestPossibleRegion().GetSize()[2];
    sz[1]=0;
    for(int c=0;c<cf;c++)
        sz[1]+=  imgs[c]->GetLargestPossibleRegion().GetSize()[1];

    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,sz);
    ImageType3D::Pointer new_img = ImageType3D::New();
    new_img->SetRegions(reg);
    new_img->Allocate();
    new_img->FillBuffer(0);

    for(int c=0;c<cf;c++)
    {
        sz = imgs[c]->GetLargestPossibleRegion().GetSize();

        ImageType3D::IndexType ind3, nind3;
        for(int k=0;k<sz[2];k++)
        {
            ind3[2]=k;
            nind3[2]=k;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;
                nind3[0]=i;
                for(int j=0;j<sz[1];j++)
                {
                    ind3[1]=j;
                    nind3[1]=j*cf+c;

                    new_img->SetPixel(nind3, imgs[c]->GetPixel(ind3));
                }
            }
        }
    }
    return new_img;
}




ImageType4D::Pointer UnRing78(ImageType4D::Pointer input_img, int nsh=25, int minW=1,int maxW=3)
{
    typedef itk::ImageDuplicator<ImageType4D> DupType;
    DupType::Pointer dup= DupType::New();
    dup->SetInputImage(input_img);
    dup->Update();
    ImageType4D::Pointer output_img= dup->GetOutput();
    ImageType4D::SizeType orig_sz= output_img->GetLargestPossibleRegion().GetSize();


    ImageType4D::SizeType lowerExtendRegion;
    lowerExtendRegion.Fill(0);

    ImageType4D::SizeType upperExtendRegion;
    upperExtendRegion.Fill(0);
    upperExtendRegion[1]= (4-(input_img->GetLargestPossibleRegion().GetSize()[1]%4))%4;

    using FilterType = itk::ConstantPadImageFilter<ImageType4D, ImageType4D>;
    FilterType::Pointer filter = FilterType::New();
    filter->SetInput(input_img);
    filter->SetPadLowerBound(lowerExtendRegion);
    filter->SetPadUpperBound(upperExtendRegion);
    filter->SetConstant(0);
    filter->Update();
    ImageType4D::Pointer padded_img4d= filter->GetOutput();


    ImageType4D::SizeType sz= padded_img4d->GetLargestPossibleRegion().GetSize();


    int dim_sz[4];
    dim_sz[0] = sz[0];
    dim_sz[1] = sz[1];
    dim_sz[2] = 1;
    dim_sz[3] = 1;


    auto stream = (TORTOISE::stream);
    if(stream)
        (*stream)<<  "Gibbs ringing correction of volume: " <<  std::flush;
    else
        std::cout<<  "Gibbs ringing correction of volume: " <<  std::flush;


    my_plans_struct my_plans_down;
    fftw_plan p_down = fftw_plan_dft_2d(dim_sz[1]*3/4,dim_sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans_down.p2d = &p_down;
    fftw_plan pinv_down = fftw_plan_dft_2d(dim_sz[1]*3/4,dim_sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans_down.pinv2d= &pinv_down;
    fftw_plan p_tr_down = fftw_plan_dft_2d(dim_sz[0],dim_sz[1]*3/4, NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans_down.p_tr2d= &p_tr_down;
    fftw_plan pinv_tr_down = fftw_plan_dft_2d(dim_sz[0],dim_sz[1]*3/4,  NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans_down.pinv_tr2d= &pinv_tr_down;

    fftw_plan p1d_down = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1d_down = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan p1dtr_down = fftw_plan_dft_1d(sz[1]*3/4, NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1dtr_down = fftw_plan_dft_1d(sz[1]*3/4, NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans_down.p1d=&p1d_down;
    my_plans_down.pinv1d=&pinv1d_down;
    my_plans_down.ptr1d=&p1dtr_down;
    my_plans_down.pinvtr1d= &pinv1dtr_down;


    my_plans_struct my_plans;
    fftw_plan p = fftw_plan_dft_2d(dim_sz[1],dim_sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans.p2d = &p;
    fftw_plan pinv = fftw_plan_dft_2d(dim_sz[1],dim_sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.pinv2d= &pinv;
    fftw_plan p_tr = fftw_plan_dft_2d(dim_sz[0],dim_sz[1], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans.p_tr2d= &p_tr;
    fftw_plan pinv_tr = fftw_plan_dft_2d(dim_sz[0],dim_sz[1],  NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.pinv_tr2d= &pinv_tr;

    fftw_plan p1d = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1d = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan p1dtr = fftw_plan_dft_1d(sz[1], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1dtr = fftw_plan_dft_1d(sz[1], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.p1d=&p1d;
    my_plans.pinv1d=&pinv1d;
    my_plans.ptr1d=&p1dtr;
    my_plans.pinvtr1d= &pinv1dtr;


    #pragma omp parallel for
    for(int v=0;v<sz[3];v++)
    {
        TORTOISE::EnableOMPThread();
        #pragma omp critical
        {
            if(stream)
                (*stream)<<  v <<", "<< std::flush;
            else
                std::cout<<  v <<", "<< std::flush;
        }

        ImageType3D::Pointer img3d= extract_3D_volume_from_4D(padded_img4d,v);


        std::vector<float> up_factors={1,3,1};
        ImageType3D::Pointer img3d_upsampled= resample_3D_image(img3d,std::vector<float>(),up_factors,"NN");
        std::vector<ImageType3D::Pointer> img3d_upsampled_split= SplitImageRows(img3d_upsampled,4);
        for(int s=0;s<img3d_upsampled_split.size();s++)
        {
            img3d_upsampled_split[s]= UnRingFull(img3d_upsampled_split[s],&my_plans_down, nsh, minW, maxW);
        }

        ImageType3D::Pointer combined_img= CombineImageRows(img3d_upsampled_split);
        up_factors[1]= 1./3.;
        ImageType3D::Pointer img3d_downsampled= resample_3D_image(combined_img,std::vector<float>(),up_factors,"NN");


        fftw_complex *data2 =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim_sz[0]*dim_sz[1]);
        double nfac = 1./double(dim_sz[0]*dim_sz[1]);

        ImageType3D::IndexType ind3;
        ImageType4D::IndexType ind4;
        ind4[3]=v;
        for (int k=0; k<sz[2]; k++)
        {
            ind4[2]=k;
            ind3[2]=k;
            for (int j = 0 ; j < sz[1];j++)
            {
                ind3[1]=j;
                for (int i = 0 ; i < sz[0];i++)
                {
                    ind3[0]=i;
                    data2[sz[1]*i+j][0] = (double) img3d_downsampled->GetPixel(ind3);
                    data2[sz[1]*i+j][1] = 0;
                }
            }

            unring_1D(data2,dim_sz[1],dim_sz[0],nsh,minW,maxW,&my_plans,1);


            for (int j = 0 ; j < sz[1];j++)
            {
                if(j<orig_sz[2])
                {
                    ind4[1]=j;
                    for (int i = 0 ; i < sz[0];i++)
                    {
                        ind4[0]=i;
                        float val = data2[sz[1]*i+j][0];
                        output_img->SetPixel(ind4,val);
                    }
                }
            }
        }

        fftw_free(data2);
        TORTOISE::DisableOMPThread();
    }

    fftw_destroy_plan(p);
    fftw_destroy_plan(pinv);
    fftw_destroy_plan(p_tr);
    fftw_destroy_plan(pinv_tr);

    fftw_destroy_plan(p1d);
    fftw_destroy_plan(pinv1d);
    fftw_destroy_plan(p1dtr);
    fftw_destroy_plan(pinv1dtr);

    fftw_destroy_plan(p_down);
    fftw_destroy_plan(pinv_down);
    fftw_destroy_plan(p_tr_down);
    fftw_destroy_plan(pinv_tr_down);

    fftw_destroy_plan(p1d_down);
    fftw_destroy_plan(pinv1d_down);
    fftw_destroy_plan(p1dtr_down);
    fftw_destroy_plan(pinv1dtr_down);


    if(stream)
        (*stream)<< std::endl;
    else
        std::cout<< std::endl;

    return output_img;

}


ImageType4D::Pointer UnRing68(ImageType4D::Pointer input_img, int nsh=25, int minW=1,int maxW=3)
{

    typedef itk::ImageDuplicator<ImageType4D> DupType;
    DupType::Pointer dup= DupType::New();
    dup->SetInputImage(input_img);
    dup->Update();
    ImageType4D::Pointer output_img= dup->GetOutput();
    output_img->FillBuffer(0);
    ImageType4D::SizeType orig_sz= output_img->GetLargestPossibleRegion().GetSize();


    ImageType4D::SizeType lowerExtendRegion;
    lowerExtendRegion.Fill(0);

    ImageType4D::SizeType upperExtendRegion;
    upperExtendRegion.Fill(0);
    upperExtendRegion[1]= (2-(input_img->GetLargestPossibleRegion().GetSize()[1]%4))%2;

    using FilterType = itk::ConstantPadImageFilter<ImageType4D, ImageType4D>;
    FilterType::Pointer filter = FilterType::New();
    filter->SetInput(input_img);
    filter->SetPadLowerBound(lowerExtendRegion);
    filter->SetPadUpperBound(upperExtendRegion);
    filter->SetConstant(0);
    filter->Update();
    ImageType4D::Pointer padded_img4d= filter->GetOutput();


    ImageType4D::SizeType sz= padded_img4d->GetLargestPossibleRegion().GetSize();

    int dim_sz[4];
    dim_sz[0] = sz[0];
    dim_sz[1] = sz[1];
    dim_sz[2] = 1;
    dim_sz[3] = 1 ;

    auto stream = (TORTOISE::stream);
    if(stream)
        (*stream)<<  "Gibbs ringing correction of volume: " <<  std::flush;
    else
        std::cout<<  "Gibbs ringing correction of volume: " <<  std::flush;


    my_plans_struct my_plans_down;
    fftw_plan p_down = fftw_plan_dft_2d(dim_sz[1]/2,dim_sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans_down.p2d = &p_down;
    fftw_plan pinv_down = fftw_plan_dft_2d(dim_sz[1]/2,dim_sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans_down.pinv2d= &pinv_down;
    fftw_plan p_tr_down = fftw_plan_dft_2d(dim_sz[0],dim_sz[1]/2, NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans_down.p_tr2d= &p_tr_down;
    fftw_plan pinv_tr_down = fftw_plan_dft_2d(dim_sz[0],dim_sz[1]/2,  NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans_down.pinv_tr2d= &pinv_tr_down;

    fftw_plan p1d_down = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1d_down = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan p1dtr_down = fftw_plan_dft_1d(sz[1]/2, NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1dtr_down = fftw_plan_dft_1d(sz[1]/2, NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans_down.p1d=&p1d_down;
    my_plans_down.pinv1d=&pinv1d_down;
    my_plans_down.ptr1d=&p1dtr_down;
    my_plans_down.pinvtr1d= &pinv1dtr_down;


    my_plans_struct my_plans;
    fftw_plan p = fftw_plan_dft_2d(dim_sz[1],dim_sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans.p2d = &p;
    fftw_plan pinv = fftw_plan_dft_2d(dim_sz[1],dim_sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.pinv2d= &pinv;
    fftw_plan p_tr = fftw_plan_dft_2d(dim_sz[0],dim_sz[1], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans.p_tr2d= &p_tr;
    fftw_plan pinv_tr = fftw_plan_dft_2d(dim_sz[0],dim_sz[1],  NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.pinv_tr2d= &pinv_tr;

    fftw_plan p1d = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1d = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan p1dtr = fftw_plan_dft_1d(sz[1], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1dtr = fftw_plan_dft_1d(sz[1], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.p1d=&p1d;
    my_plans.pinv1d=&pinv1d;
    my_plans.ptr1d=&p1dtr;
    my_plans.pinvtr1d= &pinv1dtr;


    #pragma omp parallel for
    for(int v=0;v<sz[3];v++)
    {
        TORTOISE::EnableOMPThread();

        #pragma omp critical
        {
            if(stream)
                (*stream)<<  v <<", "<< std::flush;
            else
                std::cout<<  v <<", "<< std::flush;
        }

        double nfac = 1/double(dim_sz[0]*dim_sz[1]);
        double eps = 0;


        fftw_complex *data1 =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sz[0]*sz[1]);
        fftw_complex *data2 =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sz[0]*sz[1]);
        fftw_complex *tmp1 =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim_sz[0]*dim_sz[1]);
        fftw_complex *tmp2 =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim_sz[0]*dim_sz[1]);
        fftw_complex *data2a =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sz[0]*sz[1]/2);
        fftw_complex *data2b =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sz[0]*sz[1]/2);


        ImageType4D::IndexType index;
        index[3]=v;
        for (int k=0; k<sz[2]; k++)
        {
            index[2]=k;
            for (int j = 0 ; j < sz[1];j++)
            {
                index[1]=j;
                for (int i = 0 ; i < sz[0];i++)
                {
                    index[0]=i;

                    data1[sz[0]*j+i][0] = (double) padded_img4d->GetPixel(index);
                    data1[sz[0]*j+i][1] = 0;
                    data2[sz[1]*i+j][0]=(double) padded_img4d->GetPixel(index);
                    data2[sz[1]*i+j][1]=0;
                }
            }

            fftw_execute_dft(*(my_plans.p2d), data1,tmp1);
            fftw_execute_dft(*(my_plans.p_tr2d), data2,tmp2);

            for (int j = 0 ; j < dim_sz[1];j++)
            {
                double cj = (1+cos(2*PI*(double(j)/dim_sz[1])))*0.5 +eps;
                for (int i = 0 ; i < dim_sz[0];i++)
                {
                    double ci = (1+cos(2*PI*(double(i)/dim_sz[0])))*0.5 +eps;


                    tmp1[j*dim_sz[0]+i][0] = nfac*(tmp1[j*dim_sz[0]+i][0] * cj) / (ci+cj);
                    tmp1[j*dim_sz[0]+i][1] = nfac*(tmp1[j*dim_sz[0]+i][1] * cj) / (ci+cj);
                    tmp2[i*dim_sz[1]+j][0] = nfac*(tmp2[i*dim_sz[1]+j][0] * ci) / (ci+cj);
                    tmp2[i*dim_sz[1]+j][1] = nfac*(tmp2[i*dim_sz[1]+j][1] * ci) / (ci+cj);
                }
            }

            fftw_execute_dft(*(my_plans.pinv2d),tmp1,data1);
            fftw_execute_dft(*(my_plans.pinv_tr2d),tmp2,data2);

            // bottom part of figure 4 in the paper is done with next line
            unring_1D(data1,dim_sz[0],dim_sz[1],nsh,minW,maxW,&my_plans,0);
            unring_1D(data2,dim_sz[1],dim_sz[0],nsh,minW,maxW,&my_plans,1);




            for (int yy = 0 ; yy < sz[0];yy++)
            {
                for (int xx = 0 ; xx < sz[1]/2;xx++)
                {
                    data2a[ yy*sz[1]/2+xx][0]= data2[yy*sz[1]+2*xx][0];
                    data2a[ yy*sz[1]/2+xx][1]= data2[yy*sz[1]+2*xx][1];
                    data2b[ yy*sz[1]/2+xx][0]= data2[yy*sz[1]+2*xx+1][0];
                    data2b[ yy*sz[1]/2+xx][1]= data2[yy*sz[1]+2*xx+1][1];
                }
            }

            unring_1D(data2a,dim_sz[1]/2,dim_sz[0],nsh,minW,maxW,&my_plans_down,1);
            unring_1D(data2b,dim_sz[1]/2,dim_sz[0],nsh,minW,maxW,&my_plans_down,1);


            for (int yy = 0 ; yy < sz[0];yy++)
            {
                for (int xx = 0 ; xx < sz[1]/2;xx++)
                {
                    data2[yy*sz[1]+2*xx][0]  =data2a[ yy*sz[1]/2+xx][0];
                    data2[yy*sz[1]+2*xx][1]  =data2a[ yy*sz[1]/2+xx][1];
                    data2[yy*sz[1]+2*xx+1][0]=data2b[ yy*sz[1]/2+xx][0];
                    data2[yy*sz[1]+2*xx+1][1]=data2b[ yy*sz[1]/2+xx][1];
                }
            }

            fftw_execute_dft(*(my_plans.p2d),data1,tmp1);
            fftw_execute_dft(*(my_plans.p_tr2d),data2,tmp2);

            for (int j = 0 ; j < dim_sz[1];j++)
            {
                for (int i = 0 ; i < dim_sz[0];i++)
                {
                    tmp1[j*dim_sz[0]+i][0] = nfac*(tmp1[j*dim_sz[0]+i][0]  + tmp2[i*dim_sz[1]+j][0] ) ;
                    tmp1[j*dim_sz[0]+i][1] = nfac*(tmp1[j*dim_sz[0]+i][1]  + tmp2[i*dim_sz[1]+j][1] ) ;
                }
            }

            fftw_execute_dft(*(my_plans.pinv2d),tmp1,tmp2);


            for (int j = 0 ; j < sz[1];j++)
            {
                if(j<orig_sz[1])
                {
                    index[1]=j;
                    for (int i = 0 ; i < sz[0];i++)
                    {
                        index[0]=i;
                        float val = tmp2[sz[0]*j+i][0];
                        output_img->SetPixel(index,val);
                    }
                }
            }

        }  //k:slice
        fftw_free(data1);
        fftw_free(data2);
        fftw_free(tmp1);
        fftw_free(tmp2);
        fftw_free(data2a);
        fftw_free(data2b);

        TORTOISE::DisableOMPThread();

    } //vol

    fftw_destroy_plan(p);
    fftw_destroy_plan(pinv);
    fftw_destroy_plan(p_tr);
    fftw_destroy_plan(pinv_tr);

    fftw_destroy_plan(p1d);
    fftw_destroy_plan(pinv1d);
    fftw_destroy_plan(p1dtr);
    fftw_destroy_plan(pinv1dtr);

    fftw_destroy_plan(p_down);
    fftw_destroy_plan(pinv_down);
    fftw_destroy_plan(p_tr_down);
    fftw_destroy_plan(pinv_tr_down);

    fftw_destroy_plan(p1d_down);
    fftw_destroy_plan(pinv1d_down);
    fftw_destroy_plan(p1dtr_down);
    fftw_destroy_plan(pinv1dtr_down);

    return output_img;

}




#endif

