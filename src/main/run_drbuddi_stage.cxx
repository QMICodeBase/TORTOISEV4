#ifndef _RUNDRBUDDISTAGE_CXX
#define _RUNDRBUDDISTAGE_CXX

#include "run_drbuddi_stage.h"

#include "itkWindowConvergenceMonitoringFunction.h"
#include "itkRealTimeClock.h"
#include "../tools/ResampleDWIs/resample_dwis.h"
#include "itkGaussianOperator.h"
#include "itkMultiplyImageFilter.h"

#ifdef USECUDA
    #include "../cuda_src/resample_image.h"
    #include "../cuda_src/gaussian_smooth_image.h"
    #include "../cuda_src/warp_image.h"
    #include "../cuda_src/cuda_image_utilities.h"
    #include "../cuda_src/compute_metric.h"
#else
    #include "drbuddi_image_utilities.h"
    #include "compute_metrics_msjac.h"
    #include "compute_metrics_cc.h"
    #include "compute_metrics_ccsk.h"
    #include "compute_metrics_ccjacs.h"
#endif



void DRBUDDIStage::CreateVirtualImage()
{
    if(this->settings->downsample_factor==1)
    {
        this->virtual_img= this->settings->metrics[0].up_img;
    }
    else
    {
            #ifdef USECUDA
            this->virtual_img= CurrentImageType::New();

            this->virtual_img->dir= this->settings->metrics[0].up_img->dir;
            this->virtual_img->spc.x = this->settings->metrics[0].up_img->spc.x * this->settings->downsample_factor;
            this->virtual_img->spc.y = this->settings->metrics[0].up_img->spc.y * this->settings->downsample_factor;
            this->virtual_img->spc.z = this->settings->metrics[0].up_img->spc.z * this->settings->downsample_factor;

            this->virtual_img->sz.x = static_cast<float>(
                        std::floor( (double)this->settings->metrics[0].up_img->sz.x / (double)this->settings->downsample_factor ) );
            if(this->virtual_img->sz.x<1)
                this->virtual_img->sz.x=1;

            this->virtual_img->sz.y = static_cast<float>(
                        std::floor( (double)this->settings->metrics[0].up_img->sz.y / (double)this->settings->downsample_factor ) );
            if(this->virtual_img->sz.y<1)
                this->virtual_img->sz.y=1;

            this->virtual_img->sz.z = static_cast<float>(
                        std::floor( (double)this->settings->metrics[0].up_img->sz.z / (double)this->settings->downsample_factor ) );
            if(this->virtual_img->sz.z<1)
                this->virtual_img->sz.z=1;

            float3 ici, oci;
            ici.x= ((double)this->settings->metrics[0].up_img->sz.x -1) /2.;
            ici.y= ((double)this->settings->metrics[0].up_img->sz.y -1) /2.;
            ici.z= ((double)this->settings->metrics[0].up_img->sz.z -1) /2.;

            oci.x= ((double)this->virtual_img->sz.x -1) /2.;
            oci.y= ((double)this->virtual_img->sz.y -1) /2.;
            oci.z= ((double)this->virtual_img->sz.z -1) /2.;

            float3 s1i1, s2i2, diff;
            s2i2.x = this->virtual_img->spc.x * oci.x;
            s2i2.y = this->virtual_img->spc.y * oci.y;
            s2i2.z = this->virtual_img->spc.z * oci.z;

            s1i1.x = this->settings->metrics[0].up_img->spc.x * ici.x;
            s1i1.y = this->settings->metrics[0].up_img->spc.y * ici.y;
            s1i1.z = this->settings->metrics[0].up_img->spc.z * ici.z;

            diff.x= s1i1.x- s2i2.x;
            diff.y= s1i1.y- s2i2.y;
            diff.z= s1i1.z- s2i2.z;

            this->virtual_img->orig.x = this->virtual_img->dir(0,0)* diff.x + this->virtual_img->dir(0,1)* diff.y + this->virtual_img->dir(0,2)* diff.z +  (double)this->settings->metrics[0].up_img->orig.x;
            this->virtual_img->orig.y = this->virtual_img->dir(1,0)* diff.x + this->virtual_img->dir(1,1)* diff.y + this->virtual_img->dir(1,2)* diff.z +  (double)this->settings->metrics[0].up_img->orig.y;
            this->virtual_img->orig.z = this->virtual_img->dir(2,0)* diff.x + this->virtual_img->dir(2,1)* diff.y + this->virtual_img->dir(2,2)* diff.z +  (double)this->settings->metrics[0].up_img->orig.z;
         #else
            this->virtual_img= CurrentImageType::New();

            this->virtual_img->SetDirection(this->settings->metrics[0].up_img->GetDirection());
            ImageType3D::SpacingType spc =this->settings->metrics[0].up_img->GetSpacing()*this->settings->downsample_factor;
            this->virtual_img->SetSpacing(spc);

            ImageType3D::SizeType sz;
            sz[0] = static_cast<int>( std::floor( (double)this->settings->metrics[0].up_img->GetLargestPossibleRegion().GetSize()[0] / (double)this->settings->downsample_factor ) );
            if(sz[0]<1)
                sz[0]=1;
            sz[1] = static_cast<int>( std::floor( (double)this->settings->metrics[0].up_img->GetLargestPossibleRegion().GetSize()[1] / (double)this->settings->downsample_factor ) );
            if(sz[1]<1)
                sz[1]=1;
            sz[2] = static_cast<int>( std::floor( (double)this->settings->metrics[0].up_img->GetLargestPossibleRegion().GetSize()[2] / (double)this->settings->downsample_factor ) );
            if(sz[2]<1)
                sz[2]=1;
            ImageType3D::IndexType start; start.Fill(0);
            ImageType3D::RegionType reg(start,sz);
            this->virtual_img->SetRegions(reg);

            float ici[3], oci[3];
            ici[0]= ((double)this->settings->metrics[0].up_img->GetLargestPossibleRegion().GetSize()[0] -1) /2.;
            ici[1]= ((double)this->settings->metrics[0].up_img->GetLargestPossibleRegion().GetSize()[1] -1) /2.;
            ici[2]= ((double)this->settings->metrics[0].up_img->GetLargestPossibleRegion().GetSize()[2] -1) /2.;

            oci[0]= ((double)sz[0] -1) /2.;
            oci[1]= ((double)sz[1] -1) /2.;
            oci[2]= ((double)sz[2] -1) /2.;

            float s1i1[3], s2i2[3], diff[3];
            s2i2[0] = spc[0] * oci[0];
            s2i2[1] = spc[1] * oci[1];
            s2i2[2] = spc[2] * oci[2];

            s1i1[0] = this->settings->metrics[0].up_img->GetSpacing()[0] * ici[0];
            s1i1[1] = this->settings->metrics[0].up_img->GetSpacing()[1] * ici[1];
            s1i1[2] = this->settings->metrics[0].up_img->GetSpacing()[2] * ici[2];

            diff[0]= s1i1[0]- s2i2[0];
            diff[1]= s1i1[1]- s2i2[1];
            diff[2]= s1i1[2]- s2i2[2];

            ImageType3D::PointType orig;
            orig[0] = this->virtual_img->GetDirection()(0,0)* diff[0] + this->virtual_img->GetDirection()(0,1)* diff[1] + this->virtual_img->GetDirection()(0,2)* diff[2] +  (double)this->settings->metrics[0].up_img->GetOrigin()[0];
            orig[1] = this->virtual_img->GetDirection()(1,0)* diff[0] + this->virtual_img->GetDirection()(1,1)* diff[1] + this->virtual_img->GetDirection()(1,2)* diff[2] +  (double)this->settings->metrics[0].up_img->GetOrigin()[1];
            orig[2] = this->virtual_img->GetDirection()(2,0)* diff[0] + this->virtual_img->GetDirection()(2,1)* diff[1] + this->virtual_img->GetDirection()(2,2)* diff[2] +  (double)this->settings->metrics[0].up_img->GetOrigin()[2];
            this->virtual_img->SetOrigin(orig);

         #endif
    }
}




void DRBUDDIStage::PreprocessImagesAndFields()
{
    CreateVirtualImage();

    #ifdef USECUDA
        if(this->def_FINV==nullptr || (this->def_FINV && this->def_FINV->getFloatdata().ptr==nullptr) )
        {
            this->def_FINV= CUDAIMAGE::New();
            this->def_FINV->orig=  this->virtual_img->orig;
            this->def_FINV->dir=  this->virtual_img->dir;
            this->def_FINV->spc=  this->virtual_img->spc;
            this->def_FINV->sz=  this->virtual_img->sz;
            this->def_FINV->components_per_voxel= 3;
            this->def_FINV->Allocate();

            this->def_F= CUDAIMAGE::New();
            this->def_F->orig=  this->virtual_img->orig;
            this->def_F->dir=  this->virtual_img->dir;
            this->def_F->spc=  this->virtual_img->spc;
            this->def_F->sz=  this->virtual_img->sz;
            this->def_F->components_per_voxel= 3;
            this->def_F->Allocate();
        }
        if(this->def_MINV==nullptr || (this->def_MINV && this->def_MINV->getFloatdata().ptr==nullptr) )
        {
            this->def_MINV= CUDAIMAGE::New();
            this->def_MINV->orig=  this->virtual_img->orig;
            this->def_MINV->dir=  this->virtual_img->dir;
            this->def_MINV->spc=  this->virtual_img->spc;
            this->def_MINV->sz=  this->virtual_img->sz;
            this->def_MINV->components_per_voxel= 3;
            this->def_MINV->Allocate();


            this->def_M= CUDAIMAGE::New();
            this->def_M->orig=  this->virtual_img->orig;
            this->def_M->dir=  this->virtual_img->dir;
            this->def_M->spc=  this->virtual_img->spc;
            this->def_M->sz=  this->virtual_img->sz;
            this->def_M->components_per_voxel= 3;
            this->def_M->Allocate();
        }
    #else
        if(this->def_FINV==nullptr )
        {
            this->def_FINV= DisplacementFieldType::New();
            this->def_FINV->SetOrigin(this->virtual_img->GetOrigin());
            this->def_FINV->SetSpacing(this->virtual_img->GetSpacing());
            this->def_FINV->SetDirection(this->virtual_img->GetDirection());
            this->def_FINV->SetRegions(this->virtual_img->GetLargestPossibleRegion());
            this->def_FINV->Allocate();
            DisplacementFieldType::PixelType zeros; zeros.Fill(0);
            this->def_FINV->FillBuffer(zeros);

            this->def_F= DisplacementFieldType::New();
            this->def_F->SetOrigin(this->virtual_img->GetOrigin());
            this->def_F->SetSpacing(this->virtual_img->GetSpacing());
            this->def_F->SetDirection(this->virtual_img->GetDirection());
            this->def_F->SetRegions(this->virtual_img->GetLargestPossibleRegion());
            this->def_F->Allocate();
            this->def_F->FillBuffer(zeros);
        }
        if(this->def_MINV==nullptr )
        {
            this->def_MINV= DisplacementFieldType::New();
            this->def_MINV->SetOrigin(this->virtual_img->GetOrigin());
            this->def_MINV->SetSpacing(this->virtual_img->GetSpacing());
            this->def_MINV->SetDirection(this->virtual_img->GetDirection());
            this->def_MINV->SetRegions(this->virtual_img->GetLargestPossibleRegion());
            this->def_MINV->Allocate();
            DisplacementFieldType::PixelType zeros; zeros.Fill(0);
            this->def_MINV->FillBuffer(zeros);

            this->def_M= DisplacementFieldType::New();
            this->def_M->SetOrigin(this->virtual_img->GetOrigin());
            this->def_M->SetSpacing(this->virtual_img->GetSpacing());
            this->def_M->SetDirection(this->virtual_img->GetDirection());
            this->def_M->SetRegions(this->virtual_img->GetLargestPossibleRegion());
            this->def_M->Allocate();
            this->def_M->FillBuffer(zeros);
        }
    #endif


    #ifdef USECUDA
    if(this->def_FINV && this->def_FINV->sz.x != this->virtual_img->sz.x)
    #else
    if(this->def_FINV && this->def_FINV->GetLargestPossibleRegion().GetSize()[0] != this->virtual_img->GetLargestPossibleRegion().GetSize()[0])
    #endif
    {
        this->def_FINV= ResampleImage(this->def_FINV, this->virtual_img);
        this->def_MINV= ResampleImage(this->def_MINV, this->virtual_img);
    }
    if(this->def_F==nullptr)
    {
        this->def_F= InvertField(this->def_FINV);
        this->def_M= InvertField(this->def_MINV);
    }


    resampled_smoothed_up_images.resize(this->settings->metrics.size());
    resampled_smoothed_down_images.resize(this->settings->metrics.size());
    resampled_smoothed_str_images.resize(this->settings->metrics.size());


    if(this->settings->img_smoothing_std!=0)
    {
        resampled_smoothed_up_images[0]= GaussianSmoothImage(this->settings->metrics[0].up_img,this->settings->img_smoothing_std*this->settings->img_smoothing_std);
        resampled_smoothed_down_images[0]= GaussianSmoothImage(this->settings->metrics[0].down_img,this->settings->img_smoothing_std*this->settings->img_smoothing_std);
        resampled_smoothed_str_images[0]= GaussianSmoothImage(this->settings->metrics[0].str_img,this->settings->img_smoothing_std*this->settings->img_smoothing_std);

        {
            for(int m=1;m<this->settings->metrics.size();m++)
            {
                bool ff=false;
                for(int m2=0;m2<m;m2++)
                {
                    if(this->settings->metrics[m].up_img == this->settings->metrics[m2].up_img )
                    {
                        resampled_smoothed_up_images[m]=resampled_smoothed_up_images[m2];
                        ff =true;
                        break;
                    }
                }
                if(!ff)
                {
                    resampled_smoothed_up_images[m]=GaussianSmoothImage(this->settings->metrics[m].up_img,this->settings->img_smoothing_std*this->settings->img_smoothing_std);
                }
            }
        }
        {
            for(int m=1;m<this->settings->metrics.size();m++)
            {
                bool found=false;
                for(int m2=0;m2<m;m2++)
                {
                    if(this->settings->metrics[m].down_img == this->settings->metrics[m2].down_img )
                    {
                        resampled_smoothed_down_images[m]=resampled_smoothed_down_images[m2];
                        found =true;
                        break;
                    }
                }
                if(!found)
                {
                    resampled_smoothed_down_images[m]=GaussianSmoothImage(this->settings->metrics[m].down_img,this->settings->img_smoothing_std*this->settings->img_smoothing_std);
                }
            }
        }

        {
            for(int m=1;m<this->settings->metrics.size();m++)
            {
                bool found=false;
                for(int m2=0;m2<m;m2++)
                {
                    if(this->settings->metrics[m].str_img == this->settings->metrics[m2].str_img )
                    {
                        resampled_smoothed_str_images[m]=resampled_smoothed_str_images[m2];
                        found =true;
                        break;
                    }
                }
                if(!found)
                {                    
                    resampled_smoothed_str_images[m]=GaussianSmoothImage(this->settings->metrics[m].str_img,this->settings->img_smoothing_std*this->settings->img_smoothing_std);
                }
            }
        }
    }
    else
    {
        for(int m=0;m<this->settings->metrics.size();m++)
        {
            #ifdef USECUDA
                resampled_smoothed_up_images[m]=CUDAIMAGE::New();
                resampled_smoothed_up_images[m]->DuplicateFromCUDAImage(this->settings->metrics[m].up_img);
                resampled_smoothed_down_images[m]=CUDAIMAGE::New();
                resampled_smoothed_down_images[m]->DuplicateFromCUDAImage(this->settings->metrics[m].down_img);

                if(this->settings->metrics[m].str_img)
                {
                    resampled_smoothed_str_images[m]=CUDAIMAGE::New();
                    resampled_smoothed_str_images[m]->DuplicateFromCUDAImage(this->settings->metrics[m].str_img);
                }
                else
                    resampled_smoothed_str_images[m]=nullptr;
            #else
                using DupType= itk::ImageDuplicator<ImageType3D>;
                {
                    DupType::Pointer dup= DupType::New();
                    dup->SetInputImage(this->settings->metrics[m].up_img);
                    dup->Update();
                    resampled_smoothed_up_images[m]=dup->GetOutput();
                }
                {
                    DupType::Pointer dup= DupType::New();
                    dup->SetInputImage(this->settings->metrics[m].down_img);
                    dup->Update();
                    resampled_smoothed_down_images[m]=dup->GetOutput();
                }

                if(this->settings->metrics[m].str_img)
                {
                    DupType::Pointer dup= DupType::New();
                    dup->SetInputImage(this->settings->metrics[m].str_img);
                    dup->Update();
                    resampled_smoothed_str_images[m]=dup->GetOutput();
                }
                else
                    resampled_smoothed_str_images[m]=nullptr;

            #endif
        }
    }


    #ifdef USECUDA
    if(resampled_smoothed_up_images[0]->sz.x != this->virtual_img->sz.x)
    #else
    if(resampled_smoothed_up_images[0]->GetLargestPossibleRegion().GetSize()[0] != this->virtual_img->GetLargestPossibleRegion().GetSize()[0])
    #endif
    {        
        resampled_smoothed_up_images[0]=ResampleImage(resampled_smoothed_up_images[0],this->virtual_img);
        resampled_smoothed_down_images[0]=ResampleImage(resampled_smoothed_down_images[0],this->virtual_img);
        resampled_smoothed_str_images[0]= ResampleImage(resampled_smoothed_str_images[0],this->virtual_img);

        for(int m=1;m<this->settings->metrics.size();m++)
        {
            bool found=false;
            for(int m2=0;m2<m;m2++)
            {
                if(this->settings->metrics[m].up_img == this->settings->metrics[m2].up_img )
                {
                    resampled_smoothed_up_images[m]=resampled_smoothed_up_images[m2];
                    found =true;
                    break;
                }
            }
            if(!found)
            {
                resampled_smoothed_up_images[m]=ResampleImage(resampled_smoothed_up_images[m],this->virtual_img);                
            }
        }

        for(int m=1;m<this->settings->metrics.size();m++)
        {
            bool found=false;
            for(int m2=0;m2<m;m2++)
            {
                if(this->settings->metrics[m].down_img == this->settings->metrics[m2].down_img )
                {
                    resampled_smoothed_down_images[m]=resampled_smoothed_down_images[m2];
                    found =true;
                    break;
                }
            }
            if(!found)
            {
                resampled_smoothed_down_images[m]=ResampleImage(resampled_smoothed_down_images[m],this->virtual_img);                
            }
        }

        for(int m=1;m<this->settings->metrics.size();m++)
        {
            bool found=false;
            for(int m2=0;m2<m;m2++)
            {
                if(this->settings->metrics[m].str_img == this->settings->metrics[m2].str_img )
                {
                    resampled_smoothed_str_images[m]=resampled_smoothed_str_images[m2];
                    found =true;
                    break;
                }
            }
            if(!found)
            {
                resampled_smoothed_str_images[m]=ResampleImage(resampled_smoothed_str_images[m],this->virtual_img);
            }
        }

    }


}



float DRBUDDIStage::ComputeBeta(CurrentFieldType::Pointer cfield,CurrentFieldType::Pointer pfield)
{
#ifdef USECUDA
    auto pmfield= MultiplyImage(pfield,-1);
    auto diff= AddImages(cfield,pmfield);
    auto nom= SumImage(MultiplyImages(cfield,diff));
    auto denom = SumImage(MultiplyImages(pfield,pfield));
#else
    double nom=0,denom=0;
    itk::ImageRegionIteratorWithIndex<CurrentFieldType> it(cfield,cfield->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();

        auto diff = cfield->GetPixel(ind3) - pfield->GetPixel(ind3);
        nom+=cfield->GetPixel(ind3)*diff;
        denom+= pfield->GetPixel(ind3) * pfield->GetPixel(ind3);
    }
#endif
    float Bpr=0;
    if(denom!=0)
        Bpr=nom/denom;

    return (0 < Bpr) ? Bpr : 0;
}



void DRBUDDIStage::RunDRBUDDIStage()
{
    int Nmetrics= this->settings->metrics.size();

    // Monitor the convergence
    typedef itk::Function::WindowConvergenceMonitoringFunction<float> ConvergenceMonitoringType;
    std::vector<typename ConvergenceMonitoringType::Pointer> all_ConvergenceMonitoring;
    all_ConvergenceMonitoring.resize(Nmetrics);

    for(int i=0;i<Nmetrics;i++)
    {
        all_ConvergenceMonitoring[i]=ConvergenceMonitoringType::New();
        all_ConvergenceMonitoring[i]->SetWindowSize( 10);
    }

    (*stream) << "Iteration,convergenceValue,ITERATION_TIME, metricValues..."     << std::endl;

    m_clock.Start();
    m_clock.Stop();
    const itk::RealTimeClock::TimeStampType now = m_clock.GetTotal();
    itk::RealTimeClock::TimeStampType  m_lastTotalTime = now;
    m_clock.Start();


    #ifdef USECUDA
    for(int m=0;m<this->settings->metrics.size();m++)
    {    
        if(resampled_smoothed_up_images[m]->getFloatdata().ptr !=nullptr)
            resampled_smoothed_up_images[m]->CreateTexture();

        if(resampled_smoothed_down_images[m]->getFloatdata().ptr !=nullptr)
            resampled_smoothed_down_images[m]->CreateTexture();
    }
    #endif

    int iter=0;
    bool converged=false;
    float curr_convergence =1;

    CurrentFieldType::Pointer prev_updateFieldF= nullptr,prev_updateFieldM= nullptr;
    CurrentFieldType::Pointer conjugateFieldF= nullptr,conjugateFieldM= nullptr;

    while( iter++ < this->settings->niter && !converged )
    {
        CurrentFieldType::Pointer updateFieldF= nullptr,updateFieldM= nullptr;

        if(Nmetrics>1)
        {
            updateFieldF= CurrentFieldType::New();
            updateFieldM= CurrentFieldType::New();
            #ifdef USECUDA
                updateFieldF->sz = this->def_FINV->sz;
                updateFieldF->dir = this->def_FINV->dir;
                updateFieldF->orig = this->def_FINV->orig;
                updateFieldF->spc = this->def_FINV->spc;
                updateFieldF->components_per_voxel = this->def_FINV->components_per_voxel;
                updateFieldF->Allocate();

                updateFieldM->sz = this->def_FINV->sz;
                updateFieldM->dir = this->def_FINV->dir;
                updateFieldM->orig = this->def_FINV->orig;
                updateFieldM->spc = this->def_FINV->spc;
                updateFieldM->components_per_voxel = this->def_FINV->components_per_voxel;
                updateFieldM->Allocate();
            #else
                updateFieldF->SetRegions(def_FINV->GetLargestPossibleRegion());
                updateFieldF->SetDirection(this->def_FINV->GetDirection());
                updateFieldF->SetOrigin(this->def_FINV->GetOrigin());
                updateFieldF->SetSpacing(this->def_FINV->GetSpacing());
                updateFieldF->Allocate();
                CurrentFieldType::PixelType zero; zero.Fill(0);
                updateFieldF->FillBuffer(zero);

                updateFieldM->SetRegions(def_FINV->GetLargestPossibleRegion());
                updateFieldM->SetDirection(this->def_FINV->GetDirection());
                updateFieldM->SetOrigin(this->def_FINV->GetOrigin());
                updateFieldM->SetSpacing(this->def_FINV->GetSpacing());
                updateFieldM->Allocate();
                updateFieldM->FillBuffer(zero);
            #endif
        }

        itk::GaussianOperator<float,3> Goper;
        Goper.SetDirection(phase_id);
        Goper.SetVariance(this->settings->update_gaussian_sigma);
        Goper.SetMaximumError(0.001);
        Goper.SetMaximumKernelWidth(31);
        Goper.CreateDirectional();


        CurrentFieldType::Pointer tot_finv= this->def_FINV;
        CurrentFieldType::Pointer tot_minv= this->def_MINV;
        if(settings->init_finv_const!=nullptr)
        {
            tot_finv= ComposeFields(settings->init_finv_const,this->def_FINV);
            tot_minv= ComposeFields(settings->init_minv_const,this->def_MINV);
        }


        CurrentImageType::Pointer str_img=nullptr;
        std::vector<float> metric_values;
        metric_values.resize(Nmetrics);
        for(int met=0; met< Nmetrics;met++)
        {
            CurrentImageType::Pointer warped_up_img = WarpImage(this->resampled_smoothed_up_images[met],tot_finv);
            CurrentImageType::Pointer warped_down_img = WarpImage(this->resampled_smoothed_down_images[met],tot_minv);

            float metric_value=0;            
            CurrentFieldType::Pointer  updateFieldF_temp=nullptr,updateFieldM_temp=nullptr;

            if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCSK)
            {
                metric_value = ComputeMetric_CCSK(warped_up_img,warped_down_img, this->resampled_smoothed_str_images[met],updateFieldF_temp,updateFieldM_temp );
                if(!str_img)
                    str_img=this->resampled_smoothed_str_images[met];
            }

            if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCJacS)
            {
                {
                    metric_value = ComputeMetric_CCJacS(warped_up_img,warped_down_img,this->resampled_smoothed_str_images[met],
                                                              tot_finv, tot_minv,
                                                              updateFieldF_temp,  updateFieldM_temp,
                                                              this->up_phase_vector,
                                                              Goper );
                }

            }
            if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::MSJac)
            {


                    metric_value = ComputeMetric_MSJac(warped_up_img,warped_down_img,
                                                   tot_finv , tot_minv ,
                                                   updateFieldF_temp,  updateFieldM_temp,
                                                   this->up_phase_vector,
                                                   Goper );


            }
            if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CC)
            {
                metric_value = ComputeMetric_CC(warped_up_img,warped_down_img, updateFieldF_temp,updateFieldM_temp );
            }
            metric_values[met]=metric_value;
            all_ConvergenceMonitoring[met]->AddEnergyValue( metric_value );



            if(Nmetrics>1)
            {
                float mlt=1;
              //  if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCSK || this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCJacS)
                //    mlt*=100;
                AddToUpdateField(updateFieldF,updateFieldF_temp,this->settings->metrics[met].weight*mlt);
                AddToUpdateField(updateFieldM,updateFieldM_temp,this->settings->metrics[met].weight*mlt);
            }
            else
            {
                updateFieldF=updateFieldF_temp;
                updateFieldM=updateFieldM_temp;
            }

        } //metric loop

        updateFieldF=GaussianSmoothImage(updateFieldF,this->settings->update_gaussian_sigma);
        updateFieldM=GaussianSmoothImage(updateFieldM,this->settings->update_gaussian_sigma);

        if(this->settings->restrct)
        {
            RestrictPhase(updateFieldF,this->up_phase_vector);
            RestrictPhase(updateFieldM,this->down_phase_vector);
        }

        float best_lr=this->settings->learning_rate;
        float beta_f=0,beta_m=0;
        #ifdef USECUDA
            ScaleUpdateField(updateFieldF,1);
            ScaleUpdateField(updateFieldM,1);

            if(prev_updateFieldF)
            {
                beta_f= ComputeBeta(updateFieldF,prev_updateFieldF);
                beta_m= ComputeBeta(updateFieldM,prev_updateFieldM);

                if(beta_f>2)
                    beta_f=1;
                if(beta_m>2)
                    beta_m=1;
            }

            prev_updateFieldF=CurrentFieldType::New();
            prev_updateFieldF->DuplicateFromCUDAImage(updateFieldF);
            prev_updateFieldM=CurrentFieldType::New();
            prev_updateFieldM->DuplicateFromCUDAImage(updateFieldM);

            if(conjugateFieldF)
            {
                conjugateFieldF=MultiplyImage(conjugateFieldF,beta_f);
                conjugateFieldF=AddImages(conjugateFieldF,updateFieldF);
                conjugateFieldM=MultiplyImage(conjugateFieldM,beta_m);
                conjugateFieldM=AddImages(conjugateFieldM,updateFieldM);
            }
            else
            {
                conjugateFieldF=CurrentFieldType::New();
                conjugateFieldF->DuplicateFromCUDAImage(updateFieldF);
                conjugateFieldM=CurrentFieldType::New();
                conjugateFieldM->DuplicateFromCUDAImage(updateFieldM);
            }

            float best_mv1=std::numeric_limits<float>::max();
            float best_mv2=std::numeric_limits<float>::max();
            for(float lr=this->settings->learning_rate/4;lr<= this->settings->learning_rate; lr*=2)
            {
                ScaleUpdateField(conjugateFieldF,lr);
                ScaleUpdateField(conjugateFieldM,lr);

                {
                    CurrentFieldType::Pointer f2midtmp=nullptr,f2midtmp_inv=nullptr;
                    CurrentFieldType::Pointer m2midtmp=nullptr,m2midtmp_inv=nullptr;


                    f2midtmp  = ComposeFields(this->def_F,conjugateFieldF);
                    f2midtmp = GaussianSmoothImage(f2midtmp,this->settings->total_gaussian_sigma);
                    CurrentFieldType::Pointer f2midtotal_inv= InvertField(f2midtmp, this->def_FINV );

                    m2midtmp  = ComposeFields(this->def_M,conjugateFieldM);
                    m2midtmp = GaussianSmoothImage(m2midtmp,this->settings->total_gaussian_sigma);
                    CurrentFieldType::Pointer m2midtotal_inv= InvertField(m2midtmp, this->def_MINV );

                    if(this->settings->constrain)
                    {
                        ContrainDefFields(f2midtotal_inv,m2midtotal_inv);
                    }

                    if(settings->init_finv_const!=nullptr)
                    {
                        f2midtotal_inv= ComposeFields(settings->init_finv_const,f2midtotal_inv);
                        m2midtotal_inv= ComposeFields(settings->init_minv_const,m2midtotal_inv);
                    }

                    CurrentFieldType::Pointer dummyf,dummym;
                    CurrentImageType::Pointer warped_up_temp = WarpImage(this->resampled_smoothed_up_images[0],f2midtotal_inv);
                    CurrentImageType::Pointer warped_down_temp = WarpImage(this->resampled_smoothed_down_images[0],m2midtotal_inv);



                    float mv1,mv2;
                    if(str_img)
                    {

                        //mv2=ComputeMetric_CCSK(warped_up_temp,warped_down_temp, str_img,dummyf,dummym );
                        mv2 = ComputeMetric_CCJacS(warped_up_temp,warped_down_temp,str_img,
                                                   f2midtotal_inv , m2midtotal_inv ,
                                                   dummyf,  dummym,
                                                   this->up_phase_vector,
                                                   Goper );

                        if(mv2<best_mv2)
                        {
                            best_mv2=mv2;
                            best_lr=lr;
                        }
                    }
                    else
                    {
                        mv1 = ComputeMetric_MSJac(warped_up_temp,warped_down_temp,
                                                       f2midtotal_inv , m2midtotal_inv ,
                                                       dummyf,  dummym,
                                                       this->up_phase_vector,
                                                       Goper );
                        if(mv1<best_mv1)
                        {
                            best_mv1=mv1;
                            best_lr=lr;
                        }
                    }
                }
            }
        #else
            conjugateFieldF=updateFieldF;
            conjugateFieldM=updateFieldM;
        #endif


        ScaleUpdateField(conjugateFieldF,best_lr);
        ScaleUpdateField(conjugateFieldM,best_lr);

        {
            CurrentFieldType::Pointer f2midtmp=nullptr,f2midtmp_inv=nullptr;
            CurrentFieldType::Pointer m2midtmp=nullptr,m2midtmp_inv=nullptr;


            f2midtmp  = ComposeFields(this->def_F,conjugateFieldF);
            f2midtmp = GaussianSmoothImage(f2midtmp,this->settings->total_gaussian_sigma);
            CurrentFieldType::Pointer f2midtotal_inv= InvertField(f2midtmp, this->def_FINV );

            m2midtmp  = ComposeFields(this->def_M,conjugateFieldM);
            m2midtmp = GaussianSmoothImage(m2midtmp,this->settings->total_gaussian_sigma);
            CurrentFieldType::Pointer m2midtotal_inv= InvertField(m2midtmp, this->def_MINV );

            if(this->settings->constrain)
            {
                ContrainDefFields(f2midtotal_inv,m2midtotal_inv);
            }
            CurrentFieldType::Pointer f2midtotal = InvertField(f2midtotal_inv, m2midtotal_inv );
            CurrentFieldType::Pointer m2midtotal = InvertField(m2midtotal_inv, f2midtotal_inv );

            this->def_FINV=f2midtotal_inv;
            this->def_F=f2midtotal;
            this->def_MINV=m2midtotal_inv;
            this->def_M=m2midtotal;
        }




        double average_convergence=0;
        double prev_conv= curr_convergence;
        float tot_w=0;
        for(int i=0;i<Nmetrics;i++)
        {
            double current_conv= all_ConvergenceMonitoring[i]->GetConvergenceValue();
            float curr_w= this->settings->metrics[i].weight * this->settings->metrics[i].weight;
            tot_w +=curr_w;
            average_convergence+=current_conv*curr_w;
        }
        average_convergence/=tot_w;
        curr_convergence= average_convergence;

        if( (0.7*curr_convergence+0.3*prev_conv) < 1E-6)
        {
            converged = true;
        }


        m_clock.Stop();
        const itk::RealTimeClock::TimeStampType now = m_clock.GetTotal();

        (*stream) << " "; // if the output of current iteration is written to disk, and star
                // will appear before line, else a free space will be printed to keep visual alignment.

        (*stream)<< std::setw(5) << iter << ", conv: "<<std::scientific << std::setprecision(5) << curr_convergence <<", time: "
                       <<std::fixed << std::setprecision(2) << (now - m_lastTotalTime) <<"s, ";

         for(int i=0;i<metric_values.size()-1;i++)
             (*stream)<<"Metric " << i<< ": " << std::setprecision(8)<<metric_values[i]<<", ";

         (*stream)<<"Metric " << metric_values.size()-1<< ": " << std::setprecision(8)<<metric_values[metric_values.size()-1]<< " LR: " << best_lr<<" betaf: "<<beta_f << " beta_m: "<<beta_m <<std::endl;
         m_lastTotalTime = now;
         m_clock.Start();


    } //while iter

    this->settings->output_finv= this->def_FINV;
    this->settings->output_minv= this->def_MINV;
    if(settings->init_finv_const!=nullptr)
    {
        this->settings->output_finv= ComposeFields(settings->init_finv_const,this->def_FINV);
        this->settings->output_minv= ComposeFields(settings->init_minv_const,this->def_MINV);
    }

}



#endif
