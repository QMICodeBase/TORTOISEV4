#ifndef _RUNDRBUDDISTAGE_CXX
#define _RUNDRBUDDISTAGE_CXX

#include "run_drbuddi_stage.h"

#include "itkWindowConvergenceMonitoringFunction.h"
#include "itkRealTimeClock.h"
#include "../tools/ResampleDWIs/resample_dwis.h"
#include "itkGaussianOperator.h"


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

    this->def_F= InvertField(this->def_FINV);
    this->def_M= InvertField(this->def_MINV);


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

    #ifdef USECUDA
    for(int m=0;m<this->settings->metrics.size();m++)
    {
        if(resampled_smoothed_up_images[m]->getFloatdata().ptr !=nullptr)
            resampled_smoothed_up_images[m]->CreateTexture();

        if(resampled_smoothed_down_images[m]->getFloatdata().ptr !=nullptr)
            resampled_smoothed_down_images[m]->CreateTexture();
    }
    #endif


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


    float first_MSJac_met=1E10;
    int iter=0;
    bool converged=false;
    float curr_convergence =1;
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


        std::vector<float> metric_values;
        metric_values.resize(Nmetrics);
        for(int met=0; met< Nmetrics;met++)
        {
            CurrentImageType::Pointer warped_up_img = WarpImage(this->resampled_smoothed_up_images[met],this->def_FINV);
            CurrentImageType::Pointer warped_down_img = WarpImage(this->resampled_smoothed_down_images[met],this->def_MINV);

            float metric_value;
            CurrentFieldType::Pointer  updateFieldF_temp=nullptr,updateFieldM_temp=nullptr;

            if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCSK)
            {
                metric_value = ComputeMetric_CCSK(warped_up_img,warped_down_img, this->resampled_smoothed_str_images[met],updateFieldF_temp,updateFieldM_temp );
            }
            if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCJac)
            {

            }
            if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCJacS)
            {
                metric_value = ComputeMetric_CCJacS(warped_up_img,warped_down_img,this->resampled_smoothed_str_images[met],
                                                  this->def_FINV, this->def_MINV   ,
                                                   updateFieldF_temp,updateFieldM_temp, this->up_phase_vector,
                                                   Goper );                                

            }
            if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::MSJac)
            {
                metric_value = ComputeMetric_MSJac(warped_up_img,warped_down_img,
                                                  this->def_FINV, this->def_MINV   ,
                                                   updateFieldF_temp,updateFieldM_temp, this->up_phase_vector,
                                                   Goper );
                if(iter==1)
                {
                    first_MSJac_met=metric_value;
                }
            }
            if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CC)
            {
                metric_value = ComputeMetric_CC(warped_up_img,warped_down_img, updateFieldF_temp,updateFieldM_temp );
            }
            metric_values[met]=metric_value;
            all_ConvergenceMonitoring[met]->AddEnergyValue( metric_value );

            if(Nmetrics>1)
            {
                AddToUpdateField(updateFieldF,updateFieldF_temp,this->settings->metrics[met].weight);
                AddToUpdateField(updateFieldM,updateFieldM_temp,this->settings->metrics[met].weight);
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



        if(estimate_lr_per_iter)
        {
            int CCSK_id=-1;
            int MSJac_id=-1;
            int CCJac_id=-1;
            for(int met=0;met<Nmetrics;met++)
            {
                if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCJacS)
                {
                    CCJac_id=met;
                }
                if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCSK)
                {
                    CCSK_id=met;
                }
                if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::MSJac)
                {
                    MSJac_id=met;
                }
            }

            float mult=4;
            float lr=0.01/mult;
            float prev=1E10;
            float curr=1E9;

            while(curr<prev)
            {
                lr*=mult;
                if(lr>16)
                    break;
                ScaleUpdateField(updateFieldF,lr);
                ScaleUpdateField(updateFieldM,lr);

                CurrentFieldType::Pointer f2midtmp=nullptr,f2midtmp_inv=nullptr;
                CurrentFieldType::Pointer m2midtmp=nullptr,m2midtmp_inv=nullptr;


                f2midtmp  = ComposeFields(this->def_F,updateFieldF);
                f2midtmp = GaussianSmoothImage(f2midtmp,this->settings->total_gaussian_sigma);
                CurrentFieldType::Pointer f2midtotal_inv= InvertField(f2midtmp, this->def_FINV );

                m2midtmp  = ComposeFields(this->def_M,updateFieldM);
                m2midtmp = GaussianSmoothImage(m2midtmp,this->settings->total_gaussian_sigma);
                CurrentFieldType::Pointer m2midtotal_inv= InvertField(m2midtmp, this->def_MINV );

                if(this->settings->constrain)
                {
                    ContrainDefFields(f2midtotal_inv,m2midtotal_inv);
                }

                CurrentImageType::Pointer warped_up_img = nullptr;
                CurrentImageType::Pointer warped_down_img = nullptr;
                CurrentFieldType::Pointer  updateFieldF_temp=nullptr,updateFieldM_temp=nullptr;



                prev=curr;
                float curr1=0,curr2=0,curr3=0;

                if(MSJac_id!=-1)
                {
                    if (warped_up_img==nullptr)
                    {
                        warped_up_img = WarpImage(this->resampled_smoothed_up_images[MSJac_id],f2midtotal_inv);
                        warped_down_img = WarpImage(this->resampled_smoothed_down_images[MSJac_id],m2midtotal_inv);
                    }

                    if(CCJac_id==-1 && CCSK_id==-1)
                    {
                        curr1 = ComputeMetric_MSJac(warped_up_img,warped_down_img,              f2midtotal_inv, m2midtotal_inv   ,
                                                       updateFieldF_temp,updateFieldM_temp, this->up_phase_vector,
                                                       Goper );
                        curr1=curr1/first_MSJac_met -1;
                    }
                }
                if(CCJac_id!=-1)
                {
                    if (warped_up_img==nullptr)
                    {
                        warped_up_img = WarpImage(this->resampled_smoothed_up_images[CCJac_id],f2midtotal_inv);
                        warped_down_img = WarpImage(this->resampled_smoothed_down_images[CCJac_id],m2midtotal_inv);
                    }
                    curr2 = ComputeMetric_CCJacS(warped_up_img,warped_down_img,this->resampled_smoothed_str_images[CCJac_id],
                                                      f2midtotal_inv, m2midtotal_inv   ,
                                                       updateFieldF_temp,updateFieldM_temp, this->up_phase_vector,
                                                       Goper );
                }
                if(CCSK_id!=-1)
                {
                    if (warped_up_img==nullptr)
                    {
                        warped_up_img = WarpImage(this->resampled_smoothed_up_images[CCSK_id],f2midtotal_inv);
                        warped_down_img = WarpImage(this->resampled_smoothed_down_images[CCSK_id],m2midtotal_inv);
                    }
                    curr3 = ComputeMetric_CCSK(warped_up_img,warped_down_img, this->resampled_smoothed_str_images[CCSK_id],updateFieldF_temp,updateFieldM_temp );
                }
                curr=curr1+curr2+curr3;

            }
            lr/=mult;
            this->settings->learning_rate=lr;
        }



        ScaleUpdateField(updateFieldF,this->settings->learning_rate);
        ScaleUpdateField(updateFieldM,this->settings->learning_rate);


        {
            CurrentFieldType::Pointer f2midtmp=nullptr,f2midtmp_inv=nullptr;
            CurrentFieldType::Pointer m2midtmp=nullptr,m2midtmp_inv=nullptr;


            f2midtmp  = ComposeFields(this->def_F,updateFieldF);
            f2midtmp = GaussianSmoothImage(f2midtmp,this->settings->total_gaussian_sigma);
            CurrentFieldType::Pointer f2midtotal_inv= InvertField(f2midtmp, this->def_FINV );
            CurrentFieldType::Pointer f2midtotal = InvertField(f2midtotal_inv, f2midtmp );

            m2midtmp  = ComposeFields(this->def_M,updateFieldM);
            m2midtmp = GaussianSmoothImage(m2midtmp,this->settings->total_gaussian_sigma);
            CurrentFieldType::Pointer m2midtotal_inv= InvertField(m2midtmp, this->def_MINV );
            CurrentFieldType::Pointer m2midtotal = InvertField(m2midtotal_inv, m2midtmp );

            if(this->settings->constrain)
            {
                ContrainDefFields(f2midtotal_inv,m2midtotal_inv);

                CurrentFieldType::Pointer f2midtotal = InvertField(f2midtotal_inv, m2midtotal_inv );
                CurrentFieldType::Pointer m2midtotal = InvertField(m2midtotal_inv, f2midtotal_inv );
            }

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
            average_convergence+=current_conv*this->settings->metrics[i].weight;
        }
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

         (*stream)<<"Metric " << metric_values.size()-1<< ": " << std::setprecision(8)<<metric_values[metric_values.size()-1]<< " LR: " << this->settings->learning_rate<<std::endl;
         m_lastTotalTime = now;
         m_clock.Start();


    } //while iter

    this->settings->output_finv= this->def_FINV;
    this->settings->output_minv= this->def_MINV;
}



#endif
