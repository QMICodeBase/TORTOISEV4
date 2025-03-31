#ifndef _RUNDRTAMASSTAGE_CXX
#define _RUNDRTAMASSTAGE_CXX

#include "run_drtamas_stage.h"
#include "DRTAMAS_utilities.h"
#include "compute_metric_dev.h"


#include "itkWindowConvergenceMonitoringFunction.h"
#include "itkRealTimeClock.h"



#include "../cuda_src/resample_image.h"
#include "../cuda_src/gaussian_smooth_image.h"
#include "../cuda_src/warp_image.h"
#include "../cuda_src/cuda_image_utilities.h"
#include "../cuda_src/compute_metric.h"



#include "../utilities/write_3D_image_to_4D_file.h"

void DRTAMASStage::CreateVirtualImage()
{
    if(this->settings->downsample_factor==1)
    {
        this->virtual_img= this->settings->metrics[0].fixed_img;
    }
    else
    {        
        this->virtual_img= CurrentImageType::New();

        this->virtual_img->dir= this->settings->metrics[0].fixed_img->dir;
        this->virtual_img->spc.x = this->settings->metrics[0].fixed_img->spc.x * this->settings->downsample_factor;
        this->virtual_img->spc.y = this->settings->metrics[0].fixed_img->spc.y * this->settings->downsample_factor;
        this->virtual_img->spc.z = this->settings->metrics[0].fixed_img->spc.z * this->settings->downsample_factor;

        this->virtual_img->sz.x = static_cast<float>(
                    std::floor( (double)this->settings->metrics[0].fixed_img->sz.x / (double)this->settings->downsample_factor ) );
        if(this->virtual_img->sz.x<1)
            this->virtual_img->sz.x=1;

        this->virtual_img->sz.y = static_cast<float>(
                    std::floor( (double)this->settings->metrics[0].fixed_img->sz.y / (double)this->settings->downsample_factor ) );
        if(this->virtual_img->sz.y<1)
            this->virtual_img->sz.y=1;

        this->virtual_img->sz.z = static_cast<float>(
                    std::floor( (double)this->settings->metrics[0].fixed_img->sz.z / (double)this->settings->downsample_factor ) );
        if(this->virtual_img->sz.z<1)
            this->virtual_img->sz.z=1;

        float3 ici, oci;
        ici.x= ((double)this->settings->metrics[0].fixed_img->sz.x -1) /2.;
        ici.y= ((double)this->settings->metrics[0].fixed_img->sz.y -1) /2.;
        ici.z= ((double)this->settings->metrics[0].fixed_img->sz.z -1) /2.;

        oci.x= ((double)this->virtual_img->sz.x -1) /2.;
        oci.y= ((double)this->virtual_img->sz.y -1) /2.;
        oci.z= ((double)this->virtual_img->sz.z -1) /2.;

        float3 s1i1, s2i2, diff;
        s2i2.x = this->virtual_img->spc.x * oci.x;
        s2i2.y = this->virtual_img->spc.y * oci.y;
        s2i2.z = this->virtual_img->spc.z * oci.z;

        s1i1.x = this->settings->metrics[0].fixed_img->spc.x * ici.x;
        s1i1.y = this->settings->metrics[0].fixed_img->spc.y * ici.y;
        s1i1.z = this->settings->metrics[0].fixed_img->spc.z * ici.z;

        diff.x= s1i1.x- s2i2.x;
        diff.y= s1i1.y- s2i2.y;
        diff.z= s1i1.z- s2i2.z;

        this->virtual_img->orig.x = this->virtual_img->dir(0,0)* diff.x + this->virtual_img->dir(0,1)* diff.y + this->virtual_img->dir(0,2)* diff.z +  (double)this->settings->metrics[0].fixed_img->orig.x;
        this->virtual_img->orig.y = this->virtual_img->dir(1,0)* diff.x + this->virtual_img->dir(1,1)* diff.y + this->virtual_img->dir(1,2)* diff.z +  (double)this->settings->metrics[0].fixed_img->orig.y;
        this->virtual_img->orig.z = this->virtual_img->dir(2,0)* diff.x + this->virtual_img->dir(2,1)* diff.y + this->virtual_img->dir(2,2)* diff.z +  (double)this->settings->metrics[0].fixed_img->orig.z;

    }
}




void DRTAMASStage::PreprocessImagesAndFields()
{
    CreateVirtualImage();

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


    if(this->def_FINV && this->def_FINV->sz.x != this->virtual_img->sz.x)
    {
        this->def_FINV= ResampleImage(this->def_FINV, this->virtual_img);
        this->def_MINV= ResampleImage(this->def_MINV, this->virtual_img);
    }
    if(this->def_F==nullptr)
    {
        this->def_F= InvertField(this->def_FINV);
        this->def_M= InvertField(this->def_MINV);
    }
}







void DRTAMASStage::RunDRTAMASStage()
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




    std::vector<CurrentImageType::Pointer>  log_fixed_tensor_vec;
    std::vector<CurrentImageType::Pointer>  log_moving_tensor_vec;
    std::vector<CurrentImageType::Pointer> resampled_smoothed_fixed_structurals;
    std::vector<CurrentImageType::Pointer> resampled_smoothed_moving_structurals;
    CurrentImageType::Pointer resampled_smoothed_fixed_TR=nullptr;
    CurrentImageType::Pointer resampled_smoothed_moving_TR=nullptr;
    for(int met=0; met< Nmetrics;met++)
    {
        if(this->settings->metrics[met].MetricType== DRTAMASMetricEnumeration::DTTR)
        {
            resampled_smoothed_fixed_TR=CUDAIMAGE::New();
            resampled_smoothed_fixed_TR->DuplicateFromCUDAImage(this->settings->metrics[met].fixed_img);
            resampled_smoothed_moving_TR=CUDAIMAGE::New();
            resampled_smoothed_moving_TR->DuplicateFromCUDAImage(this->settings->metrics[met].moving_img);

            if(this->settings->img_smoothing_std!=0)
            {
                resampled_smoothed_fixed_TR= GaussianSmoothImage(resampled_smoothed_fixed_TR,this->settings->img_smoothing_std*this->settings->img_smoothing_std);
                resampled_smoothed_moving_TR= GaussianSmoothImage(resampled_smoothed_moving_TR,this->settings->img_smoothing_std*this->settings->img_smoothing_std);
            }

            if(resampled_smoothed_fixed_TR->sz.x != this->virtual_img->sz.x)
            {
                resampled_smoothed_fixed_TR=ResampleImage(resampled_smoothed_fixed_TR,this->virtual_img);
                resampled_smoothed_moving_TR=ResampleImage(resampled_smoothed_moving_TR,this->virtual_img);
            }

            resampled_smoothed_fixed_TR->CreateTexture();
            resampled_smoothed_moving_TR->CreateTexture();
        }
        if(this->settings->metrics[met].MetricType== DRTAMASMetricEnumeration::DTCC)
        {
            CurrentImageType::Pointer curr_fixed=CUDAIMAGE::New();
            curr_fixed->DuplicateFromCUDAImage(this->settings->metrics[met].fixed_img);
            CurrentImageType::Pointer curr_moving=CUDAIMAGE::New();
            curr_moving->DuplicateFromCUDAImage(this->settings->metrics[met].moving_img);

            if(this->settings->img_smoothing_std!=0)
            {
                curr_fixed= GaussianSmoothImage(curr_fixed,this->settings->img_smoothing_std*this->settings->img_smoothing_std);
                curr_moving= GaussianSmoothImage(curr_moving,this->settings->img_smoothing_std*this->settings->img_smoothing_std);
            }
            if(curr_fixed->sz.x != this->virtual_img->sz.x)
            {
                curr_fixed=ResampleImage(curr_fixed,this->virtual_img);
                curr_moving=ResampleImage(curr_moving,this->virtual_img);
            }

            curr_fixed->CreateTexture();
            curr_moving->CreateTexture();

            resampled_smoothed_fixed_structurals.push_back(curr_fixed);
            resampled_smoothed_moving_structurals.push_back(curr_moving);

        }
        if(this->settings->metrics[met].MetricType== DRTAMASMetricEnumeration::DTDEV)
        {
           // CurrentImageType::Pointer log_fixed_tensor= LogTensor(this->settings->metrics[met].fixed_img);
           // CurrentImageType::Pointer log_moving_tensor= LogTensor(this->settings->metrics[met].moving_img);

            CurrentImageType::Pointer log_fixed_tensor= this->settings->metrics[met].fixed_img;
            CurrentImageType::Pointer log_moving_tensor= this->settings->metrics[met].moving_img;

            log_fixed_tensor_vec= SplitImageComponents(log_fixed_tensor);
            log_moving_tensor_vec= SplitImageComponents(log_moving_tensor);

            for(int v=0;v<log_fixed_tensor_vec.size();v++)
            {
                if(this->settings->img_smoothing_std!=0)
                {
                    log_fixed_tensor_vec[v]= GaussianSmoothImage(log_fixed_tensor_vec[v],this->settings->img_smoothing_std*this->settings->img_smoothing_std);
                    log_moving_tensor_vec[v]= GaussianSmoothImage(log_moving_tensor_vec[v],this->settings->img_smoothing_std*this->settings->img_smoothing_std);
                }

                if(log_fixed_tensor_vec[v]->sz.x != this->virtual_img->sz.x)
                {
                    log_fixed_tensor_vec[v]=ResampleImage(log_fixed_tensor_vec[v],this->virtual_img);
                    log_moving_tensor_vec[v]=ResampleImage(log_moving_tensor_vec[v],this->virtual_img);
                }

                log_fixed_tensor_vec[v]->CreateTexture();
                log_moving_tensor_vec[v]->CreateTexture();
            }
        }
    }

    int iter=0;
    bool converged=false;
    float curr_convergence =1;
    while( iter++ < this->settings->niter && !converged )
    {
        CurrentFieldType::Pointer updateFieldF= nullptr,updateFieldM= nullptr;

        if(Nmetrics>1)
        {
            updateFieldF= CurrentFieldType::New();
            updateFieldF->sz = this->def_FINV->sz;
            updateFieldF->dir = this->def_FINV->dir;
            updateFieldF->orig = this->def_FINV->orig;
            updateFieldF->spc = this->def_FINV->spc;
            updateFieldF->components_per_voxel = this->def_FINV->components_per_voxel;
            updateFieldF->Allocate();

            updateFieldM= CurrentFieldType::New();
            updateFieldM->sz = this->def_FINV->sz;
            updateFieldM->dir = this->def_FINV->dir;
            updateFieldM->orig = this->def_FINV->orig;
            updateFieldM->spc = this->def_FINV->spc;
            updateFieldM->components_per_voxel = this->def_FINV->components_per_voxel;
            updateFieldM->Allocate();
        }



        int str_id=0;
        std::vector<float> metric_values;
        metric_values.resize(Nmetrics);
        for(int met=0; met< Nmetrics;met++)
        {
            float metric_value=0;
            CurrentFieldType::Pointer  updateFieldF_temp=nullptr,updateFieldM_temp=nullptr;
            if(this->settings->metrics[met].MetricType== DRTAMASMetricEnumeration::DTCC)
            {

                CurrentImageType::Pointer warped_fixed_img = WarpImage(resampled_smoothed_fixed_structurals[str_id],this->def_FINV);
                CurrentImageType::Pointer warped_moving_img = WarpImage(resampled_smoothed_moving_structurals[str_id],this->def_MINV);
                str_id++;

                metric_value = ComputeMetric_CC(warped_fixed_img,warped_moving_img, updateFieldF_temp,updateFieldM_temp );

            }
            if(this->settings->metrics[met].MetricType== DRTAMASMetricEnumeration::DTTR)
            {
                CurrentImageType::Pointer warped_fixed_img = WarpImage(resampled_smoothed_fixed_TR,this->def_FINV);
                CurrentImageType::Pointer warped_moving_img = WarpImage(resampled_smoothed_moving_TR,this->def_MINV);

                metric_value = ComputeMetric_CC(warped_fixed_img,warped_moving_img, updateFieldF_temp,updateFieldM_temp );
            }
            if(this->settings->metrics[met].MetricType== DRTAMASMetricEnumeration::DTDEV)
            {

                std::vector<CurrentImageType::Pointer> warped_fixed_tensorv, warped_moving_tensorv;
                for(int v=0;v<log_fixed_tensor_vec.size();v++)
                {
                    CurrentImageType::Pointer ft= WarpImage(log_fixed_tensor_vec[v],this->def_FINV);
                    CurrentImageType::Pointer mt= WarpImage(log_moving_tensor_vec[v],this->def_MINV);
                    warped_fixed_tensorv.push_back(ft);
                    warped_moving_tensorv.push_back(mt);
                }

                CurrentImageType::Pointer warped_fixed_tensor = CombineImageComponents(warped_fixed_tensorv);
                CurrentImageType::Pointer warped_moving_tensor = CombineImageComponents(warped_moving_tensorv);

              //  warped_fixed_tensor = ExpTensor(warped_fixed_tensor) ;
              //  warped_moving_tensor = ExpTensor(warped_moving_tensor) ;



                metric_value = ComputeMetric_DEV(warped_fixed_tensor, warped_moving_tensor,
                                               this->def_FINV , this->def_MINV ,
                                               updateFieldF_temp,  updateFieldM_temp,
                                               this->settings->metrics[met].to);

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


        ScaleUpdateField(updateFieldF,this->settings->learning_rate);
        ScaleUpdateField(updateFieldM,this->settings->learning_rate);


        this->def_F  = ComposeFields(this->def_F,updateFieldF);
        this->def_M  = ComposeFields(this->def_M,updateFieldM);
        this->def_F = GaussianSmoothImage(this->def_F,this->settings->total_gaussian_sigma);
        this->def_M = GaussianSmoothImage(this->def_M,this->settings->total_gaussian_sigma);


        this->def_FINV  = InvertField(this->def_F);
        this->def_MINV  = InvertField(this->def_M);
        this->def_F  = InvertField(this->def_FINV);
        this->def_M  = InvertField(this->def_MINV);




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

        if( (0.7*curr_convergence+0.3*prev_conv) < 1E-7)
        {
            if(this->settings->downsample_factor<=2)
                converged = true;
        }


        m_clock.Stop();
        const itk::RealTimeClock::TimeStampType now = m_clock.GetTotal();

        std::cout << " "; // if the output of current iteration is written to disk, and star
                // will appear before line, else a free space will be printed to keep visual alignment.

        std::cout << std::setw(5) << iter << ", conv: "<<std::scientific << std::setprecision(5) << curr_convergence <<", time: "
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
