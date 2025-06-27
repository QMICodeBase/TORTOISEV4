#ifndef _RUNDRBUDDIStage_TVVF_CXX
#define _RUNDRBUDDIStage_TVVF_CXX

#include "run_drbuddi_stage_TVVF.h"

#include "itkWindowConvergenceMonitoringFunction.h"
#include "itkRealTimeClock.h"
#include "../tools/ResampleDWIs/resample_dwis.h"
#include "itkGaussianOperator.h"


#include "../cuda_src/resample_image.h"
#include "../cuda_src/gaussian_smooth_image.h"
#include "../cuda_src/warp_image.h"
#include "../cuda_src/cuda_image_utilities.h"
#include "../cuda_src/compute_metric.h"

//#include "itkTimeVaryingVelocityFieldIntegrationImageFilter.h"


void DRBUDDIStage_TVVF::PreprocessImagesAndFields()
{
    CreateVirtualImage();

    if(this->velocity_field.size()==0)
    {
        this->velocity_field.resize(NTimePoints);

        for(int T=0;T<NTimePoints;T++)
        {
            CUDAIMAGE::Pointer field= CUDAIMAGE::New();
            field->orig=  this->virtual_img->orig;
            field->dir=  this->virtual_img->dir;
            field->spc=  this->virtual_img->spc;
            field->sz=  this->virtual_img->sz;
            field->components_per_voxel= 3;
            field->Allocate();

            this->velocity_field[T]=field;
        }
    }

    if(this->velocity_field.size()>0 && this->velocity_field[0]->sz.x !=this->virtual_img->sz.x)
    {
        for(int T=0;T<NTimePoints;T++)
        {
            this->velocity_field[T]=ResampleImage(this->velocity_field[T], this->virtual_img);
        }
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
        }
    }


    if(resampled_smoothed_up_images[0]->sz.x != this->virtual_img->sz.x)    
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







void DRBUDDIStage_TVVF::RunDRBUDDIStage()
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



    itk::GaussianOperator<float,3> Goper;
    Goper.SetDirection(phase_id);
    Goper.SetVariance(this->settings->update_gaussian_sigma);
    Goper.SetMaximumError(0.001);
    Goper.SetMaximumKernelWidth(31);
    Goper.CreateDirectional();

    CurrentFieldType::Pointer update_velocity_field;

    for(int m=0;m<this->settings->metrics.size();m++)
    {
        if(resampled_smoothed_up_images[m]->getFloatdata().ptr !=nullptr)
            resampled_smoothed_up_images[m]->CreateTexture();

        if(resampled_smoothed_down_images[m]->getFloatdata().ptr !=nullptr)
            resampled_smoothed_down_images[m]->CreateTexture();
    }


    int iter=0;
    bool converged=false;
    float curr_convergence =1;
    while( iter++ < this->settings->niter *2 && !converged )
    {
        CurrentFieldType::Pointer updateFieldF= nullptr,updateFieldM= nullptr;

        std::vector<float> metric_values;
        metric_values.resize(Nmetrics);
        for(int met=0; met< Nmetrics;met++)
            metric_values[met]=0;


        for(int T=0;T<NTimePoints;T++)
        {
            float t= 1.*T/(NTimePoints-1);

            if(Nmetrics>1)
            {
                updateFieldF=CurrentFieldType::New();
                updateFieldF->sz = this->velocity_field[0]->sz;
                updateFieldF->dir = this->velocity_field[0]->dir;
                updateFieldF->orig = this->velocity_field[0]->orig;
                updateFieldF->spc = this->velocity_field[0]->spc;
                updateFieldF->components_per_voxel = 3;
                updateFieldF->Allocate();

                updateFieldM=CurrentFieldType::New();
                updateFieldM->sz = this->velocity_field[0]->sz;
                updateFieldM->dir = this->velocity_field[0]->dir;
                updateFieldM->orig = this->velocity_field[0]->orig;
                updateFieldM->spc = this->velocity_field[0]->spc;
                updateFieldM->components_per_voxel = 3;
                updateFieldM->Allocate();
            }


            CurrentFieldType::Pointer field_up=IntegrateVelocityField(t,0);
            CurrentFieldType::Pointer field_down=IntegrateVelocityField(t,1);
            CurrentFieldType::Pointer field_str=IntegrateVelocityField(t,0.5);

            if(settings->init_finv_const!=nullptr)
            {
                field_up= ComposeFields(settings->init_finv_const,field_up);
                field_down= ComposeFields(settings->init_minv_const,field_down);
            }

            if(this->settings->constrain )
            {
                if(t!=0 && t!=1)
                {
                //    field_up=MultiplyImage(AddImages(field_up,MultiplyImage(field_down,-t/(1-t))),0.5);
                //    field_down= MultiplyImage(field_up,(t-1)/t);


                }
                if(t==0.5)
                {
                    field_up= MultiplyImage(AddImages(field_up,MultiplyImage(field_down,-1)),0.5);
                    field_down= MultiplyImage(field_up,-1);
                }
            }

            for(int met=0; met< Nmetrics;met++)
            {
                CurrentImageType::Pointer warped_up_img = WarpImage(this->resampled_smoothed_up_images[met],field_up);
                CurrentImageType::Pointer  warped_down_img = WarpImage(this->resampled_smoothed_down_images[met],field_down);
                CurrentImageType::Pointer warped_str_img=nullptr, warped_str_img_jac=nullptr;


                if( this->resampled_smoothed_str_images[met])
                {
                    CUDAIMAGE::Pointer str_img_texture= CUDAIMAGE::New();
                    str_img_texture->DuplicateFromCUDAImage(this->resampled_smoothed_str_images[met]);
                    str_img_texture->CreateTexture();
                    warped_str_img = WarpImage(str_img_texture,field_str);
                  // warped_str_img_jac= ComputeDetImgMain(warped_str_img,field_str,this->up_phase_vector);
                    //warped_str_img_jac= warped_str_img;

                }

                float metric_value=0;
                CurrentFieldType::Pointer  updateFieldF_temp=nullptr,updateFieldM_temp=nullptr;

                if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::MSJac)
                {
                    this->settings->metrics[met].weight=1.;
                    metric_value = ComputeMetric_MSJac(warped_up_img,warped_down_img,
                                                       field_up , field_down ,
                                                       updateFieldF_temp,  updateFieldM_temp,
                                                       this->up_phase_vector,
                                                       Goper );
                }
                if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCSK)
                {
                    metric_value = ComputeMetric_CCSK(warped_up_img,warped_down_img, warped_str_img,updateFieldF_temp,updateFieldM_temp,0.5 );
                }
                if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CC)
                {
                   metric_value = ComputeMetric_CC(warped_up_img,warped_down_img, updateFieldF_temp,updateFieldM_temp );
                }
                if(this->settings->metrics[met].MetricType== DRBUDDIMetricEnumeration::CCJacS)
                {
                    metric_value = ComputeMetric_CCJacS(warped_up_img,warped_down_img,warped_str_img,
                                                              field_up, field_down,
                                                              updateFieldF_temp,  updateFieldM_temp,
                                                              this->up_phase_vector,
                                                              Goper );
                }

                if(t==0.5)
                    metric_values[met]=metric_value;

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

            } //for met


            //TVVF needs more smoothing
         //   updateFieldF=GaussianSmoothImage(updateFieldF,this->settings->update_gaussian_sigma);
           // updateFieldM=GaussianSmoothImage(updateFieldM,this->settings->update_gaussian_sigma);


            if(this->settings->restrct)
            {
                RestrictPhase(updateFieldF,this->up_phase_vector);
                RestrictPhase(updateFieldM,this->up_phase_vector);
            }

            // max disp vector is of norm 1
            ScaleUpdateField(updateFieldF,1);                             
            ScaleUpdateField(updateFieldM,1);

            //make the update velocity field diffeomorphic
            CurrentFieldType::Pointer updateFieldF_inv= InvertField(updateFieldF );
            updateFieldF=InvertField(updateFieldF_inv );
            CurrentFieldType::Pointer updateFieldM_inv= InvertField(updateFieldM );
            updateFieldM=InvertField(updateFieldM_inv );



            auto aa=MultiplyImage(AddImages(updateFieldF_inv,MultiplyImage(updateFieldM_inv,-1)),-1);
            ScaleUpdateField(aa,1);

            update_velocity_field = GaussianSmoothImage(aa,this->settings->update_gaussian_sigma);
            update_velocity_field= MultiplyImage(update_velocity_field,this->settings->learning_rate*2);
            this->velocity_field[T]= AddImages(this->velocity_field[T],update_velocity_field);



        } //T loop



        for(int met=0; met< Nmetrics;met++)
           all_ConvergenceMonitoring[met]->AddEnergyValue( metric_values[met]);



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

         (*stream)<<"Metric " << metric_values.size()-1<< ": " << std::setprecision(8)<<metric_values[metric_values.size()-1]<< " LR: " << this->settings->learning_rate<<std::endl;
         m_lastTotalTime = now;
         m_clock.Start();


    } //while iter    



   // this->settings->output_finv= this->def_FINV;
   // this->settings->output_minv= this->def_MINV;
}


#include "itkTimeVaryingVelocityFieldIntegrationImageFilter.h"

DRBUDDIStage_TVVF::CurrentFieldType::Pointer DRBUDDIStage_TVVF::IntegrateVelocityField(float lowt, float hight)
{
    CurrentFieldType::Pointer dfield=CurrentFieldType::New();
    dfield->sz = this->velocity_field[0]->sz;
    dfield->dir = this->velocity_field[0]->dir;
    dfield->orig = this->velocity_field[0]->orig;
    dfield->spc = this->velocity_field[0]->spc;
    dfield->components_per_voxel = 3;
    dfield->Allocate();

    if(lowt!=hight)
    {
        IntegrateVelocityFieldGPU(this->velocity_field, lowt,hight,dfield);

/*
        using VelocityFieldType =itk::Image<DisplacementFieldType::PixelType,4>;
        DisplacementFieldType::Pointer aaa= this->GetVelocityfield()[0]->CudaImageToITKField();

        VelocityFieldType::SizeType sz;
        sz[0]=aaa->GetLargestPossibleRegion().GetSize()[0];
        sz[1]=aaa->GetLargestPossibleRegion().GetSize()[1];
        sz[2]=aaa->GetLargestPossibleRegion().GetSize()[2];
        sz[3]=this->GetVelocityfield().size();
        VelocityFieldType::IndexType start; start.Fill(0);
        VelocityFieldType::RegionType reg(start,sz);

        VelocityFieldType::PointType orig;
        orig[0]= aaa->GetOrigin()[0];
        orig[1]= aaa->GetOrigin()[1];
        orig[2]= aaa->GetOrigin()[2];
        orig[3]=0;
        VelocityFieldType::SpacingType spc;
        spc[0]=aaa->GetSpacing()[0];
        spc[1]=aaa->GetSpacing()[1];
        spc[2]=aaa->GetSpacing()[2];
        spc[3]=1;
        VelocityFieldType::DirectionType dir; dir.SetIdentity();
        dir(0,0)=aaa->GetDirection()(0,0);dir(0,1)=aaa->GetDirection()(0,1);dir(0,2)=aaa->GetDirection()(0,2);
        dir(1,0)=aaa->GetDirection()(1,0);dir(1,1)=aaa->GetDirection()(1,1);dir(1,2)=aaa->GetDirection()(1,2);
        dir(2,0)=aaa->GetDirection()(2,0);dir(2,1)=aaa->GetDirection()(2,1);dir(2,2)=aaa->GetDirection()(2,2);

        VelocityFieldType::Pointer vim = VelocityFieldType::New();
        vim->SetRegions(reg);
        vim->Allocate();
        vim->SetSpacing(spc);
        vim->SetOrigin(orig);
        vim->SetDirection(dir);

        for(int T=0;T<this->GetVelocityfield().size();T++)
        {
            ImageType3D::IndexType ind3;
            VelocityFieldType::IndexType ind4;
            ind4[3]=T;

            DisplacementFieldType::Pointer curr_field = this->GetVelocityfield()[T]->CudaImageToITKField();

            itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(curr_field,curr_field->GetLargestPossibleRegion());
            for(it.GoToBegin();!it.IsAtEnd();++it)
            {
                ind3= it.GetIndex();
                ind4[0]=ind3[0];
                ind4[1]=ind3[1];
                ind4[2]=ind3[2];

                vim->SetPixel(ind4,it.Get());
            }
        }


        typedef itk::TimeVaryingVelocityFieldIntegrationImageFilter           <VelocityFieldType, DisplacementFieldType> IntegratorType;
          IntegratorType::Pointer integrator = IntegratorType::New();
          integrator->SetInput( vim);
          integrator->SetLowerTimeBound( lowt );
          integrator->SetUpperTimeBound( hight );
          integrator->SetNumberOfIntegrationSteps( this->GetVelocityfield().size()+2 );
          integrator->Update();

          DisplacementFieldType::Pointer field= integrator->GetOutput();
          CUDAIMAGE::Pointer dfield2=CUDAIMAGE::New();
          dfield2->SetImageFromITK(field);
          dfield=dfield2;

*/
    }

    return dfield;
}


#endif
