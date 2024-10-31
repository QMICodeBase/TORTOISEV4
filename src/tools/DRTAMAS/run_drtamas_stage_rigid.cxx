#ifndef _RUNDRTAMASStageRigid_CXX
#define _RUNDRTAMASStageRigid_CXX

#include "run_drtamas_stage_rigid.h"
#include "DRTAMAS_utilities.h"
#include "compute_metric_dev.h"
#include "../cuda_src/compute_metric.h"


#include "itkWindowConvergenceMonitoringFunction.h"
#include "itkRealTimeClock.h"




#include "../cuda_src/resample_image.h"
#include "../cuda_src/gaussian_smooth_image.h"
#include "../cuda_src/cuda_image_utilities.h"
#include "../cuda_src/rigid_transform_image.h"


#include "../utilities/write_3D_image_to_4D_file.h"


double NF=1;


void DRTAMASStageRigid::CreateVirtualImage()
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




void DRTAMASStageRigid::PreprocessImagesAndFields()
{
    CreateVirtualImage();

}



double DRTAMASStageRigid::ComputeMetricRigid_TR(CUDAIMAGE::Pointer fixed_tensor, CUDAIMAGE::Pointer  moving_tensor)
{
    CUDAIMAGE::Pointer fixed_TR = ComputeTRMapC(fixed_tensor);
    CUDAIMAGE::Pointer moving_TR = ComputeTRMapC(moving_tensor);


    CUDAIMAGE::Pointer  updateFieldF_temp=nullptr,updateFieldM_temp=nullptr;

    double metric_value = ComputeMetric_CC(fixed_TR,moving_TR, updateFieldF_temp,updateFieldM_temp );
    return metric_value;
}

double DRTAMASStageRigid::ComputeMetricRigid_DEV(CUDAIMAGE::Pointer fixed_tensor, CUDAIMAGE::Pointer  moving_tensor)
{
    double metric_value = ComputeMetric_DEV_ONLY(fixed_tensor,moving_tensor );
    //if(NF==1)
        return metric_value;

  //  return -NF/ metric_value / 9. - 8./9.;
}



std::vector<double> DRTAMASStageRigid::ComputeMetrics(CUDAIMAGE::Pointer fixed_img, std::vector<CurrentImageType::Pointer> log_moving_tensor_vec, RigidTransformType::Pointer rigid_trans)
{
    std::vector<CurrentImageType::Pointer>  warped_moving_tensorv;
    for(int v=0;v<log_moving_tensor_vec.size();v++)
    {
        CurrentImageType::Pointer mt= RigidTransformImageC(log_moving_tensor_vec[v],rigid_trans,fixed_img);
        warped_moving_tensorv.push_back(mt);
    }

    CurrentImageType::Pointer warped_moving_tensor = CombineImageComponents(warped_moving_tensorv);

    warped_moving_tensor = ExpTensor(warped_moving_tensor) ;
    warped_moving_tensor = RotateTensors(warped_moving_tensor,rigid_trans) ;


    /*
    using TensorVectorType = itk::Vector<float,6>;
    using TensorVectorImageType = itk::Image<TensorVectorType,3>;

    TensorVectorImageType::Pointer ff= fixed_img->CudaImageToITKImage4D();
    TensorVectorImageType::Pointer mm=warped_moving_tensor->CudaImageToITKImage4D();

    using WrType= itk::ImageFileWriter<TensorVectorImageType>;
    {
        WrType::Pointer wr= WrType::New();
        wr->SetFileName("ff.nii");
        wr->SetInput(ff);
        wr->Update();
    }
    {
        WrType::Pointer wr= WrType::New();
        wr->SetFileName("mm.nii");
        wr->SetInput(mm);
        wr->Update();
    }
*/

    std::vector<double> metric_values;

    for(int met=0; met< 2;met++)
    {
        if(this->settings->metrics[met].MetricType== DRTAMASMetricEnumeration::DTDEV)
        {
            double mm=ComputeMetricRigid_DEV(fixed_img, warped_moving_tensor);
            metric_values.push_back(mm);
            mm=ComputeMetricRigid_TR(fixed_img, warped_moving_tensor);
            metric_values.push_back(mm);
        }
    }
    return metric_values;
}


std::vector< std::pair<double,double> > DRTAMASStageRigid::BracketGrad(CUDAIMAGE::Pointer fixed_img, std::vector<CUDAIMAGE::Pointer> moving_img_vec,
                                                                                  RigidTransformType::ParametersType params, vnl_vector<double> grad, int mode)
{
    RigidTransformType::ParametersType or_params=params;

    int MAX_IT=20;
    double m_ERR_MARG=0.00001;

    double f_ini= this->class_metric_value;
    double x_ini=0;

    double f_min = f_ini;
    double x_min = x_ini;

    double f_last;
    double x_last;

    bool bail = 0;
    double counter = 1;
    int iter=0;

    double BracketParams[2];
    BracketParams[1]=0.1;
    BracketParams[0]=0.0001;

    double brk_const = BracketParams[mode];

    while(!bail)
    {
        //double konst = counter*counter*brk_const;
        double konst = counter*brk_const;
        RigidTransformType::ParametersType  temp_params=or_params;

        for(int t=0+3*mode;t<3+3*mode;t++)
            temp_params[t] -= konst*grad[t];

        RigidTransformType::Pointer ctrans= RigidTransformType::New();
        ctrans->SetParameters(temp_params);

        std::vector<double> metrics = ComputeMetrics(fixed_img,moving_img_vec,ctrans);
        //double fm =  this->settings->metrics[0].weight * metrics[0] + this->settings->metrics[1].weight * metrics[1];
        double fm=metrics[1];

        f_last= fm;
        x_last = konst;

        if(f_last <f_min)
        {
            f_min = f_last;
            x_min = x_last;
        }
        else
        {
            if(  (f_last > f_min+m_ERR_MARG)  || (iter >MAX_IT))
                bail=true;
        }
        //counter++;
        counter*=1.7;
        iter++;
    }


    std::vector< std::pair<double,double> > x_f_pairs;
    x_f_pairs.resize(3);
    x_f_pairs[0]=std::make_pair(x_ini,f_ini);
    x_f_pairs[1]=std::make_pair(x_min,f_min);
    x_f_pairs[2]=std::make_pair(x_last,f_last);

    return x_f_pairs;
}


double DRTAMASStageRigid::GoldenSearch(CUDAIMAGE::Pointer fixed_img, std::vector<CUDAIMAGE::Pointer> moving_img_vec,RigidTransformType::ParametersType or_params, std::vector< std::pair<double,double> > &x_f_pairs, vnl_vector<double> grad, int mode)
{

    double BracketParams[2];
    BracketParams[1]=0.1;
    BracketParams[0]=0.0001;
    double cst = BracketParams[mode];

    int MAX_IT=50;
    int counter=0;
    double TOL=cst;
    double R=  0.61803399;
    double C = 1.0 - R;

    double ax= x_f_pairs[0].first;
    double bx= x_f_pairs[1].first;
    double cx= x_f_pairs[2].first;

    double x0=ax;
    double x3=cx;
    double x1,x2;

    if (fabs(cx-bx) > fabs(bx-ax))
    {
        x1=bx;
        x2= bx+C*(cx-bx);

    }
    else
    {
        x2=bx;
        x1=bx- C*(bx-ax);
    }



    double fx1=0,fx2=0;

    {
        RigidTransformType::ParametersType  temp_params=or_params;

        for(int t=0+3*mode;t<3+3*mode;t++)
            temp_params[t] -= x1*grad[t];
        RigidTransformType::Pointer ctrans= RigidTransformType::New();
        ctrans->SetParameters(temp_params);
        std::vector<double> metrics = ComputeMetrics(fixed_img,moving_img_vec,ctrans);
        //fx1=this->settings->metrics[0].weight * metrics[0] + this->settings->metrics[1].weight * metrics[1];
        fx1=metrics[1];
    }
    {
        RigidTransformType::ParametersType  temp_params=or_params;

        for(int t=0+3*mode;t<3+3*mode;t++)
            temp_params[t] -= x2*grad[t];
        RigidTransformType::Pointer ctrans= RigidTransformType::New();
        ctrans->SetParameters(temp_params);
        std::vector<double> metrics = ComputeMetrics(fixed_img,moving_img_vec,ctrans);
        //fx2=this->settings->metrics[0].weight * metrics[0] + this->settings->metrics[1].weight * metrics[1];
        fx2=metrics[1];
    }

    while( (fabs(x3-x0) > TOL) && (counter <MAX_IT))
    {
        if(fx2 <fx1)
        {
            x0=x1;
            x1=x2;
            x2= R*x2+C*x3;

            RigidTransformType::ParametersType  temp_params=or_params;

            for(int t=0+3*mode;t<3+3*mode;t++)
                temp_params[t] -= x2*grad[t];
            RigidTransformType::Pointer ctrans= RigidTransformType::New();
            ctrans->SetParameters(temp_params);
            std::vector<double> metrics = ComputeMetrics(fixed_img,moving_img_vec,ctrans);
            //double xt=this->settings->metrics[0].weight * metrics[0] + this->settings->metrics[1].weight * metrics[1];
            double xt=metrics[1];

            fx1=fx2;
            fx2=xt;
        }
        else
        {
            x3=x2;
            x2=x1;
            x1=R*x1+C*x0;
            RigidTransformType::ParametersType  temp_params=or_params;

            for(int t=0+3*mode;t<3+3*mode;t++)
                temp_params[t] -= x2*grad[t];
            RigidTransformType::Pointer ctrans= RigidTransformType::New();
            ctrans->SetParameters(temp_params);
            std::vector<double> metrics = ComputeMetrics(fixed_img,moving_img_vec,ctrans);
            //double xt=this->settings->metrics[0].weight * metrics[0] + this->settings->metrics[1].weight * metrics[1];
            double xt=metrics[1];

            fx2=fx1;
            fx1=xt;
        }
        counter++;
    }


    if(fx1<fx2)
    {
        return x1;
    }
    else
    {
        return x2;
    }

}


void DRTAMASStageRigid::RunDRTAMASStageRigid()
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



    std::vector<CurrentImageType::Pointer>  log_moving_tensor_vec;

    CurrentImageType::Pointer resampled_fixed_tensor=nullptr;


    for(int met=0; met< Nmetrics;met++)
    {
        if(this->settings->metrics[met].MetricType== DRTAMASMetricEnumeration::DTDEV)
        {                        
            CurrentImageType::Pointer log_fixed_tensor= LogTensor(this->settings->metrics[met].fixed_img);
            CurrentImageType::Pointer log_moving_tensor= LogTensor(this->settings->metrics[met].moving_img);

            std::vector<CurrentImageType::Pointer> log_fixed_tensor_vec= SplitImageComponents(log_fixed_tensor);
            log_moving_tensor_vec= SplitImageComponents(log_moving_tensor);

            for(int v=0;v<log_moving_tensor_vec.size();v++)
            {
                if(this->settings->img_smoothing_std!=0)
                {
                    log_fixed_tensor_vec[v]= GaussianSmoothImage(log_fixed_tensor_vec[v],this->settings->img_smoothing_std*this->settings->img_smoothing_std);
                    log_moving_tensor_vec[v]= GaussianSmoothImage(log_moving_tensor_vec[v],this->settings->img_smoothing_std*this->settings->img_smoothing_std);
                }

                if(log_moving_tensor_vec[v]->sz.x != this->virtual_img->sz.x)
                {
                    log_fixed_tensor_vec[v]=ResampleImage(log_fixed_tensor_vec[v],this->virtual_img);
                    log_moving_tensor_vec[v]=ResampleImage(log_moving_tensor_vec[v],this->virtual_img);
                }

                log_moving_tensor_vec[v]->CreateTexture();
            }

            resampled_fixed_tensor = CombineImageComponents(log_fixed_tensor_vec);
            resampled_fixed_tensor = ExpTensor(resampled_fixed_tensor) ;
        }
    }

    vnl_vector<double> grad_scales(6,0);
    grad_scales[0]=0.05;
    grad_scales[1]=0.05;
    grad_scales[2]=0.05;
    grad_scales[3]= virtual_img->spc.x*1.25;
    grad_scales[4]= virtual_img->spc.y*1.25;
    grad_scales[5]= virtual_img->spc.z*1.25;


    int halves=0;

    int iter=0;
    bool converged=false;
    float curr_convergence =1;
    while( iter++ < this->settings->niter && !converged )
    {                     
        std::vector<float> metric_values;
        metric_values.resize(Nmetrics);

        vnl_vector<double> grad1(6,0), grad2(6,0),grad(6,0);
        auto orig_params = settings->rigid_trans->GetParameters();


        std::vector<double> metrics= ComputeMetrics(resampled_fixed_tensor, log_moving_tensor_vec,settings->rigid_trans);
        //this->class_metric_value =  this->settings->metrics[0].weight * metrics[0] + this->settings->metrics[1].weight * metrics[1];
        this->class_metric_value=metrics[1];

        metric_values[0]=metrics[0];
        metric_values[1]=metrics[1];
        all_ConvergenceMonitoring[0]->AddEnergyValue( metrics[0] );
        all_ConvergenceMonitoring[1]->AddEnergyValue( metrics[1] );

        for(int t=0;t<6;t++)
        {
            float metric_values1[2]={0};
            float metric_values2[2]={0};

            RigidTransformType::Pointer iter_trans=  RigidTransformType::New();
            iter_trans->SetParameters(orig_params);

            for(int aa=0;aa<2;aa++)
            {
                int sgn= -2*aa +1;

                auto new_params=orig_params;
                new_params[t]+=sgn*grad_scales[t];
                iter_trans->SetParameters(new_params);

                std::vector<double> metrics = ComputeMetrics(resampled_fixed_tensor, log_moving_tensor_vec,iter_trans);

                metric_values1[aa]=metrics[0];
                metric_values2[aa]=metrics[1];
            }
            grad1[t]= (metric_values1[0] - metric_values1[1]) /2./grad_scales[t];
            grad2[t]= (metric_values2[0] - metric_values2[1]) /2./grad_scales[t];
        }


        double nrm11 = sqrt(grad1[0]*grad1[0] + grad1[1]*grad1[1] + grad1[2]*grad1[2] );
        double nrm12 = sqrt(grad1[3]*grad1[3] + grad1[4]*grad1[4] + grad1[5]*grad1[5] );

        double nrm21 = sqrt(grad2[0]*grad2[0] + grad2[1]*grad2[1] + grad2[2]*grad2[2] );
        double nrm22 = sqrt(grad2[3]*grad2[3] + grad2[4]*grad2[4] + grad2[5]*grad2[5] );

        if(nrm11>0)
            grad1[0]/=nrm11; grad1[1]/=nrm11; grad1[2]/=nrm11;
        if(nrm12>0)
            grad1[3]/=nrm12; grad1[4]/=nrm12; grad1[5]/=nrm12;
        if(nrm21>0)
            grad2[0]/=nrm21; grad2[1]/=nrm21; grad2[2]/=nrm21;
        if(nrm22>0)
            grad2[3]/=nrm22; grad2[4]/=nrm22; grad2[5]/=nrm22;

        for(int t=0;t<6;t++)
            grad[t]+=  this->settings->metrics[0].weight * grad1[t] + this->settings->metrics[1].weight * grad2[t];


        double nrm1 = sqrt(grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2] );
        double nrm2 = sqrt(grad[3]*grad[3] + grad[4]*grad[4] + grad[5]*grad[5] );
        if(nrm1>0)
            grad[0]/=nrm1; grad[1]/=nrm1; grad[2]/=nrm1;
        if(nrm2>0)
            grad[3]/=nrm2; grad[4]/=nrm2; grad[5]/=nrm2;


        auto new_params=orig_params;

/*
        new_params[0]+= -   grad_scales[0]*grad[0]*0.05;
        new_params[1]+= -   grad_scales[1]*grad[1]*0.05;
        new_params[2]+= -   grad_scales[2]*grad[2] *0.05;
        new_params[3]+= - grad_scales[3]*grad[3]*0.05 ;
        new_params[4]+= - grad_scales[4]*grad[4]*0.05;
        new_params[5]+= - grad_scales[5]*grad[5]*0.05;

        settings->rigid_trans->SetParameters(new_params);
*/



        if(nrm1>0)
        {
            std::vector< std::pair<double,double> > x_f_pairs= BracketGrad(resampled_fixed_tensor, log_moving_tensor_vec,orig_params, grad,0);

            if(std::abs(x_f_pairs[0].first-x_f_pairs[1].first)>0)
            {
                double step_length= GoldenSearch(resampled_fixed_tensor, log_moving_tensor_vec,orig_params,x_f_pairs,grad,0);
                auto new_params=orig_params;

                new_params[0]+= - step_length*grad[0];
                new_params[1]+= - step_length*grad[1];
                new_params[2]+= - step_length*grad[2];

                settings->rigid_trans->SetParameters(new_params);
            }
            else
            {
                auto new_change= orig_params;
                auto new_params=orig_params;
                for(int p=0;p<3;p++)
                    new_change[p]= grad[p] * grad_scales[p]*0.05;

                for(int iter=0;iter<3;iter++)
                {
                    for(int p=0;p<3;p++)
                        new_params[p]= orig_params[p] - new_change[p];

                    RigidTransformType::Pointer iter_trans=  RigidTransformType::New();
                    iter_trans->SetParameters(new_params);

                    std::vector<double> metrics = ComputeMetrics(resampled_fixed_tensor, log_moving_tensor_vec,iter_trans);

                    double fm=metrics[1];
                    //double fm =  this->settings->metrics[0].weight * metrics[0] + this->settings->metrics[1].weight * metrics[1];
                    if(fm < this->class_metric_value)
                    {
                        settings->rigid_trans->SetParameters(new_params);
                        break;
                    }
                    else
                        new_change= new_change/2.;
                }

            } // if bracket
        }  //if translation

        if(nrm2>0)
        {
            std::vector< std::pair<double,double> > x_f_pairs= BracketGrad(resampled_fixed_tensor, log_moving_tensor_vec,orig_params,grad,1);

            if(std::abs(x_f_pairs[0].first-x_f_pairs[1].first)>0)
            {
                double new_cost;
                double step_length= GoldenSearch(resampled_fixed_tensor, log_moving_tensor_vec,orig_params,x_f_pairs,grad,1);
                auto new_params=orig_params;

                new_params[3]+= - step_length*grad[3];
                new_params[4]+= - step_length*grad[4];
                new_params[5]+= - step_length*grad[5];

                settings->rigid_trans->SetParameters(new_params);
            }
            else
            {
                auto new_change= orig_params;
                auto new_params=orig_params;
                for(int p=3;p<6;p++)
                    new_change[p]= grad[p] * grad_scales[p]*0.05;

                for(int iter=0;iter<3;iter++)
                {
                    for(int p=3;p<6;p++)
                        new_params[p]= orig_params[p] - new_change[p];

                    RigidTransformType::Pointer iter_trans=  RigidTransformType::New();
                    iter_trans->SetParameters(new_params);

                    std::vector<double> metrics = ComputeMetrics(resampled_fixed_tensor, log_moving_tensor_vec,iter_trans);

                    double fm=metrics[1];

                    //double fm =  this->settings->metrics[0].weight * metrics[0] + this->settings->metrics[1].weight * metrics[1];
                    if(fm < this->class_metric_value)
                    {
                        settings->rigid_trans->SetParameters(new_params);
                        break;
                    }
                    else
                        new_change= new_change/2.;
                }

            } // if bracket
        }  //if rotation



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
            if(halves<5)
            {
                iter=0;
                grad_scales/=1.7;
                all_ConvergenceMonitoring[0]->ClearEnergyValues();
                all_ConvergenceMonitoring[1]->ClearEnergyValues();

                halves++;
            }
            else
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



}



#endif
