#ifndef _DRTAMASRIGIDBULK_CXX
#define _DRTAMASRIGIDBULK_CXX




#include "DRTAMASRigid_Bulk.h"
#include "DRTAMAS_utilities.h"

#include "run_drtamas_stage_rigid.h"

#include "itkCenteredTransformInitializer.h"


void DRTAMASRigid_Bulk::SetUpStages()
{
    int Nstg= parser->getNumberOfStages();
    int Nstr=parser->getNumberOfStructurals();

    if(Nstg==0)
    {
        SetDefaultStages();
    }
    else
    {
        this->stages.resize(Nstg);
        for(int st=0;st<Nstg;st++)
        {
            this->stages[st].niter=parser->GetNIter(st);
            this->stages[st].img_smoothing_std=parser->GetS(st);
            this->stages[st].downsample_factor=parser->GetF(st);
            this->stages[st].learning_rate=parser->GetLR(st);
            this->stages[st].update_gaussian_sigma=parser->GetUStd(st);
            this->stages[st].total_gaussian_sigma=parser->GetTStd(st);            

            int Nmetrics= parser->GetNMetrics(st);
            if(Nmetrics==0)
            {
                DRTAMASMetric metric1;
                metric1.SetMetricType( DRTAMASMetricEnumeration::DTTR);
                this->stages[st].metrics.push_back(metric1);
                DRTAMASMetric metric2;
                metric2.SetMetricType( DRTAMASMetricEnumeration::DTDEV);
                this->stages[st].metrics.push_back(metric2);
            }
            else
            {
                for(int m=0;m<Nmetrics;m++)
                {
                    DRTAMASMetric metric;
                    std::string metric_string = parser->GetMetricString(st,m);
                    if(metric_string=="DEV")
                    {
                        metric.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
                    }
                    if(metric_string=="TR")
                    {
                        metric.SetMetricType( DRTAMASMetricEnumeration::DTTR);
                    }

                    this->stages[st].metrics.push_back(metric);
                }

            }
        }
    }
}




void DRTAMASRigid_Bulk::SetDefaultStages()
{
    int Nstr=parser->getNumberOfStructurals();


    {
        DRTAMASStageSettings curr_stage;                                   //1
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=4.;
        curr_stage.downsample_factor=8;
        curr_stage.learning_rate=0.05;
        curr_stage.update_gaussian_sigma=5.;
        curr_stage.total_gaussian_sigma=0.25;

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=1;
        curr_stage.metrics.push_back(metric2);



        this->stages.push_back(curr_stage);
    }
    {
        DRTAMASStageSettings curr_stage;                                   //2
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=3.;
        curr_stage.downsample_factor=6;
        curr_stage.learning_rate=0.1;
        curr_stage.update_gaussian_sigma=5.;
        curr_stage.total_gaussian_sigma=0.05;

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=1;
        curr_stage.metrics.push_back(metric2);




        this->stages.push_back(curr_stage);
    }
    {
        DRTAMASStageSettings curr_stage;                                   //3
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2.;
        curr_stage.downsample_factor=4;
        curr_stage.learning_rate=0.15;
        curr_stage.update_gaussian_sigma=4.5;
        curr_stage.total_gaussian_sigma=0.05;

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=2;
        curr_stage.metrics.push_back(metric1);

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=2;
        curr_stage.metrics.push_back(metric2);



        this->stages.push_back(curr_stage);
    }
    {
        DRTAMASStageSettings curr_stage;                                   //4
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=1.;
        curr_stage.downsample_factor=2;
        curr_stage.learning_rate=0.2;
        curr_stage.update_gaussian_sigma=4.;
        curr_stage.total_gaussian_sigma=0.0;

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=2;
        curr_stage.metrics.push_back(metric2);




        this->stages.push_back(curr_stage);
    }
    {
        DRTAMASStageSettings curr_stage;                                   //5
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=0.1;
        curr_stage.downsample_factor=1;
        curr_stage.learning_rate=0.2;
        curr_stage.update_gaussian_sigma=3.;
        curr_stage.total_gaussian_sigma=0.;

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=2;
        curr_stage.metrics.push_back(metric1);

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=2;
        curr_stage.metrics.push_back(metric2);



        this->stages.push_back(curr_stage);
    }



}


void DRTAMASRigid_Bulk::SetImagesForMetrics()
{
    CurrentImageType::Pointer fixed_TR_img = ComputeTRMapC(this->fixed_tensor);
    CurrentImageType::Pointer moving_TR_img = ComputeTRMapC(this->moving_tensor);

    CurrentImageType::Pointer preprocessed_fixed_TR = PreprocessImage(fixed_TR_img,0,1);
    CurrentImageType::Pointer preprocessed_moving_TR = PreprocessImage(moving_TR_img,0,1);


    int Nstg=this->stages.size();
    for(int st=0;st<Nstg;st++)
    {
        int Nmetrics = this->stages[st].metrics.size();
        for(int m=0;m<Nmetrics;m++)
        {
            if(this->stages[st].metrics[m].MetricType == DRTAMASMetricEnumeration::DTDEV)
            {
                this->stages[st].metrics[m].fixed_img= this->fixed_tensor;
                this->stages[st].metrics[m].moving_img= this->moving_tensor;
            }
            if(this->stages[st].metrics[m].MetricType == DRTAMASMetricEnumeration::DTTR)
            {
                this->stages[st].metrics[m].fixed_img= preprocessed_fixed_TR;
                this->stages[st].metrics[m].moving_img= preprocessed_moving_TR;
            }
        }
    }
}



void DRTAMASRigid_Bulk::Process()
{
    if(this->stages.size()==0)
    {
        SetUpStages();
    }
    SetImagesForMetrics();

    my_rigid_trans=RigidTransformType::New();
    my_rigid_trans->SetIdentity();


    {
        auto fixed_TR = ComputeTRMapC(stages[0].metrics[0].fixed_img);
        auto moving_TR = ComputeTRMapC(stages[0].metrics[0].moving_img);

        auto fixed_TR_itk= fixed_TR->CudaImageToITKImage();
        auto moving_TR_itk= moving_TR->CudaImageToITKImage();



        typedef itk::CenteredTransformInitializer<RigidTransformType, ImageType3D, ImageType3D> TransformInitializerType;
        typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();

        initializer->MomentsOn();

        initializer->SetTransform( my_rigid_trans );
        initializer->SetFixedImage( fixed_TR_itk );
        initializer->SetMovingImage( moving_TR_itk );
       // initializer->GeometryOn();
        initializer->InitializeTransform();
    }

    std::cout<<my_rigid_trans->GetParameters()<<std::endl;

    for(int st=0;st< stages.size();st++)
    {
        (*stream)<<"Stage number: "<<st+1<< " / " << stages.size() << std::endl;
        (*stream)<<"Current learning rate: "<<stages[st].learning_rate<<std::endl;
        (*stream)<<"Number of iterations: "<<stages[st].niter<<std::endl;
        (*stream)<<"Image smoothing stdev: "<<stages[st].img_smoothing_std<<std::endl;
        (*stream)<<"Downsampling factor: "<<stages[st].downsample_factor<<std::endl;

        (*stream)<<"Current update sigma: "<<stages[st].update_gaussian_sigma<<std::endl;
        (*stream)<<"Current total sigma: "<<stages[st].total_gaussian_sigma<<std::endl;

        (*stream)<<"Current metrics: ";
        for(int m=0;m<this->stages[st].metrics.size();m++)
            (*stream)<<this->stages[st].metrics[m].metric_name<<" ";
        (*stream)<<std::endl;

        stages[st].rigid_trans= my_rigid_trans;



        DRTAMASStageRigid current_stage(&(stages[st]));
        current_stage.PreprocessImagesAndFields();
        current_stage.RunDRTAMASStageRigid();

        my_rigid_trans= stages[st].rigid_trans;


    }

    std::cout<< my_rigid_trans->GetParameters() << std::endl;

}


#endif

