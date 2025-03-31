#ifndef _DRTAMASDIFFEO_CXX
#define _DRTAMASDIFFEO_CXX




#include "DRTAMAS_Diffeo.h"
#include "DRTAMAS_utilities.h"
#include "run_drtamas_stage.h"



//#include "run_DRTAMAS_stage_TVVF.h"


void DRTAMAS_Diffeo::SetUpStages()
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

                if(Nstr!=0)
                {
                    DRTAMASMetric metric3;
                    metric3.SetMetricType( DRTAMASMetricEnumeration::DTCC);
                    this->stages[st].metrics.push_back(metric3);
                }
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
                    if(metric_string.find("CC")!=std::string::npos)
                    {
                        metric.SetMetricType( DRTAMASMetricEnumeration::DTCC);
                    }

                    this->stages[st].metrics.push_back(metric);
                }

            }
        }
    }
}




void DRTAMAS_Diffeo::SetDefaultStages()
{
    int Nstr=parser->getNumberOfStructurals();


    {
        DRTAMASStageSettings curr_stage;                                   //1
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=4.;
        curr_stage.downsample_factor=8;
        curr_stage.learning_rate=0.1;
        curr_stage.update_gaussian_sigma=5.;
        curr_stage.total_gaussian_sigma=0.05;

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=2;
        curr_stage.metrics.push_back(metric2);

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);

        for(int s=0;s<Nstr;s++)
        {
            DRTAMASMetric metric3;
            metric3.SetMetricType( DRTAMASMetricEnumeration::DTCC);
            metric3.weight=1;
            curr_stage.metrics.push_back(metric3);
        }
        this->stages.push_back(curr_stage);
    }
    {
        DRTAMASStageSettings curr_stage;                                   //2
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=3.;
        curr_stage.downsample_factor=6;
        curr_stage.learning_rate=0.2;
        curr_stage.update_gaussian_sigma=4.;
        curr_stage.total_gaussian_sigma=0.05;

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=2;
        curr_stage.metrics.push_back(metric2);

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);

        for(int s=0;s<Nstr;s++)
        {
            DRTAMASMetric metric3;
            metric3.SetMetricType( DRTAMASMetricEnumeration::DTCC);
            metric3.weight=1;
            curr_stage.metrics.push_back(metric3);
        }
        this->stages.push_back(curr_stage);
    }
    {
        DRTAMASStageSettings curr_stage;                                   //3
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2.;
        curr_stage.downsample_factor=4;
        curr_stage.learning_rate=0.25;
        curr_stage.update_gaussian_sigma=4.;
        curr_stage.total_gaussian_sigma=0.0;

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=3;
        curr_stage.metrics.push_back(metric2);

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=2;
        curr_stage.metrics.push_back(metric1);

        for(int s=0;s<Nstr;s++)
        {
            DRTAMASMetric metric3;
            metric3.SetMetricType( DRTAMASMetricEnumeration::DTCC);
            metric3.weight=1;
            curr_stage.metrics.push_back(metric3);
        }
        this->stages.push_back(curr_stage);
    }
    {
        DRTAMASStageSettings curr_stage;                                   //4
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=1.;
        curr_stage.downsample_factor=2;
        curr_stage.learning_rate=0.5;
        curr_stage.update_gaussian_sigma=3.;
        curr_stage.total_gaussian_sigma=0.0;

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=3;
        curr_stage.metrics.push_back(metric2);

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=2;
        curr_stage.metrics.push_back(metric1);

        for(int s=0;s<Nstr;s++)
        {
            DRTAMASMetric metric3;
            metric3.SetMetricType( DRTAMASMetricEnumeration::DTCC);
            metric3.weight=1;
            curr_stage.metrics.push_back(metric3);
        }
        this->stages.push_back(curr_stage);
    }
    {
        DRTAMASStageSettings curr_stage;                                   //5
        curr_stage.niter=100;
        curr_stage.img_smoothing_std=0.;
        curr_stage.downsample_factor=1;
        curr_stage.learning_rate=1.5;

        if(parser->getNoSmoothingLastStage())
        {
            curr_stage.update_gaussian_sigma=0.25;
            curr_stage.total_gaussian_sigma=0;
        }
        else
        {
            curr_stage.update_gaussian_sigma=3.;
            curr_stage.total_gaussian_sigma=0;
        }

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=3;
        curr_stage.metrics.push_back(metric2);

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=2;
        curr_stage.metrics.push_back(metric1);

        for(int s=0;s<Nstr;s++)
        {
            DRTAMASMetric metric3;
            metric3.SetMetricType( DRTAMASMetricEnumeration::DTCC);
            metric3.weight=1;
            curr_stage.metrics.push_back(metric3);
        }
        this->stages.push_back(curr_stage);
    }
    {
        DRTAMASStageSettings curr_stage;                                   //6
        curr_stage.niter=100;
        curr_stage.img_smoothing_std=0.;
        curr_stage.downsample_factor=1;
        curr_stage.learning_rate=1.25;

        if(parser->getNoSmoothingLastStage())
        {
            curr_stage.update_gaussian_sigma=0.25;
            curr_stage.total_gaussian_sigma=0;
        }
        else
        {
            curr_stage.update_gaussian_sigma=3.;
            curr_stage.total_gaussian_sigma=0;
        }

        DRTAMASMetric metric2;
        metric2.SetMetricType(DRTAMASMetricEnumeration::DTTR);
        metric2.weight=2;
        curr_stage.metrics.push_back(metric2);

        DRTAMASMetric metric1;
        metric1.SetMetricType(DRTAMASMetricEnumeration::DTDEV);
        metric1.weight=2;
        metric1.to=0;
        curr_stage.metrics.push_back(metric1);

        for(int s=0;s<Nstr;s++)
        {
            DRTAMASMetric metric3;
            metric3.SetMetricType( DRTAMASMetricEnumeration::DTCC);
            metric3.weight=1;
            curr_stage.metrics.push_back(metric3);
        }
        this->stages.push_back(curr_stage);
    }

}


void DRTAMAS_Diffeo::SetImagesForMetrics()
{
    CurrentImageType::Pointer fixed_TR_img = ComputeTRMapC(this->fixed_tensor);
    CurrentImageType::Pointer moving_TR_img = ComputeTRMapC(this->moving_tensor);

    CurrentImageType::Pointer preprocessed_fixed_TR = PreprocessImage(fixed_TR_img,0,1);
    CurrentImageType::Pointer preprocessed_moving_TR = PreprocessImage(moving_TR_img,0,1);

    int Nstr=parser->getNumberOfStructurals();
    std::vector<CurrentImageType::Pointer> preprocessed_fixed_structurals, preprocessed_moving_structurals;
    for(int s=0;s<Nstr;s++)
    {
        preprocessed_fixed_structurals.push_back(PreprocessImage(this->fixed_structurals[s],0,1));
        preprocessed_moving_structurals.push_back(PreprocessImage(this->moving_structurals[s],0,1));
    }


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
            if(this->stages[st].metrics[m].MetricType == DRTAMASMetricEnumeration::DTCC)
            {
                if(Nstr==0)
                {
                    this->stages[st].metrics.pop_back();
                }
                else
                {
                    //this->stages[st].metrics[m].fixed_img= preprocessed_fixed_structurals[0];
                    //this->stages[st].metrics[m].moving_img= preprocessed_moving_structurals[0];

                    for(int s=0;s<Nstr;s++)
                    {
                        //this->stages[st].metrics.push_back(this->stages[st].metrics[m]);
                        this->stages[st].metrics[m+s].fixed_img= preprocessed_fixed_structurals[s];
                        this->stages[st].metrics[m+s].moving_img= preprocessed_moving_structurals[s];
                    }
                    m+=Nstr;
                }
            }

        }
    }
}



void DRTAMAS_Diffeo::Process()
{
    if(this->stages.size()==0)
    {
        SetUpStages();
    }
    SetImagesForMetrics();


    CurrentFieldType::Pointer prev_finv=nullptr;
    CurrentFieldType::Pointer prev_minv=nullptr;

    if(parser->GetInitialFINV()!="")
    {
        DisplacementFieldType::Pointer init_finv= readImageD<DisplacementFieldType>(parser->GetInitialFINV());
        prev_finv=CurrentFieldType::New();
        prev_finv->SetImageFromITK(init_finv);
    }
    if(parser->GetInitialMINV()!="")
    {
        DisplacementFieldType::Pointer init_minv= readImageD<DisplacementFieldType>(parser->GetInitialMINV());
        prev_minv=CurrentFieldType::New();
        prev_minv->SetImageFromITK(init_minv);
    }


    //DRTAMASStage_TVVF prev_stage(&(stages[0]));



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



        if(prev_finv && prev_minv)
        {
            stages[st].init_finv=prev_finv;
            stages[st].init_minv=prev_minv;
        }

        DRTAMASStage current_stage(&(stages[st]));
        current_stage.PreprocessImagesAndFields();
        current_stage.RunDRTAMASStage();

        prev_finv= stages[st].output_finv;
        prev_minv= stages[st].output_minv;

    }

    this->def_FINV= stages[stages.size()-1].output_finv;
    this->def_MINV= stages[stages.size()-1].output_minv;




}


#endif

