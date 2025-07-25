#ifndef _DRBUDDIDIFFEO_CXX
#define _DRBUDDIDIFFEO_CXX




#include "DRBUDDI_Diffeo.h"
#include "run_drbuddi_stage.h"


#ifdef USECUDA
    #include "cuda_image_utilities.h"
    #include "run_drbuddi_stage_TVVF.h"
#else
    #include "drbuddi_image_utilities.h"
#endif

void DRBUDDI_Diffeo::SetUpStages()
{
    int Nstg= parser->getNumberOfStages();

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
            this->stages[st].restrct=parser->GetRestrict(st);
            this->stages[st].constrain=parser->GetConstrain(st);

            int Nmetrics= parser->GetNMetrics(st);
            for(int m=0;m<Nmetrics;m++)
            {
                DRBUDDIMetric metric;
                std::string metric_string = parser->GetMetricString(st,m);
                if(metric_string=="MSJac")
                {
                    metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
                }
                if(metric_string=="CC")
                {
                    metric.SetMetricType( DRBUDDIMetricEnumeration::CC);
                }
                if(metric_string.find("CCSK")!=std::string::npos)
                {
                    metric.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
                }
                if(metric_string=="CCJac")
                {
                    metric.SetMetricType( DRBUDDIMetricEnumeration::CCJac);
                }
                if(metric_string.find("CCJacS")!=std::string::npos)
                {
                    metric.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
                }

                this->stages[st].metrics.push_back(metric);

            }
        }
    }
}




void DRBUDDI_Diffeo::SetDefaultStages()
{
    int Nstr=parser->getNumberOfStructurals();

    float str_weight=parser->getStructuralWeight();


    {
        DRBUDDIStageSettings curr_stage;                                //1
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=4.5;
        curr_stage.downsample_factor=8;
        curr_stage.learning_rate=0.35;
        curr_stage.update_gaussian_sigma=13.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        curr_stage.metrics.push_back(metric);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                //2
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=4.;
        curr_stage.downsample_factor=8;
        curr_stage.learning_rate=0.25;
        curr_stage.update_gaussian_sigma=11.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        curr_stage.metrics.push_back(metric);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                //3
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=3.5;
        curr_stage.downsample_factor=8;
        curr_stage.learning_rate=0.15;
        curr_stage.update_gaussian_sigma=9.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        curr_stage.metrics.push_back(metric);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                //4
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=3;
        curr_stage.downsample_factor=8;
        curr_stage.learning_rate=0.3;
        curr_stage.update_gaussian_sigma=7.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        curr_stage.metrics.push_back(metric);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                //5
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2.5;
        curr_stage.downsample_factor=8;
        curr_stage.learning_rate=0.05;
        curr_stage.update_gaussian_sigma=5.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        curr_stage.metrics.push_back(metric);
        this->stages.push_back(curr_stage);
    }

    {
        DRBUDDIStageSettings curr_stage;                                //6
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2;
        curr_stage.downsample_factor=8;
        curr_stage.learning_rate=0.2;
        curr_stage.update_gaussian_sigma=4.5;
        curr_stage.total_gaussian_sigma=0.0;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1.;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric2.weight=0.8*str_weight;
            curr_stage.metrics.push_back(metric2);
        }
        this->stages.push_back(curr_stage);
    }    
    {
        DRBUDDIStageSettings curr_stage;                                //7
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=4;
        curr_stage.downsample_factor=8;
        curr_stage.learning_rate=0.5;
        curr_stage.update_gaussian_sigma=5.;
        curr_stage.total_gaussian_sigma=0.0;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=0.8;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1.*str_weight;
            curr_stage.metrics.push_back(metric2);
        }
        this->stages.push_back(curr_stage);
    }


    {
        DRBUDDIStageSettings curr_stage;                                 //8
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=4;
        curr_stage.downsample_factor=6;
        curr_stage.learning_rate=0.35;
        curr_stage.update_gaussian_sigma=11.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        if((this->FA_up_img  && this->FA_down_img) )
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CC);
            metric2.weight=1;
            curr_stage.metrics.push_back(metric2);
        }
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=1*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                 //9
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=3.5;
        curr_stage.downsample_factor=6;
        curr_stage.learning_rate=0.3;
        curr_stage.update_gaussian_sigma=9.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        if((this->FA_up_img  && this->FA_down_img) )
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CC);
            metric2.weight=1;
            curr_stage.metrics.push_back(metric2);
        }
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=1*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                  //10
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=3;
        curr_stage.downsample_factor=6;
        curr_stage.learning_rate=0.4;
        curr_stage.update_gaussian_sigma=7.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        if((this->FA_up_img  && this->FA_down_img) )
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CC);
            metric2.weight=1;
            curr_stage.metrics.push_back(metric2);
        }

        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=1*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                //11
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2.5;
        curr_stage.downsample_factor=6;
        curr_stage.learning_rate=0.2;
        curr_stage.update_gaussian_sigma=6.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        if((this->FA_up_img  && this->FA_down_img) )
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CC);
            metric2.weight=1;
            curr_stage.metrics.push_back(metric2);
        }
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=1*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                              //12
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2;
        curr_stage.downsample_factor=6;
        curr_stage.learning_rate=0.1;
        curr_stage.update_gaussian_sigma=5.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;        
        curr_stage.metrics.push_back(metric1);
        if((this->FA_up_img  && this->FA_down_img) )
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CC);
            metric2.weight=1;
            curr_stage.metrics.push_back(metric2);
        }

        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=1*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                            //13
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2;
        curr_stage.downsample_factor=6;
        curr_stage.learning_rate=0.15;
        curr_stage.update_gaussian_sigma=4.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }    
    {
        DRBUDDIStageSettings curr_stage;                            //14
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2;
        curr_stage.downsample_factor=6;
        curr_stage.learning_rate=0.75;
        curr_stage.update_gaussian_sigma=3.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=0.8;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
        }
        if((this->FA_up_img  && this->FA_down_img) )
        {
            DRBUDDIMetric metric3;
            metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
            metric3.weight=1;
            curr_stage.metrics.push_back(metric3);
        }
        this->stages.push_back(curr_stage);
    }



    {
        DRBUDDIStageSettings curr_stage;                                  //15
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=3.;
        curr_stage.downsample_factor=4;
        curr_stage.learning_rate=0.75;
        curr_stage.update_gaussian_sigma=9.5;
        curr_stage.total_gaussian_sigma=0.2;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);

            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                   //16
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=4.;
        curr_stage.downsample_factor=4;
        curr_stage.learning_rate=0.75;
        curr_stage.update_gaussian_sigma=7.5;
        curr_stage.total_gaussian_sigma=0.1;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                //17
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=3.5;
        curr_stage.downsample_factor=4;
        curr_stage.learning_rate=0.5;
        curr_stage.update_gaussian_sigma=5.5;
        curr_stage.total_gaussian_sigma=0.1;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                   //18
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=3;
        curr_stage.downsample_factor=4;
        curr_stage.learning_rate=0.35;
        curr_stage.update_gaussian_sigma=4.5;
        curr_stage.total_gaussian_sigma=0.1;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                     //19
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=3.;
        curr_stage.downsample_factor=4;
        curr_stage.learning_rate=0.25;
        curr_stage.update_gaussian_sigma=3.5;
        curr_stage.total_gaussian_sigma=0.1;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                               //20
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2.5;
        curr_stage.downsample_factor=4;
        curr_stage.learning_rate=.2;
        curr_stage.update_gaussian_sigma=3.0;
        curr_stage.total_gaussian_sigma=0.1;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                               //21
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2.;
        curr_stage.downsample_factor=4;
        curr_stage.learning_rate=1.;
        curr_stage.update_gaussian_sigma=5;
        curr_stage.total_gaussian_sigma=0.;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=0.5;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }


    {
        DRBUDDIStageSettings curr_stage;                                 //22
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=2.;
        curr_stage.downsample_factor=2;
        curr_stage.learning_rate=1.;
        curr_stage.update_gaussian_sigma=7.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                             //23
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=1.;
        curr_stage.downsample_factor=2;
        curr_stage.learning_rate=0.85;
        curr_stage.update_gaussian_sigma=5.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric2);
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=1*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                //24
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=0.5;
        curr_stage.downsample_factor=2;
        curr_stage.learning_rate=0.75;
        curr_stage.update_gaussian_sigma=4.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=0.8;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                   //25
        curr_stage.niter=300;
        curr_stage.img_smoothing_std=1.;
        curr_stage.downsample_factor=1;
        curr_stage.learning_rate=1.5;
        curr_stage.update_gaussian_sigma=7.5;
        curr_stage.total_gaussian_sigma=0.05;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1*str_weight;
            curr_stage.metrics.push_back(metric2);
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        this->stages.push_back(curr_stage);
    }
    {
        DRBUDDIStageSettings curr_stage;                                   //26
        curr_stage.niter=60;
        curr_stage.img_smoothing_std=0.25;
        curr_stage.downsample_factor=1;
        curr_stage.learning_rate=1.0;
        curr_stage.update_gaussian_sigma=5.5;
        curr_stage.total_gaussian_sigma=0.0;
        curr_stage.restrct=1;
        curr_stage.constrain=1;
        DRBUDDIMetric metric1;
        metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric1.weight=1;
        curr_stage.metrics.push_back(metric1);

        DRBUDDIMetric metric3;
        metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
        metric3.weight=1;
        curr_stage.metrics.push_back(metric3);
        for(int s=0;s<Nstr;s++)
        {
            DRBUDDIMetric metric2;
            metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
            metric2.weight=1.1*str_weight;
            curr_stage.metrics.push_back(metric2);
            DRBUDDIMetric metric4;
            metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
            metric4.weight=0.5*str_weight;
            curr_stage.metrics.push_back(metric4);
        }
        this->stages.push_back(curr_stage);

    }

    if(!parser->getEnforceFullAntiSymmetry())
    {
        if(Nstr>0 || (this->FA_up_img  && this->FA_down_img) )
        {
            DRBUDDIStageSettings curr_stage;                                   //27
            curr_stage.niter=300;
            curr_stage.img_smoothing_std=0.;
            curr_stage.downsample_factor=1;
            curr_stage.learning_rate=0.9;
            curr_stage.update_gaussian_sigma=4.5;
            curr_stage.total_gaussian_sigma=0.0;
            curr_stage.restrct=0;
            curr_stage.constrain=0;
            DRBUDDIMetric metric1;
            metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
            metric1.weight=0.5;
            curr_stage.metrics.push_back(metric1);
            DRBUDDIMetric metric3;
            metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
            metric3.weight=1.5;
            curr_stage.metrics.push_back(metric3);

            for(int s=0;s<Nstr;s++)
            {
                DRBUDDIMetric metric2;
                metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
                metric2.weight=1.5*str_weight;
                curr_stage.metrics.push_back(metric2);
                DRBUDDIMetric metric4;
                metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
                metric4.weight=1.*str_weight;
                curr_stage.metrics.push_back(metric4);
            }
            this->stages.push_back(curr_stage);
        }

        if(parser->getDisableLastStage()==0 && Nstr>0 && (this->FA_up_img  && this->FA_down_img) )
        {
            DRBUDDIStageSettings curr_stage;                                   //28
            curr_stage.niter=300;
            curr_stage.img_smoothing_std=0.;
            curr_stage.downsample_factor=1;
            curr_stage.learning_rate=0.1;
            curr_stage.update_gaussian_sigma=3.;
            curr_stage.total_gaussian_sigma=0.;
            curr_stage.restrct=0;
            curr_stage.constrain=0;
            DRBUDDIMetric metric1;
            metric1.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
            metric1.weight=0.2;
            curr_stage.metrics.push_back(metric1);
            DRBUDDIMetric metric3;
            metric3.SetMetricType( DRBUDDIMetricEnumeration::CC);
            metric3.weight=1.2;
            curr_stage.metrics.push_back(metric3);

            for(int s=0;s<Nstr;s++)
            {
                DRBUDDIMetric metric2;
                metric2.SetMetricType( DRBUDDIMetricEnumeration::CCSK);
                metric2.weight=1.2*str_weight;
                curr_stage.metrics.push_back(metric2);
                DRBUDDIMetric metric4;
                metric4.SetMetricType( DRBUDDIMetricEnumeration::CCJacS);
                metric4.weight=0.8*str_weight;
                curr_stage.metrics.push_back(metric4);
            }
            this->stages.push_back(curr_stage);
        }
    }
}





void DRBUDDI_Diffeo::SetImagesForMetrics()
{
    CurrentImageType::Pointer preprocessed_b0_up= PreprocessImage(this->b0_up_img,0,1);
    CurrentImageType::Pointer preprocessed_b0_down= PreprocessImage(this->b0_down_img,0,1);

    int Nstg= this->stages.size();
    int Nstr=parser->getNumberOfStructurals();

    for(int st=0;st<Nstg;st++)
    {
        bool redo=true;
        while(redo)
        {
            int str_id2_CCJacS=0;
            int str_id2_CCSK=0;
            redo=false;
            int Nmetrics = this->stages[st].metrics.size();

            for(int m=0;m<Nmetrics;m++)
            {
                if(this->stages[st].metrics[m].MetricType == DRBUDDIMetricEnumeration::MSJac)
                {
                    this->stages[st].metrics[m].up_img= this->b0_up_img;
                    this->stages[st].metrics[m].down_img= this->b0_down_img;
                }
                if(this->stages[st].metrics[m].MetricType == DRBUDDIMetricEnumeration::CC)
                {
                    if(this->FA_up_img !=nullptr)
                    {
                        this->stages[st].metrics[m].up_img= this->FA_up_img;
                        this->stages[st].metrics[m].down_img= this->FA_down_img;
                    }
                    else
                    {
                        this->stages[st].metrics.erase(this->stages[st].metrics.begin()+m);
                        redo=true;
                        break;
                    }
                }
                if(this->stages[st].metrics[m].MetricType == DRBUDDIMetricEnumeration::CCSK)
                {
                    if(Nstr>0)
                    {
                        this->stages[st].metrics[m].up_img= preprocessed_b0_up;
                        this->stages[st].metrics[m].down_img= preprocessed_b0_down;

                        std::string metric_string = parser->GetMetricString(st,m);
                        if(metric_string !="")
                        {
                            std::string sub_string = metric_string.substr(metric_string.find("str_id=")+7,metric_string.rfind("}")-7-metric_string.find("str_id="));
                            int str_id = atoi(sub_string.c_str());

                            this->stages[st].metrics[m].str_img = PreprocessImage(this->structural_imgs[str_id],0,1);
                        }
                        else
                        {
                            this->stages[st].metrics[m].str_img = PreprocessImage(this->structural_imgs[str_id2_CCSK],0,1);
                            str_id2_CCSK++;
                        }
                    }
                    else
                    {
                        this->stages[st].metrics.erase(this->stages[st].metrics.begin()+m);
                        redo=true;
                        break;
                    }
                }
                if(this->stages[st].metrics[m].MetricType == DRBUDDIMetricEnumeration::CCJacS)
                {
                    if(Nstr>0)
                    {
                        this->stages[st].metrics[m].up_img= preprocessed_b0_up;
                        this->stages[st].metrics[m].down_img= preprocessed_b0_down;

                        std::string metric_string = parser->GetMetricString(st,m);
                        if(metric_string !="")
                        {
                            std::string sub_string = metric_string.substr(metric_string.find("str_id=")+7,metric_string.rfind("}")-7-metric_string.find("str_id="));
                            int str_id = atoi(sub_string.c_str());

                            this->stages[st].metrics[m].str_img = PreprocessImage(this->structural_imgs[str_id],0,1);
                        }
                        else
                        {
                            this->stages[st].metrics[m].str_img = PreprocessImage(this->structural_imgs[str_id2_CCJacS],0,1);
                            str_id2_CCJacS++;
                        }
                        this->stages[st].metrics[m].weight=0.5;
                    }
                    else
                    {
                        this->stages[st].metrics.erase(this->stages[st].metrics.begin()+m);
                        redo=true;
                        break;
                    }
                }

            }
        } //while redo

/*        if(Nstg > 20 && st<21)
        {
            DRBUDDIMetric metric1;
            metric1.SetMetricType( DRBUDDIMetricEnumeration::CC);
            metric1.weight=1.;
            metric1.up_img= preprocessed_b0_up;
            metric1.down_img= preprocessed_b0_down;

            this->stages[st].metrics.push_back(metric1);
        }
*/
    }  //for stage
}



void DRBUDDI_Diffeo::Process()
{
    if(this->stages.size()==0)
    {
        SetUpStages();
    }
    SetImagesForMetrics();


    CurrentFieldType::Pointer prev_finv=nullptr;
    CurrentFieldType::Pointer prev_minv=nullptr;
    std::vector<CurrentFieldType::Pointer> init_vfield;

    if(parser->GetInitialFINV()!="")
    {
        DisplacementFieldType::Pointer init_finv= readImageD<DisplacementFieldType>(parser->GetInitialFINV());
        #ifdef USECUDA
            prev_finv=CurrentFieldType::New();
            prev_finv->SetImageFromITK(init_finv);
        #else
            prev_finv=init_finv;
        #endif
    }
    if(parser->GetInitialMINV()!="")
    {
        DisplacementFieldType::Pointer init_minv= readImageD<DisplacementFieldType>(parser->GetInitialMINV());
        #ifdef USECUDA
            prev_minv=CurrentFieldType::New();
            prev_minv->SetImageFromITK(init_minv);
        #else
            prev_minv=init_minv;
        #endif
    }

    #ifdef USECUDA
        DRBUDDIStage_TVVF prev_stage(&(stages[0]));
        DRBUDDIStage_TVVF final_stage;
    #endif



    for(int st=0;st< stages.size();st++)
    {
        (*stream)<<"Stage number: "<<st+1<< " / " << stages.size() << std::endl;
        (*stream)<<"Current learning rate: "<<stages[st].learning_rate<<std::endl;
        (*stream)<<"Number of iterations: "<<stages[st].niter<<std::endl;
        (*stream)<<"Image smoothing stdev: "<<stages[st].img_smoothing_std<<std::endl;
        (*stream)<<"Downsampling factor: "<<stages[st].downsample_factor<<std::endl;

        (*stream)<<"Current update sigma: "<<stages[st].update_gaussian_sigma<<std::endl;
        (*stream)<<"Current total sigma: "<<stages[st].total_gaussian_sigma<<std::endl;

        (*stream)<<"Current restrict: "<<stages[st].restrct<<std::endl;
        (*stream)<<"Current constrain: "<<stages[st].constrain<<std::endl;
        (*stream)<<"Current metrics: ";
        for(int m=0;m<this->stages[st].metrics.size();m++)
            (*stream)<<this->stages[st].metrics[m].metric_name<<"\{"<<this->stages[st].metrics[m].weight<<"\} ";
        (*stream)<<std::endl;


        #ifdef USECUDA
            if(this->GetRegistrationMethodType()=="TVVF")
            {
                DRBUDDIStageSettings new_stage= stages[st];
                new_stage.init_finv=nullptr;
                new_stage.init_minv=nullptr;
                new_stage.init_finv_const=prev_finv;
                new_stage.init_minv_const=prev_minv;
                new_stage.init_vfield=init_vfield;

                DRBUDDIStage_TVVF current_stage(&new_stage);
                current_stage.SetUpPhaseEncoding(up_phase_vector);
                current_stage.SetDownPhaseEncoding(down_phase_vector);


                current_stage.PreprocessImagesAndFields();
                current_stage.RunDRBUDDIStage();
                init_vfield=current_stage.GetVelocityfield();

                final_stage=current_stage;
            }
            else
            {
                if(st>=27)
                {
                    stages[st].init_finv_const=prev_finv;
                    stages[st].init_minv_const=prev_minv;
                }
                else
                {
                    if(prev_finv && prev_minv)
                    {
                        stages[st].init_finv=prev_finv;
                        stages[st].init_minv=prev_minv;
                    }
                }

                DRBUDDIStage current_stage(&(stages[st]));
                current_stage.SetUpPhaseEncoding(up_phase_vector);
                current_stage.SetDownPhaseEncoding(down_phase_vector);
                current_stage.SetEstimateLRPerIteration(parser->getEstimateLRPerIteration());

                current_stage.PreprocessImagesAndFields();
                current_stage.RunDRBUDDIStage();

                prev_finv= stages[st].output_finv;
                prev_minv= stages[st].output_minv;
            }

        #else
            if(st>=27)
            {
                stages[st].init_finv_const=prev_finv;
                stages[st].init_minv_const=prev_minv;
            }
            else
            {
                if(prev_finv && prev_minv)
                {
                    stages[st].init_finv=prev_finv;
                    stages[st].init_minv=prev_minv;
                }
            }

            DRBUDDIStage current_stage(&(stages[st]));
            current_stage.SetUpPhaseEncoding(up_phase_vector);
            current_stage.SetDownPhaseEncoding(down_phase_vector);
            current_stage.SetEstimateLRPerIteration(parser->getEstimateLRPerIteration());

            current_stage.PreprocessImagesAndFields();
            current_stage.RunDRBUDDIStage();

            prev_finv= stages[st].output_finv;
            prev_minv= stages[st].output_minv;
        #endif

    } //for stages

    #ifdef USECUDA
        if(this->GetRegistrationMethodType()=="TVVF")
        {
            final_stage.ComputeFields(this->def_FINV, this->def_MINV);
            if(prev_finv)
                this->def_FINV=ComposeFields(prev_finv,this->def_FINV);
            if(prev_minv)
                this->def_MINV=ComposeFields(prev_minv,this->def_MINV);
        }
        else
        {
            this->def_FINV= stages[stages.size()-1].output_finv;
            this->def_MINV= stages[stages.size()-1].output_minv;
        }

    #else
        this->def_FINV= stages[stages.size()-1].output_finv;
        this->def_MINV= stages[stages.size()-1].output_minv;
    #endif


}


DisplacementFieldType::Pointer DRBUDDI_Diffeo::getUp2DownINV()
{
    auto disp_f = InvertField(def_FINV);
    auto disp2a = ComposeFields(disp_f, def_MINV);
    auto disp2= InvertField(disp2a);


#ifdef USECUDA
    DisplacementFieldType::Pointer disp=disp2->CudaImageToITKField();
#else
    DisplacementFieldType::Pointer disp=disp2;
#endif
    disp->SetDirection(orig_dir);
    itk::ImageRegionIterator<DisplacementFieldType> it(disp,disp->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        DisplacementFieldType::PixelType pix= it.Get();
        vnl_vector<double> vec=orig_dir.GetVnlMatrix()*new_dir.GetTranspose()* pix.GetVnlVector();
        pix[0]=vec[0];
        pix[1]=vec[1];
        pix[2]=vec[2];
        it.Set(pix);
    }
    return disp;
}

#endif

