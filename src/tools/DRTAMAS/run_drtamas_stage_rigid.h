#ifndef _RUNDRTAMASStageRigid_H
#define _RUNDRTAMASStageRigid_H

#include "drtamas_structs.h"

#include "DRTAMASRigid_Bulk.h"
#include "TORTOISE.h"

#ifdef USECUDA
#include "cuda_image.h"
#endif

#include "defines.h"
#include "itkTimeProbe.h"

class DRTAMASStageRigid
{
public:

    using CurrentImageType=CUDAIMAGE;
    using RigidTransformType=TORTOISE::RigidTransformType;


    DRTAMASStageRigid(){};
    ~DRTAMASStageRigid(){};

    DRTAMASStageRigid(DRTAMASStageSettings *my_settings)
    {
        settings=my_settings;        

        float tot_weight=0;
        for(int m=0;m<settings->metrics.size();m++)
            tot_weight+=settings->metrics[m].weight;
        for(int m=0;m<settings->metrics.size();m++)
            settings->metrics[m].weight/=tot_weight;

        this->stream= &(std::cout);

    }




    void RunDRTAMASStageRigid();
    void PreprocessImagesAndFields();


private:
    std::vector<double>  ComputeMetrics(CUDAIMAGE::Pointer fixed_img, std::vector<CUDAIMAGE::Pointer> moving_img_vec, RigidTransformType::Pointer rigid_trans);

    std::vector< std::pair<double,double> > BracketGrad(CUDAIMAGE::Pointer fixed_img, std::vector<CUDAIMAGE::Pointer> moving_img_vec,RigidTransformType::ParametersType params, vnl_vector<double> grad, int mode);
    double GoldenSearch(CUDAIMAGE::Pointer fixed_img, std::vector<CUDAIMAGE::Pointer> moving_img_vec,RigidTransformType::ParametersType params, std::vector< std::pair<double,double> > &x_f_pairs, vnl_vector<double> grad, int mode);

    double ComputeMetricRigid_TR(CUDAIMAGE::Pointer fixed_tensor, CUDAIMAGE::Pointer  moving_tensor);
    double ComputeMetricRigid_DEV(CUDAIMAGE::Pointer fixed_tensor, CUDAIMAGE::Pointer  moving_tensor);

protected:
    void CreateVirtualImage();


    DRTAMASStageSettings *settings;
    CurrentImageType::Pointer virtual_img{nullptr};

    itk::TimeProbe                    m_clock;
    std::ostream  *stream;

    double class_metric_value{0};



};




#endif
