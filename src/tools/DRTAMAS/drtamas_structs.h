#ifndef _DRTAMASSTRUCTS_H
#define _DRTAMASSTRUCTS_H


#include "defines.h"
#include "cuda_image.h"

#include "TORTOISE.h"


enum DRTAMASMetricEnumeration
{
    DTDEV = 0,
    DTTR =1,
    DTCC=2,
    DTIllegalMetric = 3
};

class DRTAMASMetric
{
public:

    void SetMetricType(DRTAMASMetricEnumeration type)
    {
        MetricType=type;
        if(type==DRTAMASMetricEnumeration::DTCC)
            metric_name="CC";
        if(type==DRTAMASMetricEnumeration::DTDEV)
            metric_name="DEV";
        if(type==DRTAMASMetricEnumeration::DTTR)
            metric_name="TR";
    }


    CUDAIMAGE::Pointer fixed_img{nullptr};
    CUDAIMAGE::Pointer moving_img{nullptr};

    float weight{1};
    float param{5};
    bool to{1};

    std::string metric_name;
    DRTAMASMetricEnumeration MetricType;
};



struct DRTAMASStageSettings
{
    int niter{100};
    float img_smoothing_std{0};
    int downsample_factor{1};


    CUDAIMAGE::Pointer init_finv{nullptr};
    CUDAIMAGE::Pointer init_minv{nullptr};
    CUDAIMAGE::Pointer output_finv{nullptr};
    CUDAIMAGE::Pointer output_minv{nullptr};
    using RigidTransformType=TORTOISE::RigidTransformType;

    RigidTransformType::Pointer rigid_trans{nullptr};

    std::vector<DRTAMASMetric> metrics;

    float learning_rate{0.25};
    float update_gaussian_sigma{3};
    float total_gaussian_sigma{0};

};



#endif
