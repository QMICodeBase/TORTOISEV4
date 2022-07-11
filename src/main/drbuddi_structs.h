#ifndef _DRBUDDISTRUCTS_H
#define _DRBUDDISTRUCTS_H


#include "defines.h"

#ifdef USECUDA
#include "cuda_image.h"
#endif

enum DRBUDDIMetricEnumeration
{
    CCSK = 0,
    CCJac =1,
    CCJacS=2,
    MSJac = 3,
    CC=4,
    IllegalMetric = 5
};

class DRBUDDIMetric
{
public:

    void SetMetricType(DRBUDDIMetricEnumeration type)
    {
        MetricType=type;
        if(type==DRBUDDIMetricEnumeration::CCSK)
            metric_name="CCSK";
        if(type==DRBUDDIMetricEnumeration::CCJac)
            metric_name="CCJac";
        if(type==DRBUDDIMetricEnumeration::CCJacS)
            metric_name="CCJacS";
        if(type==DRBUDDIMetricEnumeration::MSJac)
            metric_name="MSJac";
        if(type==DRBUDDIMetricEnumeration::CC)
            metric_name="CC";
    }

    #ifdef USECUDA
        CUDAIMAGE::Pointer up_img{nullptr};
        CUDAIMAGE::Pointer down_img{nullptr};
        CUDAIMAGE::Pointer str_img{nullptr};
    #else
        ImageType3D::Pointer up_img{nullptr};
        ImageType3D::Pointer down_img{nullptr};
        ImageType3D::Pointer str_img{nullptr};
    #endif

    float weight{1};
    float param{5};

    std::string metric_name;
    DRBUDDIMetricEnumeration MetricType;
};



struct DRBUDDIStageSettings
{
    int niter{100};
    float img_smoothing_std{0};
    int downsample_factor{1};

    #ifdef USECUDA
        CUDAIMAGE::Pointer init_finv{nullptr};
        CUDAIMAGE::Pointer init_minv{nullptr};
        CUDAIMAGE::Pointer output_finv{nullptr};
        CUDAIMAGE::Pointer output_minv{nullptr};
    #else
        DisplacementFieldType::Pointer init_finv{nullptr};
        DisplacementFieldType::Pointer init_minv{nullptr};
        DisplacementFieldType::Pointer output_finv{nullptr};
        DisplacementFieldType::Pointer output_minv{nullptr};
    #endif

    std::vector<DRBUDDIMetric> metrics;

    float learning_rate{0.25};
    float update_gaussian_sigma{3};
    float total_gaussian_sigma{0};

    bool restrct{0};
    bool constrain{0};
};



#endif
