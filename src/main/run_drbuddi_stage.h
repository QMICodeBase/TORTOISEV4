#ifndef _RUNDRBUDDISTAGE_H
#define _RUNDRBUDDISTAGE_H

#include "drbuddi_structs.h"

#include "DRBUDDI_Diffeo.h"
#include "TORTOISE.h"

#ifdef USECUDA
#include "cuda_image.h"
#endif

#include "defines.h"
#include "itkTimeProbe.h"

class DRBUDDIStage
{
public:

    using CurrentFieldType = DRBUDDI_Diffeo::CurrentFieldType;
    using CurrentImageType = DRBUDDI_Diffeo::CurrentImageType;
    using PhaseEncodingVectorType=DRBUDDI_Diffeo::PhaseEncodingVectorType;

    DRBUDDIStage(DRBUDDIStageSettings *my_settings)
    {
        settings=my_settings;
        this->def_FINV= settings->init_finv;
        this->def_MINV= settings->init_minv;

        float tot_weight=0;
        for(int m=0;m<settings->metrics.size();m++)
            tot_weight+=settings->metrics[m].weight;
        for(int m=0;m<settings->metrics.size();m++)
            settings->metrics[m].weight/=tot_weight;

        #ifdef DRBUDDIALONE
            this->stream= &(std::cout);
        #else
            this->stream= TORTOISE::stream;
        #endif
    }
    ~DRBUDDIStage(){};


    void PreprocessImagesAndFields();
    void RunDRBUDDIStage();

    void SetUpPhaseEncoding(PhaseEncodingVectorType phase_vector)
    {
        up_phase_vector=phase_vector;
        phase_id=0;
        #ifdef USECUDA
            if( (fabs(phase_vector.y) > fabs(phase_vector.x)) && (fabs(phase_vector.y) > fabs(phase_vector.z)))
                    phase_id=1;
            if( (fabs(phase_vector.z) > fabs(phase_vector.x)) && (fabs(phase_vector.z) > fabs(phase_vector.y)))
                    phase_id=2;
        #else
            if( (fabs(phase_vector[1]) > fabs(phase_vector[0])) && (fabs(phase_vector[1]) > fabs(phase_vector[2])))
                    phase_id=1;
            if( (fabs(phase_vector[2]) > fabs(phase_vector[0])) && (fabs(phase_vector[2]) > fabs(phase_vector[1])))
                    phase_id=2;
        #endif
    }
    void SetDownPhaseEncoding(PhaseEncodingVectorType dv){down_phase_vector=dv;};
    void SetEstimateLRPerIteration(bool el){estimate_lr_per_iter=el;}

private:

    void CreateVirtualImage();


    DRBUDDIStageSettings *settings;


    std::vector<CurrentImageType::Pointer> resampled_smoothed_up_images;
    std::vector<CurrentImageType::Pointer> resampled_smoothed_down_images;
    std::vector<CurrentImageType::Pointer> resampled_smoothed_str_images;
    CurrentImageType::Pointer virtual_img{nullptr};

    CurrentFieldType::Pointer def_FINV{nullptr};
    CurrentFieldType::Pointer def_MINV{nullptr};

    CurrentFieldType::Pointer def_F{nullptr};
    CurrentFieldType::Pointer def_M{nullptr};

    PhaseEncodingVectorType up_phase_vector;
    PhaseEncodingVectorType down_phase_vector;
    int phase_id;

    itk::TimeProbe                    m_clock;
#ifdef DRBUDDIALONE
    std::ostream  *stream;
#else
    TORTOISE::TeeStream *stream;
#endif

    bool estimate_lr_per_iter{true};


};




#endif
