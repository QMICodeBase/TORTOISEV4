#ifndef _RUNDRBUDDIStage_TVVF_H
#define _RUNDRBUDDIStage_TVVF_H

#include "drbuddi_structs.h"

#include "DRBUDDI_Diffeo.h"
#include "TORTOISE.h"

#ifdef USECUDA
#include "cuda_image.h"
#endif

#include "defines.h"
#include "itkTimeProbe.h"
#include "run_drbuddi_stage.h"

class DRBUDDIStage_TVVF: public DRBUDDIStage
{
public:

    using CurrentFieldType = DRBUDDI_Diffeo::CurrentFieldType;
    using CurrentImageType = DRBUDDI_Diffeo::CurrentImageType;
    using PhaseEncodingVectorType=DRBUDDI_Diffeo::PhaseEncodingVectorType;

    DRBUDDIStage_TVVF(){};
    DRBUDDIStage_TVVF(DRBUDDIStageSettings *my_settings)
    {
        settings=my_settings;
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
    ~DRBUDDIStage_TVVF(){};


    void RunDRBUDDIStage();
    void PreprocessImagesAndFields();
    std::vector<CurrentFieldType::Pointer> GetVelocityfield(){return velocity_field;}
    void SetVelocityField(std::vector<CurrentFieldType::Pointer> vf){velocity_field=vf;}

    CurrentFieldType::Pointer IntegrateVelocityField(float lowt, float hight);


    void ComputeFields(CurrentFieldType::Pointer &def_finv, CurrentFieldType::Pointer &def_minv)
    {
        def_finv =IntegrateVelocityField(0.5,0);
        def_minv =IntegrateVelocityField(0.5,1);
    }
    
private:
    std::vector<CurrentFieldType::Pointer> velocity_field;
public:
    int NTimePoints{11};
};




#endif
