#ifndef _RUNDRTAMASSTAGE_H
#define _RUNDRTAMASSTAGE_H

#include "drtamas_structs.h"

#include "DRTAMAS_Diffeo.h"
#include "TORTOISE.h"

#ifdef USECUDA
#include "cuda_image.h"
#endif

#include "defines.h"
#include "itkTimeProbe.h"

class DRTAMASStage
{
public:

    using CurrentFieldType = DRTAMAS_Diffeo::CurrentFieldType;
    using CurrentImageType = DRTAMAS_Diffeo::CurrentImageType;    

    DRTAMASStage(){};
    ~DRTAMASStage(){};

    DRTAMASStage(DRTAMASStageSettings *my_settings)
    {
        settings=my_settings;
        this->def_FINV= settings->init_finv;
        this->def_MINV= settings->init_minv;

        float tot_weight=0;
        for(int m=0;m<settings->metrics.size();m++)
            tot_weight+=settings->metrics[m].weight;
        for(int m=0;m<settings->metrics.size();m++)
            settings->metrics[m].weight/=tot_weight;

        this->stream= &(std::cout);

    }




    void RunDRTAMASStage();
    void PreprocessImagesAndFields();


protected:
    void CreateVirtualImage();


    DRTAMASStageSettings *settings;
    CurrentImageType::Pointer virtual_img{nullptr};

    itk::TimeProbe                    m_clock;
    std::ostream  *stream;

private:


    CurrentFieldType::Pointer def_F{nullptr};
    CurrentFieldType::Pointer def_M{nullptr};
    CurrentFieldType::Pointer def_FINV{nullptr};
    CurrentFieldType::Pointer def_MINV{nullptr};


};




#endif
