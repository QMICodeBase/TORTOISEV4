#ifndef _DRTAMASDIFFEO_H
#define _DRTAMASDIFFEO_H


#include "drtamas_structs.h"
#include "defines.h"
#include "TORTOISE.h"

#include "DRTAMAS_parser.h"


#include "cuda_image.h"
#include "../cuda_src/cuda_image_utilities.h"


class DRTAMASRigid_Bulk
{    
    public:


    using CurrentFieldType = CUDAIMAGE;
    using CurrentImageType = CUDAIMAGE;

    using RigidTransformType = TORTOISE::RigidTransformType;


    DRTAMASRigid_Bulk()
    {
        this->stream= &(std::cout);
    }

    ~DRTAMASRigid_Bulk(){};

    void SetFixedTensor(DTMatrixImageType::Pointer tens)
    {
        this->fixed_tensor = CUDAIMAGE::New();
        this->fixed_tensor->SetTImageFromITK(tens);
    }

    void SetMovingTensor(DTMatrixImageType::Pointer tens)
    {
        this->moving_tensor = CUDAIMAGE::New();
        this->moving_tensor->SetTImageFromITK(tens);
    }

    void SetStagesFromExternal(std::vector<DRTAMASStageSettings> st){stages=st;}
    void SetParser(DRTAMAS_PARSER *prs){parser=prs;};

    RigidTransformType::Pointer GetRigidTrans(){return my_rigid_trans;}


private:            //Subfunctions the main processing functions use
    void SetImagesForMetrics();
    void SetUpStages();
    void SetDefaultStages();
    

public:                    //Main processing functions
    void Process();

  //  std::string GetRegistrationMethodType(){return parser->getRegistrationMethodType();}



private:                    //Main processing functions





private:          //class member variables

    CUDAIMAGE::Pointer fixed_tensor{nullptr};
    CUDAIMAGE::Pointer moving_tensor{nullptr};

    std::vector<DRTAMASStageSettings> stages;




    std::ostream  *stream;    
    DRTAMAS_PARSER *parser{nullptr};

    RigidTransformType::Pointer my_rigid_trans{nullptr};

};



#endif

