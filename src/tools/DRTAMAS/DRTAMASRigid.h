#ifndef _DRTAMASRigid_h
#define _DRTAMASRigid_h


#include "DRTAMAS_parser.h"
#include "defines.h"

#include "TORTOISE.h"





class DRTAMASRigid
{
public:
    DRTAMASRigid(){};
    ~DRTAMASRigid(){};

    using RigidTransformType = TORTOISE::RigidTransformType;

        
public:
    void SetParser(DRTAMAS_PARSER* p){parser=p;};
    void Process2();

    
private:
    void Step0_ReadImages();    
    void Step1_RigidRegistration();
    void Step2_TransformAndWriteAffineImage();




private:

    
private:       
    DRTAMAS_PARSER* parser{nullptr};

    DTMatrixImageType::Pointer fixed_tensor{nullptr};
    DTMatrixImageType::Pointer moving_tensor{nullptr};    

    RigidTransformType::Pointer my_rigid_trans{nullptr};



};





#endif
