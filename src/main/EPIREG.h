#ifndef _EPIREG_H
#define _EPIREG_H


#include "DRBUDDIBase.h"

class EPIREG : public DRBUDDIBase
{    
    using SuperClass=DRBUDDIBase;

    using OkanQuadraticTransformType= SuperClass::OkanQuadraticTransformType;
    using DisplacementFieldTransformType= SuperClass::DisplacementFieldTransformType;
    using DisplacementFieldType=SuperClass::DisplacementFieldType;
    using RigidTransformType = SuperClass::RigidTransformType;

    using RGBPixelType=SuperClass::RGBPixelType;
    using RGBImageType=SuperClass::RGBImageType;

public:

    EPIREG(std::string uname,std::vector<std::string> str_names,json mjson);
    ~EPIREG(){};  

private:            //Subfunctions the main processing functions use        


public:                    //Main processing functions
    void Process();

private:                    //Main processing functions
    void Step0_CreateImages();
    void Step1_RigidRegistration();
    void Step2_DiffeoRegistration();
    void Step3_WriteOutput();


};



#endif

