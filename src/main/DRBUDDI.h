#ifndef _DRBUDDI_H
#define _DRBUDDI_H


#include "DRBUDDIBase.h"

class DRBUDDI : public DRBUDDIBase
{    
    using SuperClass=DRBUDDIBase;

    using OkanQuadraticTransformType= SuperClass::OkanQuadraticTransformType;
    using DisplacementFieldTransformType= SuperClass::DisplacementFieldTransformType;
    using DisplacementFieldType=SuperClass::DisplacementFieldType;
    using RigidTransformType = SuperClass::RigidTransformType;
    using CompositeTransformType= SuperClass::CompositeTransformType;

    using RGBPixelType=SuperClass::RGBPixelType;
    using RGBImageType=SuperClass::RGBImageType;

public:

    DRBUDDI(std::string uname,std::string dname,std::vector<std::string> str_names,json mjson);
    ~DRBUDDI(){};  

private:            //Subfunctions the main processing functions use        

    RigidTransformType::Pointer RigidDiffeoRigidRegisterB0DownToB0Up(ImageType3D::Pointer up_image, ImageType3D::Pointer b0_down_image, std::string mtype, ImageType3D::Pointer & initial_corrected_b0);
    std::vector<DisplacementFieldType::Pointer> DRBUDDI_Initial_Register_Up_Down(ImageType3D::Pointer b0_up_img,ImageType3D::Pointer blip_down_img, std::string phase,bool small);



public:                    //Main processing functions
    void Process();

private:                    //Main processing functions
    void Step0_CreateImages();
    void Step1_RigidRegistration();
    void Step2_DiffeoRegistration();
    void Step3_WriteOutput();

    DisplacementFieldType::Pointer CompositeToDispField(CompositeTransformType::Pointer comp_trans, ImageType3D::Pointer ref_img);
    ImageType3D::Pointer PreprocessImage(  ImageType3D::ConstPointer  inputImage,
                                              ImageType3D::PixelType lowerScaleValue,
                                              ImageType3D::PixelType upperScaleValue,
                                              float winsorizeLowerQuantile, float winsorizeUpperQuantile,
                                                  ImageType3D::ConstPointer histogramMatchSourceImage=nullptr );



private:
    std::string down_nii_name;


    ImageType3D::Pointer FA_up_quad{nullptr}, FA_down_quad{nullptr};
    ImageType3D::Pointer b0_down_quad{nullptr};

    ImageType3D::Pointer FA_up{nullptr}, FA_down{nullptr};
    ImageType3D::Pointer b0_down{nullptr};


    DisplacementFieldType::Pointer def_MINV{nullptr};
};



#endif

