#ifndef _DRBUDDIBase_H
#define _DRBUDDIBase_H

#include "TORTOISE.h"
#include "TORTOISE_parser.h"

#include <string>

#include "defines.h"


class DRBUDDIBase
{ 

public:   
    using OkanQuadraticTransformType= TORTOISE::OkanQuadraticTransformType;
    using DisplacementFieldTransformType= TORTOISE::DisplacementFieldTransformType;
    using DisplacementFieldType=DisplacementFieldTransformType::DisplacementFieldType;
    using RigidTransformType = TORTOISE::RigidTransformType;

    using RGBPixelType=itk::RGBPixel< unsigned char >;
    using RGBImageType=itk::Image< RGBPixelType, 3 >;



    DRBUDDIBase() {};
    virtual ~DRBUDDIBase(){};

    //void SetParser(TORTOISE_PARSER* p){this->parser=p;}
    void SetParser(DRBUDDI_PARSERBASE* p)
    {
        this->parser=p;
        if(parser->getDRBUDDIOutput()!="")
        {
            this->proc_folder = parser->getDRBUDDIOutput();
            if(!fs::exists(this->proc_folder))
                fs::create_directories(this->proc_folder);
        }
    }

    void SetMaskImg(ImageType3D::Pointer mi){this->main_mask_img=mi;}


protected:            //Subfunctions the main processing functions use
    void CreateCorrectionImage(std::string nii_filename,ImageType3D::Pointer &b0_img, ImageType3D::Pointer &FA_img);



    void CreateBlipUpQuadImage();

    ImageType3D::Pointer JacobianTransformImage(ImageType3D::Pointer img,DisplacementFieldType::Pointer field,ImageType3D::Pointer ref_img);
    InternalMatrixType ComputeJacobianAtIndex(DisplacementFieldType::Pointer disp_field, DisplacementFieldType::IndexType index);


public:                    //Main processing functions
    void Process();

private:                    //Main processing functions
    virtual void Step0_CreateImages() {}
    virtual void Step1_RigidRegistration(){}
    virtual void Step2_DiffeoRegistration() {}
    virtual void Step3_WriteOutput() {}







protected:

    std::string PE_string;    
    json my_json;
    std::string up_nii_name;    
    std::vector<std::string> structural_names;
    std::string proc_folder;


#ifdef DRBUDDIALONE
    std::ostream  *stream;
#else
    TORTOISE::TeeStream *stream;
#endif

    DRBUDDI_PARSERBASE *parser{nullptr};

    ImageType3D::Pointer b0_up_quad{nullptr};
    ImageType3D::Pointer b0_up{nullptr};
    ImageType3D::Pointer main_mask_img{nullptr};

    std::vector<ImageType3D::Pointer> structural_imgs;

    DisplacementFieldType::Pointer def_FINV{nullptr};
};



#endif

