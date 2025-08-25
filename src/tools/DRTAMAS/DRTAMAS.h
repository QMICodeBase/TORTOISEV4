#ifndef _DRTAMAS_h
#define _DRTAMAS_h


#include "DRTAMAS_parser.h"
#include "defines.h"

#include "TORTOISE.h"





class DRTAMAS
{
public:
    DRTAMAS(  ){};
    ~DRTAMAS(){};

    using RigidTransformType = TORTOISE::RigidTransformType;
    using AffineTransformType=  TORTOISE::AffineTransformType;
    using DisplacementFieldType = TORTOISE::DisplacementFieldType;
    using DisplacementFieldTransformType = TORTOISE::DisplacementFieldTransformType;


    

    
public:
    void SetParser(DRTAMAS_PARSER* p){parser=p;}
    void Process();

    
private:

    ImageType3D::Pointer PreprocessImage(ImageType3D::Pointer inputImage,
                                                  ImageType3D::PixelType lowerScaleValue,
                                                  ImageType3D::PixelType upperScaleValue,
                                                  float winsorizeLowerQuantile, float winsorizeUpperQuantile,
                                                  ImageType3D::Pointer histogramMatchSourceImage=nullptr );

    void Step0_ReadImages();
    RigidTransformType::Pointer Step00_RigidRegistration(ImageType3D::Pointer fixed_img_orig,ImageType3D::Pointer moving_img_orig);
    void Step0_AffineRegistration();
    void Step0_TransformAndWriteAffineImage();
    void Step1_DiffeoRegistration();
    void Step2_WriteImages();

  //  vnl_matrix_fixed<double,3,3> ComputeJacobian(DisplacementFieldType::Pointer field,DisplacementFieldType::IndexType ind3 );



private:
   // DTMatrixImageType::Pointer  ReadAndOrientTensor(std::string fname);
    
    
private:       
    DRTAMAS_PARSER* parser{nullptr};

    DTMatrixImageType::Pointer fixed_tensor{nullptr};
    DTMatrixImageType::Pointer moving_tensor{nullptr};
    DTMatrixImageType::Pointer moving_tensor_aff{nullptr};
    std::vector<ImageType3D::Pointer> fixed_structurals;
    std::vector<ImageType3D::Pointer> moving_structurals;

    AffineTransformType::Pointer my_affine_trans{nullptr};
    DisplacementFieldType::Pointer def{nullptr};


};





#endif
