#ifndef _DIFFUSIONMODEL_H
#define _DIFFUSIONMODEL_H


#include "defines.h"
#include "TORTOISE.h"


template< typename T >
class DiffusionModel
{
public:
    using OutputImageType =  T;


private:


public:
    DiffusionModel(){stream=TORTOISE::stream;}
    ~DiffusionModel(){};

    void SetJson( json njson){my_json=njson;}
    void SetBmatrix( vnl_matrix<double> bmat){Bmatrix=bmat;}
    void SetDWIData(std::vector<ImageType3D::Pointer> ndata){dwi_data=ndata;}
    void SetWeightImage(std::vector<ImageType3D::Pointer> ninc){weight_imgs=ninc;}
    void SetVoxelwiseBmatrix(std::vector<ImageType4D::Pointer> vbmat){voxelwise_Bmatrix=vbmat;}
    void SetMaskImage(ImageType3D::Pointer mi){mask_img=mi;}
    void SetVolIndicesForFitting(std::vector<int> inf){indices_fitting=inf;}
    void SetA0Image(ImageType3D::Pointer ai){A0_img=ai;}
    ImageType3D::Pointer GetA0Image(){return A0_img;}
    typename OutputImageType::Pointer GetOutput(){return output_img;}
    void SetOutput(typename OutputImageType::Pointer oi){output_img=oi;}


    virtual void PerformFitting(){};
    virtual ImageType3D::Pointer SynthesizeDWI(vnl_vector<double> bmat_vec){return nullptr;};





protected:

    json my_json;
    vnl_matrix<double> Bmatrix;

    std::vector<ImageType3D::Pointer> dwi_data;
    std::vector<ImageType3D::Pointer> weight_imgs;
    std::vector<ImageType4D::Pointer> voxelwise_Bmatrix;
    ImageType3D::Pointer mask_img{nullptr};
    std::vector<int> indices_fitting;

    ImageType3D::Pointer A0_img{nullptr};
    typename OutputImageType::Pointer output_img{nullptr};


    TORTOISE::TeeStream *stream;

};



#endif
