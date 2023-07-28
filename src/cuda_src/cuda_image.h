#ifndef _CUDAIMAGE_H
#define _CUDAIMAGE_H

#include <stdio.h>
#include <iostream>
#include "itkImage.h"
#include "itkDisplacementFieldTransform.h"
#include <memory.h>

#include "cuda_utils.h"



class CUDAIMAGE
{

public:
    using DataType=float;
    using ImageType3D=itk::Image<DataType,3>;
    using ImageType4D=itk::Image<DataType,4>;
    typedef itk::DisplacementFieldTransform<double,3> DisplacementTransformType;
    typedef itk::DisplacementFieldTransform<float,3> DisplacementTransformTypeFloat;
    typedef DisplacementTransformType::DisplacementFieldType DisplacementFieldType;
    typedef DisplacementTransformTypeFloat::DisplacementFieldType DisplacementFieldTypeFloat;

    using  InternalMatrixType=vnl_matrix_fixed< double, 3, 3 >;
    using DTMatrixImageType = itk::Image<InternalMatrixType,3>;

    using TensorVectorType = itk::Vector<float,6>;
    using TensorVectorImageType = itk::Image<TensorVectorType,3>;




    using Self = CUDAIMAGE;
    using Pointer = std::shared_ptr<Self>;

    static Pointer   New();

public:   
    CUDAIMAGE(){PitchedFloatData.ptr=nullptr; CudaArraydata=nullptr;};
    ~CUDAIMAGE();
    



    void DuplicateFromCUDAImage(CUDAIMAGE::Pointer cp_img);
    void SetImageFromITK(ImageType3D::Pointer itk_image, bool create_texture=false);
    void SetImageFromITK(DisplacementFieldType::Pointer itk_field);    
    void SetTImageFromITK(DTMatrixImageType::Pointer tensor_img);

    ImageType3D::Pointer CudaImageToITKImage();
    TensorVectorImageType::Pointer CudaImageToITKImage4D();
    DisplacementFieldType::Pointer CudaImageToITKField();


    cudaPitchedPtr getFloatdata(){return PitchedFloatData;};

    void SetFloatDataPointer( cudaPitchedPtr data){PitchedFloatData=data;};
    ImageType3D::DirectionType GetDirection(){return dir;};

    void Allocate();

    

    cudaTextureObject_t GetTexture(){return texobj;};
    void CreateTexture();
 
    ImageType3D::DirectionType dir;
    float3 orig;
    float3 spc;
    int3 sz;

    int components_per_voxel{1};

 
 private:
    cudaPitchedPtr PitchedFloatData{nullptr};

    cudaArray *CudaArraydata{nullptr};
    cudaTextureObject_t texobj;


};


#endif
