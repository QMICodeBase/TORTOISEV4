#ifndef _QUADRATICTRANSFORMIMAGE_CXX
#define _QUADRATICTRANSFORMIMAGE_CXX

#include "quadratic_transform_image.h"


CUDAIMAGE::Pointer QuadraticTransformImageC(CUDAIMAGE::Pointer main_image, TransformType::Pointer tp,CUDAIMAGE::Pointer target_img)
{
    cudaPitchedPtr d_output={0};
    cudaExtent extent =  make_cudaExtent(sizeof(float)*target_img->sz.x,target_img->sz.y,target_img->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);


    TransformType::MatrixType mat= tp->GetMatrix();    
    
    float mat_arr[9]={mat(0,0),mat(0,1),mat(0,2),mat(1,0),mat(1,1),mat(1,2),mat(2,0),mat(2,1),mat(2,2)};
    TransformType::ParametersType params = tp->GetParameters();

    float params_arr[TransformType::NQUADPARAMS];
    for(int p=0;p<TransformType::NQUADPARAMS;p++)
        params_arr[p]=params[p];


    float main_img_dir[9]={main_image->dir(0,0),main_image->dir(0,1),main_image->dir(0,2),main_image->dir(1,0),main_image->dir(1,1),main_image->dir(1,2),main_image->dir(2,0),main_image->dir(2,1),main_image->dir(2,2)};
    float target_img_dir[9]={target_img->dir(0,0),target_img->dir(0,1),target_img->dir(0,2),target_img->dir(1,0),target_img->dir(1,1),target_img->dir(1,2),target_img->dir(2,0),target_img->dir(2,1),target_img->dir(2,2)};

    QuadraticTransformImage_cuda(main_image->GetTexture(),
                                 main_image->sz,  main_image->spc, main_image->orig,main_img_dir,
                                 target_img->sz,  target_img->spc, target_img->orig,target_img_dir,
                                 mat_arr,
                                 params_arr,
                                 d_output );

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=target_img->sz;
    output->dir=target_img->dir;
    output->orig=target_img->orig;
    output->spc=target_img->spc;
    output->components_per_voxel= target_img->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;

}

#endif
