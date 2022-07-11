#ifndef _RESAMPLEIMAGE_CXX
#define _RESAMPLEIMAGE_CXX

#include "resample_image.h"



 CUDAIMAGE::Pointer ResampleImage(CUDAIMAGE::Pointer main_field, CUDAIMAGE::Pointer virtual_img)
{

     if(main_field==nullptr)
         return nullptr;

    cudaPitchedPtr d_output={0};
    cudaExtent extent =  make_cudaExtent(main_field->components_per_voxel*sizeof(float)*virtual_img->sz.x,virtual_img->sz.y,virtual_img->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);


    ResampleImage_cuda(main_field->getFloatdata(),
                   main_field->sz,
                   main_field->spc,
                   main_field->dir(0,0),main_field->dir(0,1),main_field->dir(0,2),main_field->dir(1,0),main_field->dir(1,1),main_field->dir(1,2),main_field->dir(2,0),main_field->dir(2,1),main_field->dir(2,2),
                   main_field->orig,
                   virtual_img->sz,
                   virtual_img->spc,
                   virtual_img->dir(0,0),main_field->dir(0,1),main_field->dir(0,2),main_field->dir(1,0),main_field->dir(1,1),main_field->dir(1,2),main_field->dir(2,0),main_field->dir(2,1),main_field->dir(2,2),
                   virtual_img->orig,
                   main_field->components_per_voxel,
                   d_output
                   );

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=virtual_img->sz;
    output->dir=virtual_img->dir;
    output->orig=virtual_img->orig;
    output->spc=virtual_img->spc;
    output->components_per_voxel= main_field->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;

}

#endif
