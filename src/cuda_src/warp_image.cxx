#ifndef _WARPIMAGE_CXX
#define _WARPIMAGE_CXX

#include "warp_image.h"



CUDAIMAGE::Pointer WarpImage(CUDAIMAGE::Pointer main_image, CUDAIMAGE::Pointer field_image)
{

    cudaPitchedPtr d_output={0};
    cudaExtent extent =  make_cudaExtent(sizeof(float)*main_image->sz.x,main_image->sz.y,main_image->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);


    WarpImage_cuda(main_image->GetTexture(),
                   main_image->sz,
                   main_image->spc,
                   main_image->dir(0,0),main_image->dir(0,1),main_image->dir(0,2),main_image->dir(1,0),main_image->dir(1,1),main_image->dir(1,2),main_image->dir(2,0),main_image->dir(2,1),main_image->dir(2,2),
                   field_image->getFloatdata(),
                   d_output
                   );


    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=main_image->sz;
    output->dir=main_image->dir;
    output->orig=main_image->orig;
    output->spc=main_image->spc;
    output->components_per_voxel= main_image->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;

}

#endif
