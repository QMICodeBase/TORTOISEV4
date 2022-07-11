#ifndef _CUDAIMAGEUTILITIES_CXX
#define _CUDAIMAGEUTILITIES_CXX

#include "cuda_image_utilities.h"



void AddToUpdateField(CUDAIMAGE::Pointer updateField,CUDAIMAGE::Pointer  updateField_temp,float weight)
{
    AddToUpdateField_cuda(updateField->getFloatdata(), updateField_temp->getFloatdata(),weight, updateField->sz,updateField->components_per_voxel );
}



void ScaleUpdateField(CUDAIMAGE::Pointer  field,float scale_factor)
{
    ScaleUpdateField_cuda(field->getFloatdata(), field->sz, field->spc, scale_factor );
}



void RestrictPhase(CUDAIMAGE::Pointer  field, float3 phase)
{
    RestrictPhase_cuda(field->getFloatdata(),  field->sz,phase);
}


void ContrainDefFields(CUDAIMAGE::Pointer  ufield, CUDAIMAGE::Pointer  dfield)
{
    ContrainDefFields_cuda(ufield->getFloatdata(),dfield->getFloatdata(),  ufield->sz);
}




CUDAIMAGE::Pointer ComposeFields(CUDAIMAGE::Pointer main_field, CUDAIMAGE::Pointer update_field)
{
    cudaPitchedPtr d_output={0};
    cudaExtent extent =  make_cudaExtent(main_field->components_per_voxel*sizeof(float)*main_field->sz.x,main_field->sz.y,main_field->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);


    ComposeFields_cuda(main_field->getFloatdata(),update_field->getFloatdata(),
                   main_field->sz,
                   main_field->spc,
                   main_field->dir(0,0),main_field->dir(0,1),main_field->dir(0,2),main_field->dir(1,0),main_field->dir(1,1),main_field->dir(1,2),main_field->dir(2,0),main_field->dir(2,1),main_field->dir(2,2),
                   main_field->orig,
                   d_output
                   );

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=main_field->sz;
    output->dir=main_field->dir;
    output->orig=main_field->orig;
    output->spc=main_field->spc;
    output->components_per_voxel= main_field->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;


}



 CUDAIMAGE::Pointer InvertField(CUDAIMAGE::Pointer field,CUDAIMAGE::Pointer initial_estimate)
{
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(field->components_per_voxel*sizeof(float)*field->sz.x,field->sz.y,field->sz.z);
    cudaMalloc3D(&d_output, extent);

    if(!initial_estimate)
    {
        cudaMemset3D(d_output,0,extent);
    }
    else
    {
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = initial_estimate->getFloatdata();
        copyParams.dstPtr = d_output;
        copyParams.extent   = extent;
        copyParams.kind     = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParams);
    }

    InvertField_cuda(field->getFloatdata(),
                   field->sz,
                   field->spc,
                   field->dir(0,0),field->dir(0,1),field->dir(0,2),field->dir(1,0),field->dir(1,1),field->dir(1,2),field->dir(2,0),field->dir(2,1),field->dir(2,2),
                   field->orig,
                   d_output
                   );



    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=field->sz;
    output->dir=field->dir;
    output->orig=field->orig;
    output->spc=field->spc;
    output->components_per_voxel= field->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;

}



 CUDAIMAGE::Pointer  PreprocessImage(CUDAIMAGE::Pointer img,float low_val, float up_val)
 {
     cudaPitchedPtr d_output={0};
     cudaExtent extent =  make_cudaExtent(sizeof(float)*img->sz.x,img->sz.y,img->sz.z);
     cudaMalloc3D(&d_output, extent);
     cudaMemset3D(d_output,0,extent);

     PreprocessImage_cuda(img->getFloatdata(),
                          img->sz,
                          low_val, up_val,
                          d_output);

     CUDAIMAGE::Pointer output = CUDAIMAGE::New();
     output->sz=img->sz;
     output->dir=img->dir;
     output->orig=img->orig;
     output->spc=img->spc;
     output->components_per_voxel= 1;
     output->SetFloatDataPointer( d_output);
     return output;
 }





#endif
