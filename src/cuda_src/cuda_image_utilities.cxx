#ifndef _CUDAIMAGEUTILITIES_CXX
#define _CUDAIMAGEUTILITIES_CXX

#include "cuda_image_utilities.h"



void AddToUpdateField(CUDAIMAGE::Pointer updateField,CUDAIMAGE::Pointer  updateField_temp,float weight,bool normalize)
{
    AddToUpdateField_cuda(updateField->getFloatdata(), updateField_temp->getFloatdata(),weight, updateField->sz,updateField->components_per_voxel ,normalize);
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

 CUDAIMAGE::Pointer NegateField(CUDAIMAGE::Pointer field)
 {
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(field->components_per_voxel*sizeof(float)*field->sz.x,field->sz.y,field->sz.z);
    cudaMalloc3D(&d_output, extent);


        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = field->getFloatdata();
        copyParams.dstPtr = d_output;
        copyParams.extent   = extent;
        copyParams.kind     = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParams);


        NegateField_cuda(d_output,  field->sz);



    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=field->sz;
    output->dir=field->dir;
    output->orig=field->orig;
    output->spc=field->spc;
    output->components_per_voxel= field->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;
}



 CUDAIMAGE::Pointer AddImages(CUDAIMAGE::Pointer im1, CUDAIMAGE::Pointer im2)
 {
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(im1->components_per_voxel*sizeof(float)*im1->sz.x,im1->sz.y,im1->sz.z);
    cudaMalloc3D(&d_output, extent);

    AddImages_cuda(im1->getFloatdata(),im2->getFloatdata(), d_output,  im1->sz,im1->components_per_voxel);

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=im1->sz;
    output->dir=im1->dir;
    output->orig=im1->orig;
    output->spc=im1->spc;
    output->components_per_voxel= im1->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;
}


 CUDAIMAGE::Pointer MultiplyImage(CUDAIMAGE::Pointer im1, float factor)
 {
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(im1->components_per_voxel*sizeof(float)*im1->sz.x,im1->sz.y,im1->sz.z);
    cudaMalloc3D(&d_output, extent);

    MultiplyImage_cuda(im1->getFloatdata(),factor, d_output,  im1->sz,im1->components_per_voxel);

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=im1->sz;
    output->dir=im1->dir;
    output->orig=im1->orig;
    output->spc=im1->spc;
    output->components_per_voxel= im1->components_per_voxel;
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



std::vector<CUDAIMAGE::Pointer> ComputeImageGradientImg(CUDAIMAGE::Pointer img)
 {
    cudaPitchedPtr d_output_x={0};
    cudaPitchedPtr d_output_y={0};
    cudaPitchedPtr d_output_z={0};

    cudaExtent extent =  make_cudaExtent(1*sizeof(float)*img->sz.x,img->sz.y,img->sz.z);
    cudaMalloc3D(&d_output_x, extent);
    cudaMemset3D(d_output_x,0,extent);
    cudaMalloc3D(&d_output_y, extent);
    cudaMemset3D(d_output_y,0,extent);
    cudaMalloc3D(&d_output_z, extent);
    cudaMemset3D(d_output_z,0,extent);


    ComputeImageGradient_cuda(img->getFloatdata(), img->sz,img->spc,
                              img->dir(0,0),img->dir(0,1),img->dir(0,2),img->dir(1,0),img->dir(1,1),img->dir(1,2),img->dir(2,0),img->dir(2,1),img->dir(2,2),
                              d_output_x,d_output_y,d_output_z);

    CUDAIMAGE::Pointer outputx = CUDAIMAGE::New();
    outputx->sz=img->sz;
    outputx->dir=img->dir;
    outputx->orig=img->orig;
    outputx->spc=img->spc;
    outputx->components_per_voxel= 1;
    outputx->SetFloatDataPointer( d_output_x);

    CUDAIMAGE::Pointer outputy = CUDAIMAGE::New();
    outputy->sz=img->sz;
    outputy->dir=img->dir;
    outputy->orig=img->orig;
    outputy->spc=img->spc;
    outputy->components_per_voxel= 1;
    outputy->SetFloatDataPointer( d_output_y);

    CUDAIMAGE::Pointer outputz = CUDAIMAGE::New();
    outputz->sz=img->sz;
    outputz->dir=img->dir;
    outputz->orig=img->orig;
    outputz->spc=img->spc;
    outputz->components_per_voxel= 1;
    outputz->SetFloatDataPointer( d_output_z);

    std::vector<CUDAIMAGE::Pointer> output;
    output.push_back(outputx);
    output.push_back(outputy);
    output.push_back(outputz);
    return output;

}



void IntegrateVelocityFieldGPU(std::vector<CUDAIMAGE::Pointer> velocity_field, float lowt, float hight, CUDAIMAGE::Pointer output_field)
{
    int NT=velocity_field.size();


    cudaPitchedPtr *cuda_vfield2= new cudaPitchedPtr[NT];
    for(int t=0;t<NT;t++)
        cuda_vfield2[t]=velocity_field[t]->getFloatdata();




    cudaPitchedPtr *cuda_vfield;
    cudaMalloc((void**) &cuda_vfield, NT*sizeof(cudaPitchedPtr));
    cudaMemcpy(cuda_vfield, cuda_vfield2, NT*sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);



    IntegrateVelocityField_cuda(cuda_vfield,
                                output_field->getFloatdata(),
                                lowt,hight,NT,
                                output_field->sz,
                                output_field->spc,
                                output_field->dir(0,0),output_field->dir(0,1),output_field->dir(0,2),output_field->dir(1,0),output_field->dir(1,1),output_field->dir(1,2),output_field->dir(2,0),output_field->dir(2,1),output_field->dir(2,2),
                                output_field->orig);

    delete[] cuda_vfield2;
    cudaFree(cuda_vfield);

}







#endif
