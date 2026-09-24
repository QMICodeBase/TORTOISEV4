#ifndef _CUDAIMAGEUTILITIES_CXX
#define _CUDAIMAGEUTILITIES_CXX

#include "cuda_image_utilities.h"
#include "gpu_capture.h"




CUDAIMAGE::Pointer AnisotropicSmoothField(CUDAIMAGE::Pointer field, CUDAIMAGE::Pointer TR_img, CUDAIMAGE::Pointer FA_img )
{
    cudaPitchedPtr d_output={0};
    cudaExtent extent =  make_cudaExtent(field->components_per_voxel*sizeof(float)*field->sz.x,field->sz.y,field->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset(d_output.ptr,0,d_output.pitch*extent.height*extent.depth);



    AnisotropicSmoothField_cuda(field->getFloatdata(),
                       TR_img->getFloatdata(),FA_img->getFloatdata(),
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


void AddToUpdateField(CUDAIMAGE::Pointer updateField,CUDAIMAGE::Pointer  updateField_temp,float weight,bool normalize)
{
    gpucap::Rec cap("AddToUpdateField");
    if(cap.on())
    {
    cap.param("weight",weight).param("normalize",(double)normalize);
    cap.in("updateField",updateField).in("updateField_temp",updateField_temp);
    }

    AddToUpdateField_cuda(updateField->getFloatdata(), updateField_temp->getFloatdata(),weight, updateField->sz,updateField->components_per_voxel ,normalize);
    if(cap.on()) {
    cap.out("updateField",updateField).save();
    }
}



float DotProduct(CUDAIMAGE::Pointer  field1,CUDAIMAGE::Pointer  field2)
{
    float nrm1= TotalNorm_cuda(field1->getFloatdata(), field1->sz);
    float nrm2= TotalNorm_cuda(field2->getFloatdata(), field2->sz);

    auto ff1= MultiplyImage(field1,1./nrm1);
    auto ff2= MultiplyImage(field2,1./nrm2);

    auto mlt= MultiplyImages(ff1,ff2);

    auto mlt_itk = mlt->CudaImageToITKImage();

    using ImageType3D=itk::Image<float,3>;
    float sm=0;
    itk::ImageRegionConstIterator<ImageType3D> it(mlt_itk,mlt_itk->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        sm+=it.Get();
    }

    return sm;
}


void ScaleUpdateField(CUDAIMAGE::Pointer  field,float scale_factor)
{
    gpucap::Rec cap("ScaleUpdateField");
    if(cap.on())
    {
    cap.param("scale_factor",scale_factor).in("field",field);
    }

    ScaleUpdateField_cuda(field->getFloatdata(), field->sz, field->spc, scale_factor );
    if(cap.on()) {
    cap.out("field",field).save();
    }
}



void RestrictPhase(CUDAIMAGE::Pointer  field, float3 phase)
{
    gpucap::Rec cap("RestrictPhase");
    if(cap.on())
    {
    cap.param("phase",phase).in("field",field);
    }

    RestrictPhase_cuda(field->getFloatdata(),  field->sz,phase);
    if(cap.on()) {
    cap.out("field",field).save();
    }
}


void ContrainDefFields(CUDAIMAGE::Pointer  ufield, CUDAIMAGE::Pointer  dfield)
{
    gpucap::Rec cap("ContrainDefFields");
    if(cap.on())
    {
    cap.in("ufield",ufield).in("dfield",dfield);
    }

    ContrainDefFields_cuda(ufield->getFloatdata(),dfield->getFloatdata(),  ufield->sz);
    if(cap.on()) {
    cap.out("ufield",ufield).out("dfield",dfield).save();
    }
}



std::vector<CUDAIMAGE::Pointer> ContrainVelocityFields(std::vector<CUDAIMAGE::Pointer>  vfield)
{
    int NT=vfield.size();


    cudaPitchedPtr *cuda_vfield2= new cudaPitchedPtr[NT];
    for(int t=0;t<NT;t++)
        cuda_vfield2[t]=vfield[t]->getFloatdata();

    cudaPitchedPtr *cuda_vfield;
    cudaMalloc((void**) &cuda_vfield, NT*sizeof(cudaPitchedPtr));
    cudaMemcpy(cuda_vfield, cuda_vfield2, NT*sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);


    std::vector<CUDAIMAGE::Pointer> new_vfield;
    new_vfield.resize(NT);
    for(int T=0;T<NT;T++)
    {
        CUDAIMAGE::Pointer field= CUDAIMAGE::New();
        field->orig=  vfield[0]->orig;
        field->dir=  vfield[0]->dir;
        field->spc=  vfield[0]->spc;
        field->sz=  vfield[0]->sz;
        field->components_per_voxel= 3;
        field->Allocate();

        new_vfield[T]=field;
    }

    cudaPitchedPtr *cuda_newvfield2= new cudaPitchedPtr[NT];
    for(int t=0;t<NT;t++)
        cuda_newvfield2[t]=new_vfield[t]->getFloatdata();

    cudaPitchedPtr *cuda_newvfield;
    cudaMalloc((void**) &cuda_newvfield, NT*sizeof(cudaPitchedPtr));
    cudaMemcpy(cuda_newvfield, cuda_newvfield2, NT*sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);



    CUDAIMAGE::Pointer dfield=CUDAIMAGE::New();
    dfield->sz =vfield[0]->sz;
    dfield->dir = vfield[0]->dir;
    dfield->orig = vfield[0]->orig;
    dfield->spc = vfield[0]->spc;
    dfield->components_per_voxel = 3;
    dfield->Allocate();


    IntegrateVelocityFieldGPU(vfield, 0.5,0,dfield);


    CUDAIMAGE::Pointer output_field = vfield[0];
    ContrainVelocityFields_cuda(cuda_vfield,
                                cuda_newvfield,
                                dfield->getFloatdata(),
                                NT,
                                output_field->sz,
                                output_field->spc,
                                output_field->dir(0,0),output_field->dir(0,1),output_field->dir(0,2),output_field->dir(1,0),output_field->dir(1,1),output_field->dir(1,2),output_field->dir(2,0),output_field->dir(2,1),output_field->dir(2,2),
                                output_field->orig);

    delete[] cuda_vfield2;
    cudaFree(cuda_vfield);

    delete[] cuda_newvfield2;
    cudaFree(cuda_newvfield);

    return new_vfield;
}






CUDAIMAGE::Pointer ComposeFields(CUDAIMAGE::Pointer main_field, CUDAIMAGE::Pointer update_field)
{
    cudaPitchedPtr d_output={0};
    cudaExtent extent =  make_cudaExtent(main_field->components_per_voxel*sizeof(float)*main_field->sz.x,main_field->sz.y,main_field->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset(d_output.ptr,0,d_output.pitch*extent.height*extent.depth);


    gpucap::Rec cap("ComposeFields");
    if(cap.on())
    {
    cap.in("main_field",main_field).in("update_field",update_field);
    }

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
    if(cap.on()) {
    cap.out("output",output).save();
    }

    return output;


}

 CUDAIMAGE::Pointer NegateField(CUDAIMAGE::Pointer field)
 {
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(field->components_per_voxel*sizeof(float)*field->sz.x,field->sz.y,field->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset(d_output.ptr,0,d_output.pitch*extent.height*extent.depth);   // deterministic row padding (PERF_NOTES 2.3/12)


        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = field->getFloatdata();
        copyParams.dstPtr = d_output;
        copyParams.extent   = extent;
        copyParams.kind     = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParams);


    gpucap::Rec cap("NegateField");
    if(cap.on())
    {
    cap.in("field",field);
    }

        NegateField_cuda(d_output,  field->sz);



    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=field->sz;
    output->dir=field->dir;
    output->orig=field->orig;
    output->spc=field->spc;
    output->components_per_voxel= field->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    if(cap.on()) {
    cap.out("output",output).save();
    }

    return output;
}



 CUDAIMAGE::Pointer AddImages(CUDAIMAGE::Pointer im1, CUDAIMAGE::Pointer im2)
 {
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(im1->components_per_voxel*sizeof(float)*im1->sz.x,im1->sz.y,im1->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset(d_output.ptr,0,d_output.pitch*extent.height*extent.depth);   // deterministic row padding (PERF_NOTES 2.3/12)

    gpucap::Rec cap("AddImages");
    if(cap.on())
    {
    cap.in("im1",im1).in("im2",im2);
    }

    AddImages_cuda(im1->getFloatdata(),im2->getFloatdata(), d_output,  im1->sz,im1->components_per_voxel);

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=im1->sz;
    output->dir=im1->dir;
    output->orig=im1->orig;
    output->spc=im1->spc;
    output->components_per_voxel= im1->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    if(cap.on()) {
    cap.out("output",output).save();
    }

    return output;
}


 CUDAIMAGE::Pointer MultiplyImage(CUDAIMAGE::Pointer im1, float factor)
 {
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(im1->components_per_voxel*sizeof(float)*im1->sz.x,im1->sz.y,im1->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset(d_output.ptr,0,d_output.pitch*extent.height*extent.depth);   // deterministic row padding (PERF_NOTES 2.3/12)

    gpucap::Rec cap("MultiplyImage");
    if(cap.on())
    {
    cap.param("factor",factor).in("im1",im1);
    }

    MultiplyImage_cuda(im1->getFloatdata(),factor, d_output,  im1->sz,im1->components_per_voxel);

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=im1->sz;
    output->dir=im1->dir;
    output->orig=im1->orig;
    output->spc=im1->spc;
    output->components_per_voxel= im1->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    if(cap.on()) {
    cap.out("output",output).save();
    }

    return output;
}


 CUDAIMAGE::Pointer InvertField(CUDAIMAGE::Pointer field,CUDAIMAGE::Pointer initial_estimate)
{
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(field->components_per_voxel*sizeof(float)*field->sz.x,field->sz.y,field->sz.z);
    cudaMalloc3D(&d_output, extent);

    if(!initial_estimate)
    {
        cudaMemset(d_output.ptr,0,d_output.pitch*extent.height*extent.depth);
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

    gpucap::Rec cap("InvertField");
    if(cap.on())
    {
    if(initial_estimate) cap.in("initial_estimate",initial_estimate);
    cap.in("field",field);
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
    if(cap.on()) {
    cap.out("output",output).save();
    }

    return output;

}



 CUDAIMAGE::Pointer  PreprocessImage(CUDAIMAGE::Pointer img,float low_val, float up_val)
 {
     cudaPitchedPtr d_output={0};
     cudaExtent extent =  make_cudaExtent(sizeof(float)*img->sz.x,img->sz.y,img->sz.z);
     cudaMalloc3D(&d_output, extent);
     cudaMemset(d_output.ptr,0,d_output.pitch*extent.height*extent.depth);

    gpucap::Rec cap("PreprocessImage");
    if(cap.on())
    {
    cap.param("low_val",low_val).param("up_val",up_val).in("img",img);
    }

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
    if(cap.on()) {
     cap.out("output",output).save();
    }

     return output;
 }



std::vector<CUDAIMAGE::Pointer> ComputeImageGradientImg(CUDAIMAGE::Pointer img)
 {
    cudaPitchedPtr d_output_x={0};
    cudaPitchedPtr d_output_y={0};
    cudaPitchedPtr d_output_z={0};

    cudaExtent extent =  make_cudaExtent(1*sizeof(float)*img->sz.x,img->sz.y,img->sz.z);
    cudaMalloc3D(&d_output_x, extent);
    cudaMemset(d_output_x.ptr,0,d_output_x.pitch*extent.height*extent.depth);
    cudaMalloc3D(&d_output_y, extent);
    cudaMemset(d_output_y.ptr,0,d_output_y.pitch*extent.height*extent.depth);
    cudaMalloc3D(&d_output_z, extent);
    cudaMemset(d_output_z.ptr,0,d_output_z.pitch*extent.height*extent.depth);


    gpucap::Rec cap("ComputeImageGradientImg");
    if(cap.on())
    {
    cap.in("img",img);
    }

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
    if(cap.on()) {
    cap.out("grad_x",outputx).out("grad_y",outputy).out("grad_z",outputz).save();
    }

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



float  ComputeFieldScale(CUDAIMAGE::Pointer  field)
{
    return ComputeFieldScale_cuda(field->getFloatdata(), field->sz, field->spc );

}

CUDAIMAGE::Pointer DivideImages(CUDAIMAGE::Pointer im1, CUDAIMAGE::Pointer im2)
{
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(im1->components_per_voxel*sizeof(float)*im1->sz.x,im1->sz.y,im1->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset(d_output.ptr,0,d_output.pitch*extent.height*extent.depth);   // deterministic row padding (PERF_NOTES 2.3/12)

    DivideImages_cuda(im1->getFloatdata(),im2->getFloatdata(), d_output,  im1->sz,im1->components_per_voxel);

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=im1->sz;
    output->dir=im1->dir;
    output->orig=im1->orig;
    output->spc=im1->spc;
    output->components_per_voxel= im1->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;

}


CUDAIMAGE::Pointer MultiplyImages(CUDAIMAGE::Pointer im1, CUDAIMAGE::Pointer im2)
{
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(im1->components_per_voxel*sizeof(float)*im1->sz.x,im1->sz.y,im1->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset(d_output.ptr,0,d_output.pitch*extent.height*extent.depth);   // deterministic row padding (PERF_NOTES 2.3/12)

    gpucap::Rec cap("MultiplyImages");
    if(cap.on())
    {
    cap.in("im1",im1).in("im2",im2);
    }

    MultiplyImages_cuda(im1->getFloatdata(),im2->getFloatdata(), d_output,  im1->sz,im1->components_per_voxel);

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=im1->sz;
    output->dir=im1->dir;
    output->orig=im1->orig;
    output->spc=im1->spc;
    output->components_per_voxel= im1->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    if(cap.on()) {
    cap.out("output",output).save();
    }

    return output;

}



#endif
