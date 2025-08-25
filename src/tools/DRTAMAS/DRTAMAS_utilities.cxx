#ifndef _DRTAMAS_UTILITIES_CXX
#define _DRTAMAS_UTILITIES_CXX



#include "DRTAMAS_utilities.h"


CUDAIMAGE::Pointer ComputeTRMapC(CUDAIMAGE::Pointer tensor_img)
{
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(1*sizeof(float)*tensor_img->sz.x,tensor_img->sz.y,tensor_img->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);


    ComputeTRMapC_cuda(tensor_img->getFloatdata(), d_output,  tensor_img->sz);


    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=tensor_img->sz;
    output->dir=tensor_img->dir;
    output->orig=tensor_img->orig;
    output->spc=tensor_img->spc;
    output->components_per_voxel= 1;
    output->SetFloatDataPointer( d_output);
    return output;
}

CUDAIMAGE::Pointer ComputeFAMapC(CUDAIMAGE::Pointer tensor_img)
{
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(1*sizeof(float)*tensor_img->sz.x,tensor_img->sz.y,tensor_img->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);


    ComputeFAMapC_cuda(tensor_img->getFloatdata(), d_output,  tensor_img->sz);


    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=tensor_img->sz;
    output->dir=tensor_img->dir;
    output->orig=tensor_img->orig;
    output->spc=tensor_img->spc;
    output->components_per_voxel= 1;
    output->SetFloatDataPointer( d_output);
    return output;
}




std::vector<CUDAIMAGE::Pointer>  SplitImageComponents(CUDAIMAGE::Pointer img)
{
    int Ncomps= img->components_per_voxel;
    cudaExtent extent =  make_cudaExtent(1*sizeof(float)*img->sz.x,img->sz.y,img->sz.z);


    cudaPitchedPtr *d_output={0};
    d_output = new cudaPitchedPtr[Ncomps];
    for(int v=0; v<Ncomps;v++)
    {
        cudaMalloc3D(&(d_output[v]), extent);
        cudaMemset3D(d_output[v],0,extent);
    }

    cudaPitchedPtr *d_outputc={0};
    cudaMalloc((void**) &d_outputc, Ncomps*sizeof(cudaPitchedPtr));
    cudaMemcpy(d_outputc, d_output, Ncomps*sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);


    SplitImageComponents_cuda(img->getFloatdata(),
                              d_outputc,
                              img->sz,
                              Ncomps);


    std::vector<CUDAIMAGE::Pointer> outputa;
    for(int v=0;v<Ncomps;v++)
    {
        CUDAIMAGE::Pointer output = CUDAIMAGE::New();
        output->sz=img->sz;
        output->dir=img->dir;
        output->orig=img->orig;
        output->spc=img->spc;
        output->components_per_voxel= 1;
        output->SetFloatDataPointer( d_output[v]);
        outputa.push_back(output);
    }
    cudaFree(d_outputc);

    return outputa;
}



CUDAIMAGE::Pointer  CombineImageComponents(std::vector<CUDAIMAGE::Pointer> img)
{
     int Ncomps= img.size();

     cudaPitchedPtr d_output={0};
     cudaExtent extent =  make_cudaExtent(Ncomps*sizeof(float)*img[0]->sz.x,img[0]->sz.y,img[0]->sz.z);
     cudaMalloc3D(&d_output, extent);
     cudaMemset3D(d_output,0,extent);

     cudaPitchedPtr *d_input={0};
     d_input = new cudaPitchedPtr[Ncomps];
     for(int v=0; v<Ncomps;v++)
     {
         d_input[v]= img[v]->getFloatdata();
     }

     cudaPitchedPtr *d_inputc;
     cudaMalloc((void**) &d_inputc, Ncomps*sizeof(cudaPitchedPtr));
     cudaMemcpy(d_inputc, d_input, Ncomps*sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);




     CombineImageComponents_cuda(d_inputc,
                                 d_output,
                                 img[0]->sz,
                                 Ncomps);


     CUDAIMAGE::Pointer output = CUDAIMAGE::New();
     output->sz=img[0]->sz;
     output->dir=img[0]->dir;
     output->orig=img[0]->orig;
     output->spc=img[0]->spc;
     output->components_per_voxel= Ncomps;
     output->SetFloatDataPointer( d_output);

     delete[] d_input;
     cudaFree(d_inputc);
     return output;
}


CUDAIMAGE::Pointer  LogTensor(CUDAIMAGE::Pointer tensor_img)
{
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(tensor_img->components_per_voxel*sizeof(float)*tensor_img->sz.x,tensor_img->sz.y,tensor_img->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);


    LogTensor_cuda(tensor_img->getFloatdata(), d_output,  tensor_img->sz);


    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=tensor_img->sz;
    output->dir=tensor_img->dir;
    output->orig=tensor_img->orig;
    output->spc=tensor_img->spc;
    output->components_per_voxel= tensor_img->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;
}

CUDAIMAGE::Pointer  ExpTensor(CUDAIMAGE::Pointer tensor_img)
{
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(tensor_img->components_per_voxel*sizeof(float)*tensor_img->sz.x,tensor_img->sz.y,tensor_img->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);


    ExpTensor_cuda(tensor_img->getFloatdata(), d_output,  tensor_img->sz);


    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=tensor_img->sz;
    output->dir=tensor_img->dir;
    output->orig=tensor_img->orig;
    output->spc=tensor_img->spc;
    output->components_per_voxel= tensor_img->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;
}


CUDAIMAGE::Pointer  RotateTensors(CUDAIMAGE::Pointer tensor_img,TransformType::Pointer rigid_trans)
{
    cudaPitchedPtr d_output={0};

    cudaExtent extent =  make_cudaExtent(tensor_img->components_per_voxel*sizeof(float)*tensor_img->sz.x,tensor_img->sz.y,tensor_img->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);
    
    
    TransformType::MatrixType mat= rigid_trans->GetMatrix();        
    float mat_arr[9]={mat(0,0),mat(0,1),mat(0,2),mat(1,0),mat(1,1),mat(1,2),mat(2,0),mat(2,1),mat(2,2)};


    RotateTensors_cuda(tensor_img->getFloatdata(), d_output,  tensor_img->sz,  mat_arr);

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz=tensor_img->sz;
    output->dir=tensor_img->dir;
    output->orig=tensor_img->orig;
    output->spc=tensor_img->spc;
    output->components_per_voxel= tensor_img->components_per_voxel;
    output->SetFloatDataPointer( d_output);
    return output;
}


#endif
