#ifndef _COMPUTEMETRICMSJAC_CXX
#define _COMPUTEMETRICMSJAC_CXX


#include "compute_metric.h"




float ComputeMetric_CCJacSSingle(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img, const CUDAIMAGE::Pointer str_img,
                                 std::vector<CUDAIMAGE::Pointer>  up_grad_img, std::vector<CUDAIMAGE::Pointer>  down_grad_img,
                                 const CUDAIMAGE::Pointer def_FINV,
                                 CUDAIMAGE::Pointer &updateFieldFINV,
                                 float3 phase_vector,itk::GaussianOperator<float,3> &oper)
{
    updateFieldFINV = CUDAIMAGE::New();
    updateFieldFINV->sz=up_img->sz;
    updateFieldFINV->dir=up_img->dir;
    updateFieldFINV->orig=up_img->orig;
    updateFieldFINV->spc=up_img->spc;
    updateFieldFINV->components_per_voxel= 3;
    updateFieldFINV->Allocate();


    auto aa= oper.GetBufferReference();
    int kernel_sz= aa.size();

    float *h_kernel = new float[kernel_sz];
    for(int m=0;m<kernel_sz;m++)
          h_kernel[m]= aa[m];



    float metric_value;

    ComputeMetric_CCJacSSingle_cuda(up_img->getFloatdata(), down_img->getFloatdata(), str_img->getFloatdata(),
                             up_grad_img[0]->GetTexture(), up_grad_img[1]->GetTexture(), up_grad_img[2]->GetTexture(),
                             down_grad_img[0]->GetTexture(), down_grad_img[1]->GetTexture(), down_grad_img[2]->GetTexture(),
                             up_img->sz, up_img->spc,
                             up_img->dir(0,0),up_img->dir(0,1),up_img->dir(0,2),up_img->dir(1,0),up_img->dir(1,1),up_img->dir(1,2),up_img->dir(2,0),up_img->dir(2,1),up_img->dir(2,2),
                             def_FINV->getFloatdata(),
                             updateFieldFINV->getFloatdata(),
                             phase_vector, kernel_sz,h_kernel, metric_value
                             );

    delete[] h_kernel;

    return metric_value;

}


float ComputeMetric_CCJacS(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img, const CUDAIMAGE::Pointer str_img,
                          const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV   ,
                          CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                          float3 phase_vector,itk::GaussianOperator<float,3> &oper)
{
    updateFieldF = CUDAIMAGE::New();
    updateFieldF->sz=up_img->sz;
    updateFieldF->dir=up_img->dir;
    updateFieldF->orig=up_img->orig;
    updateFieldF->spc=up_img->spc;
    updateFieldF->components_per_voxel= 3;
    updateFieldF->Allocate();

    updateFieldM = CUDAIMAGE::New();
    updateFieldM->sz=up_img->sz;
    updateFieldM->dir=up_img->dir;
    updateFieldM->orig=up_img->orig;
    updateFieldM->spc=up_img->spc;
    updateFieldM->components_per_voxel= 3;
    updateFieldM->Allocate();

    auto aa= oper.GetBufferReference();
    int kernel_sz= aa.size();

    float *h_kernel = new float[kernel_sz];
    for(int m=0;m<kernel_sz;m++)
        h_kernel[m]= aa[m];



    float metric_value;

    ComputeMetric_CCJacS_cuda(up_img->getFloatdata(), down_img->getFloatdata(), str_img->getFloatdata(),
                              up_img->sz, up_img->spc,
                              up_img->dir(0,0),up_img->dir(0,1),up_img->dir(0,2),up_img->dir(1,0),up_img->dir(1,1),up_img->dir(1,2),up_img->dir(2,0),up_img->dir(2,1),up_img->dir(2,2),
                              def_FINV->getFloatdata(), def_MINV->getFloatdata(),
                              updateFieldF->getFloatdata(), updateFieldM->getFloatdata(),
                              phase_vector, kernel_sz,h_kernel, metric_value
                              );

    delete[] h_kernel;

    return metric_value;

}


float ComputeMetric_MSJacSingle(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                                std::vector<CUDAIMAGE::Pointer>  up_grad_img,  std::vector<CUDAIMAGE::Pointer> down_grad_img,
                          const CUDAIMAGE::Pointer def_FINV,
                          CUDAIMAGE::Pointer &updateFieldFINV,
                          float3 phase_vector,itk::GaussianOperator<float,3> &oper)
{
    updateFieldFINV = CUDAIMAGE::New();
    updateFieldFINV->sz=up_img->sz;
    updateFieldFINV->dir=up_img->dir;
    updateFieldFINV->orig=up_img->orig;
    updateFieldFINV->spc=up_img->spc;
    updateFieldFINV->components_per_voxel= 3;
    updateFieldFINV->Allocate();


    auto aa= oper.GetBufferReference();
    int kernel_sz= aa.size();

    float *h_kernel = new float[kernel_sz];
    for(int m=0;m<kernel_sz;m++)
          h_kernel[m]= aa[m];



    float metric_value;

    ComputeMetric_MSJacSingle_cuda(up_img->getFloatdata(), down_img->getFloatdata(),
                             up_grad_img[0]->GetTexture(), up_grad_img[1]->GetTexture(), up_grad_img[2]->GetTexture(),
                             down_grad_img[0]->GetTexture(), down_grad_img[1]->GetTexture(), down_grad_img[2]->GetTexture(),
                             up_img->sz, up_img->spc,
                             up_img->dir(0,0),up_img->dir(0,1),up_img->dir(0,2),up_img->dir(1,0),up_img->dir(1,1),up_img->dir(1,2),up_img->dir(2,0),up_img->dir(2,1),up_img->dir(2,2),
                             def_FINV->getFloatdata(),
                             updateFieldFINV->getFloatdata(),
                             phase_vector, kernel_sz,h_kernel, metric_value
                             );

    delete[] h_kernel;

    return metric_value;

}



float ComputeMetric_MSJac(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img, 
                          const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV   ,
                          CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                          float3 phase_vector, itk::GaussianOperator<float,3> &oper)
{
    updateFieldF = CUDAIMAGE::New();
    updateFieldF->sz=up_img->sz;
    updateFieldF->dir=up_img->dir;
    updateFieldF->orig=up_img->orig;
    updateFieldF->spc=up_img->spc;
    updateFieldF->components_per_voxel= 3;
    updateFieldF->Allocate();
    
    updateFieldM = CUDAIMAGE::New();
    updateFieldM->sz=up_img->sz;
    updateFieldM->dir=up_img->dir;
    updateFieldM->orig=up_img->orig;
    updateFieldM->spc=up_img->spc;
    updateFieldM->components_per_voxel= 3;
    updateFieldM->Allocate();

    auto aa= oper.GetBufferReference();
    int kernel_sz= aa.size();

    float *h_kernel = new float[kernel_sz];
    for(int m=0;m<kernel_sz;m++)
          h_kernel[m]= aa[m];


    
    float metric_value;
    
    ComputeMetric_MSJac_cuda(up_img->getFloatdata(), down_img->getFloatdata(),
                             up_img->sz, up_img->spc,
                             up_img->dir(0,0),up_img->dir(0,1),up_img->dir(0,2),up_img->dir(1,0),up_img->dir(1,1),up_img->dir(1,2),up_img->dir(2,0),up_img->dir(2,1),up_img->dir(2,2),
                             def_FINV->getFloatdata(), def_MINV->getFloatdata(),
                             updateFieldF->getFloatdata(), updateFieldM->getFloatdata(),
                             phase_vector, kernel_sz,h_kernel, metric_value
                             );

    delete[] h_kernel;

    return metric_value;
}








float ComputeMetric_CCSK(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img, const CUDAIMAGE::Pointer str_img,
                         CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,float t)

{
    updateFieldF = CUDAIMAGE::New();
    updateFieldF->sz=up_img->sz;
    updateFieldF->dir=up_img->dir;
    updateFieldF->orig=up_img->orig;
    updateFieldF->spc=up_img->spc;
    updateFieldF->components_per_voxel= 3;
    updateFieldF->Allocate();

    updateFieldM = CUDAIMAGE::New();
    updateFieldM->sz=up_img->sz;
    updateFieldM->dir=up_img->dir;
    updateFieldM->orig=up_img->orig;
    updateFieldM->spc=up_img->spc;
    updateFieldM->components_per_voxel= 3;
    updateFieldM->Allocate();


    float metric_value;

    ComputeMetric_CCSK_cuda(up_img->getFloatdata(), down_img->getFloatdata(), str_img->getFloatdata(),
     up_img->sz, up_img->spc,
     up_img->dir(0,0),up_img->dir(0,1),up_img->dir(0,2),up_img->dir(1,0),up_img->dir(1,1),up_img->dir(1,2),up_img->dir(2,0),up_img->dir(2,1),up_img->dir(2,2),
     updateFieldF->getFloatdata(), updateFieldM->getFloatdata(),
     metric_value,t
     );

    return metric_value;

}




float ComputeMetric_CC(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                         CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM)

{
    updateFieldF = CUDAIMAGE::New();
    updateFieldF->sz=up_img->sz;
    updateFieldF->dir=up_img->dir;
    updateFieldF->orig=up_img->orig;
    updateFieldF->spc=up_img->spc;
    updateFieldF->components_per_voxel= 3;
    updateFieldF->Allocate();

    updateFieldM = CUDAIMAGE::New();
    updateFieldM->sz=up_img->sz;
    updateFieldM->dir=up_img->dir;
    updateFieldM->orig=up_img->orig;
    updateFieldM->spc=up_img->spc;
    updateFieldM->components_per_voxel= 3;
    updateFieldM->Allocate();


    float metric_value;

    ComputeMetric_CC_cuda(up_img->getFloatdata(), down_img->getFloatdata(),
     up_img->sz, up_img->spc,
     up_img->dir(0,0),up_img->dir(0,1),up_img->dir(0,2),up_img->dir(1,0),up_img->dir(1,1),up_img->dir(1,2),up_img->dir(2,0),up_img->dir(2,1),up_img->dir(2,2),
     updateFieldF->getFloatdata(), updateFieldM->getFloatdata(),
     metric_value
     );

    return metric_value;

}


CUDAIMAGE::Pointer ComputeDetImgMain(CUDAIMAGE::Pointer img, CUDAIMAGE::Pointer field, float3 phase_vector)
{
    cudaPitchedPtr d_output={0};
    cudaExtent extent =  make_cudaExtent(sizeof(float)*img->sz.x,img->sz.y,img->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);

    ComputeDetImg_cuda(img->getFloatdata(), field->getFloatdata(),
                       img->sz, img->spc,
                       img->dir(0,0),img->dir(0,1),img->dir(0,2),img->dir(1,0),img->dir(1,1),img->dir(1,2),img->dir(2,0),img->dir(2,1),img->dir(2,2),
                       phase_vector,
                       d_output
                       );

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
