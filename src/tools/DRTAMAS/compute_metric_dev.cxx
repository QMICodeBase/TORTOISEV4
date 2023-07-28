#ifndef _COMPUTEMETRICDEV_CXX
#define _COMPUTEMETRICDEV_CXX


#include "compute_metric_dev.h"
#include "defines.h"

float ComputeMetric_DEV(const CUDAIMAGE::Pointer up_img_or, const CUDAIMAGE::Pointer down_img_or,
                          const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV   ,
                          CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM)

{
    CUDAIMAGE::Pointer up_img= CUDAIMAGE::New();
    up_img->DuplicateFromCUDAImage(up_img_or);
    CUDAIMAGE::Pointer down_img= CUDAIMAGE::New();
    down_img->DuplicateFromCUDAImage(down_img_or);

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


    ComputeDeviatoricTensor_cuda(up_img->getFloatdata(),   up_img->sz);
    ComputeDeviatoricTensor_cuda(down_img->getFloatdata(),   down_img->sz);

/*
    {
        auto aa = up_img->CudaImageToITKImage4D();
        std::string nm="/qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/src/tools/DRTAMAS/test/fdev.nii";
        using WrType= itk::ImageFileWriter<CUDAIMAGE::TensorVectorImageType>;
        WrType::Pointer wr= WrType::New();
        wr->SetFileName(nm);
        wr->SetInput(aa);
        wr->Update();
    }


    {
        auto aa = down_img->CudaImageToITKImage4D();
        std::string nm="/qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/src/tools/DRTAMAS/test/mdev.nii";
        using WrType= itk::ImageFileWriter<CUDAIMAGE::TensorVectorImageType>;
        WrType::Pointer wr= WrType::New();
        wr->SetFileName(nm);
        wr->SetInput(aa);
        wr->Update();
    }
*/

    
    float metric_value;

    ComputeMetric_DEV_cuda(up_img->getFloatdata(), down_img->getFloatdata(),
                             up_img->sz, up_img->spc,
                             up_img->dir(0,0),up_img->dir(0,1),up_img->dir(0,2),up_img->dir(1,0),up_img->dir(1,1),up_img->dir(1,2),up_img->dir(2,0),up_img->dir(2,1),up_img->dir(2,2),
                             def_FINV->getFloatdata(), def_MINV->getFloatdata(),
                             updateFieldF->getFloatdata(), updateFieldM->getFloatdata(),
                             metric_value
                             );


    return metric_value;
}









#endif
