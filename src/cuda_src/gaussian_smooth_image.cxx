#ifndef _GAUSSIANSMOOTHIMAGE_CXX
#define _GAUSSIANSMOOTHIMAGE_CXX

#include "itkGaussianOperator.h"
#include "gaussian_smooth_image.h"



 CUDAIMAGE::Pointer GaussianSmoothImage(CUDAIMAGE::Pointer main_image, float std)
{
    if(std==0)
    {
         return main_image;
    }

     if(main_image==nullptr)
         return nullptr;


    cudaPitchedPtr d_output={0};
    cudaExtent extent =  make_cudaExtent(main_image->components_per_voxel*sizeof(float)*main_image->sz.x,main_image->sz.y,main_image->sz.z);
    cudaMalloc3D(&d_output, extent);
    cudaMemset3D(d_output,0,extent);


    itk::GaussianOperator<float,3 > oper;

    float max_error=0.01;
    if(main_image->components_per_voxel==3)
        max_error=0.001;

    int radius;

    oper.SetDirection(0);
    oper.SetVariance(std);
    oper.SetMaximumKernelWidth(31);
    oper.SetMaximumError(max_error);
    oper.CreateDirectional();


    radius=oper.GetRadius(0);

    auto aa= oper.GetBufferReference();
    int kernel_sz= aa.size();

    float *h_kernel = new float[kernel_sz];
    for(int m=0;m<kernel_sz;m++)
    {
        h_kernel[m]= aa[m];
    }

    GaussianSmoothImage_cuda(main_image->getFloatdata(),
                   main_image->sz,                                     
                   main_image->components_per_voxel,
                   kernel_sz,
                   h_kernel,
                   d_output
                   );

    delete[] h_kernel;


    if(main_image->components_per_voxel==3)
    {
        float weight1=1;
        if( std < 0.5 )
        {
            weight1 = 1.0 - 1.0 * ( std / 0.5 );
        }
        float weight2 = 1.0 - weight1;

        AdjustFieldBoundary(main_image->getFloatdata(),d_output,main_image->sz,  weight1,weight2);
    }


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
