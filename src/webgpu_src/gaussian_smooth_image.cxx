#ifndef _WGPU_GAUSSIANSMOOTH_CXX
#define _WGPU_GAUSSIANSMOOTH_CXX

#include "itkGaussianOperator.h"
#include "gaussian_smooth_image.h"
#include "webgpu_context.h"
#include "gaussian_smooth_image.wgsl.h"
#include <stdexcept>

struct SmoothParams
{
    int32_t sz[4];        // x, y, z, Ncomponents
    int32_t ksz[4];       // kernel_sz
    float   wts[4];       // weight1, weight2 (boundary pass only)
    float   kernel[8][4]; // up to 32 taps
};

CUDAIMAGE::Pointer GaussianSmoothImage(CUDAIMAGE::Pointer main_image, float std)
{
    if(std == 0)
        return main_image;
    if(main_image == nullptr)
        return nullptr;

    // Kernel taps built exactly as the CUDA wrapper does.
    itk::GaussianOperator<float, 3> oper;
    float max_error = 0.01;
    if(main_image->components_per_voxel == 3)
        max_error = 0.001;
    oper.SetDirection(0);
    oper.SetVariance(std);
    oper.SetMaximumKernelWidth(31);
    oper.SetMaximumError(max_error);
    oper.CreateDirectional();

    auto aa = oper.GetBufferReference();
    std::vector<float> taps(aa.size());
    for(size_t m = 0; m < aa.size(); m++)
        taps[m] = aa[m];
    return GaussianSmoothImageWithTaps(main_image, taps, std);
}

CUDAIMAGE::Pointer GaussianSmoothImageWithTaps(CUDAIMAGE::Pointer main_image,
                                               const std::vector<float> &tapv, float std)
{
    if(main_image == nullptr)
        return nullptr;
    const int kernel_sz = (int)tapv.size();

    SmoothParams p{};
    p.sz[0] = main_image->sz.x; p.sz[1] = main_image->sz.y; p.sz[2] = main_image->sz.z;
    p.sz[3] = main_image->components_per_voxel;
    // CUDA's c_Kernel is __constant__ float[31], so cudaMemcpyToSymbol + gpuErrchk
    // aborts above 31. Match that bound rather than the uniform block's 32.
    if(kernel_sz > 31)
        throw std::runtime_error("Gaussian kernel has too many taps for the uniform block");
    p.ksz[0] = kernel_sz;
    for(int m = 0; m < kernel_sz; m++)
        p.kernel[m / 4][m % 4] = tapv[m];

    const size_t bytes = main_image->Bytes();
    wgpu::Buffer buf1 = wgpuctx::CreateStorage(bytes);
    wgpu::Buffer buf2 = wgpuctx::CreateStorage(bytes);

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz   = main_image->sz;
    output->dir  = main_image->dir;
    output->orig = main_image->orig;
    output->spc  = main_image->spc;
    output->components_per_voxel = main_image->components_per_voxel;
    output->Allocate();

    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    const uint32_t gx = (main_image->sz.x + wgx - 1) / wgx;
    const uint32_t gy = (main_image->sz.y + wgy - 1) / wgy;
    const uint32_t gz = (main_image->sz.z + wgz - 1) / wgz;

    // Three separable passes: x -> buf1, y -> buf2, z -> output.
    wgpu::Buffer params = wgpuctx::CreateUniform(&p, sizeof(p));
    struct { uint32_t axis; wgpu::Buffer in, out; } passes[3] = {
        {0u, main_image->getFloatdata().buf, buf1},
        {1u, buf1,                           buf2},
        {2u, buf2,                           output->getFloatdata().buf}};

    for(int s = 0; s < 3; s++)
    {
        wgpu::ComputePipeline pipe =
            wgpuctx::Pipeline("gaussian_smooth_image", kgaussian_smooth_imageWGSL, "main",
                              wgx, wgy, wgz, {{"AXIS", (double)passes[s].axis}});
        wgpuctx::Dispatch(pipe, {passes[s].in, passes[s].out, params}, gx, gy, gz);
    }

    // Vector fields get the boundary treatment the CUDA wrapper applies.
    if(main_image->components_per_voxel == 3)
    {
        float weight1 = 1;
        if(std < 0.5)
            weight1 = 1.0 - 1.0 * (std / 0.5);
        p.wts[0] = weight1;
        p.wts[1] = 1.0 - weight1;
        wgpu::Buffer bparams = wgpuctx::CreateUniform(&p, sizeof(p));
        wgpu::ComputePipeline pipe =
            wgpuctx::Pipeline("gaussian_smooth_image", kgaussian_smooth_imageWGSL,
                              "adjust_boundary", wgx, wgy, wgz, {{"AXIS", 0.0}});
        wgpuctx::Dispatch(pipe,
                          {main_image->getFloatdata().buf, output->getFloatdata().buf, bparams},
                          gx, gy, gz);
    }
    return output;
}

#endif
