#ifndef _MTL_WARPIMAGE_CXX
#define _MTL_WARPIMAGE_CXX

#include "warp_image.h"
#include "metal_context.h"
#include "warp_image.metal.h"
#include <cstdlib>

struct WarpParams
{
    int32_t sz[4];
    float   res[4];
    float   dir[3][4];
};

CUDAIMAGE::Pointer WarpImage(CUDAIMAGE::Pointer main_image, CUDAIMAGE::Pointer field_image)
{
    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz   = main_image->sz;
    output->dir  = main_image->dir;
    output->orig = main_image->orig;
    output->spc  = main_image->spc;
    output->components_per_voxel = main_image->components_per_voxel;
    output->Allocate();

    WarpParams p{};
    p.sz[0] = main_image->sz.x; p.sz[1] = main_image->sz.y; p.sz[2] = main_image->sz.z;
    p.res[0] = main_image->spc.x; p.res[1] = main_image->spc.y; p.res[2] = main_image->spc.z;
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            p.dir[r][c] = (float)main_image->dir(r, c);

    mtlctx::Buffer params = mtlctx::CreateUniform(&p, sizeof(p));

    // Optional emulation of CUDA's 1.8 fixed-point texture filter weights, for
    // A/B validation against the hardware sampler (CLAUDE.md 5.2).
    const bool quantise = std::getenv("TORTOISE_METAL_CUDA_TEXFILTER") != nullptr;
    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    mtlctx::ComputePipeline pipe =
        mtlctx::Pipeline("warp_image", kwarp_imageMSL, "main_warp", wgx, wgy, wgz,
                          {{"TEXFILTER_QUANTISE", quantise ? 1.0 : 0.0}});
    mtlctx::Dispatch(pipe,
                      {main_image->getFloatdata().buf, field_image->getFloatdata().buf,
                       output->getFloatdata().buf, params},
                      (main_image->sz.x + wgx - 1) / wgx,
                      (main_image->sz.y + wgy - 1) / wgy,
                      (main_image->sz.z + wgz - 1) / wgz);
    return output;
}

#endif
