#ifndef _WGPU_RESAMPLEIMAGE_CXX
#define _WGPU_RESAMPLEIMAGE_CXX

#include "resample_image.h"
#include "webgpu_context.h"
#include "resample_image.wgsl.h"

// Uniform-buffer layout mirroring `struct Params` in resample_image.wgsl.
// Everything is vec4-sized so the std140 rules cannot bite.
struct ResampleParams
{
    int32_t dsz[4];      // data x,y,z, Ncomponents
    int32_t vsz[4];
    float   dspc[4];
    float   vspc[4];
    float   dorig[4];
    float   vorig[4];
    float   ddir[3][4];
    float   vdir[3][4];
};

CUDAIMAGE::Pointer ResampleImage(CUDAIMAGE::Pointer main_field, CUDAIMAGE::Pointer virtual_img)
{
    if(main_field == nullptr)
        return nullptr;

    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz   = virtual_img->sz;
    output->dir  = virtual_img->dir;
    output->orig = virtual_img->orig;
    output->spc  = virtual_img->spc;
    output->components_per_voxel = main_field->components_per_voxel;
    output->Allocate();

    ResampleParams p{};
    p.dsz[0] = main_field->sz.x; p.dsz[1] = main_field->sz.y; p.dsz[2] = main_field->sz.z;
    p.dsz[3] = main_field->components_per_voxel;
    p.vsz[0] = virtual_img->sz.x; p.vsz[1] = virtual_img->sz.y; p.vsz[2] = virtual_img->sz.z;
    p.dspc[0]  = main_field->spc.x;  p.dspc[1]  = main_field->spc.y;  p.dspc[2]  = main_field->spc.z;
    p.vspc[0]  = virtual_img->spc.x; p.vspc[1]  = virtual_img->spc.y; p.vspc[2]  = virtual_img->spc.z;
    p.dorig[0] = main_field->orig.x; p.dorig[1] = main_field->orig.y; p.dorig[2] = main_field->orig.z;
    p.vorig[0] = virtual_img->orig.x;p.vorig[1] = virtual_img->orig.y;p.vorig[2] = virtual_img->orig.z;
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
        {
            p.ddir[r][c] = (float)main_field->dir(r, c);
            p.vdir[r][c] = (float)virtual_img->dir(r, c);
        }

    wgpu::Buffer params = wgpuctx::CreateUniform(&p, sizeof(p));

    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    wgpu::ComputePipeline pipe =
        wgpuctx::Pipeline("resample_image", kresample_imageWGSL, "main", wgx, wgy, wgz);
    wgpuctx::Dispatch(pipe,
                      {main_field->getFloatdata().buf, output->getFloatdata().buf, params},
                      (virtual_img->sz.x + wgx - 1) / wgx,
                      (virtual_img->sz.y + wgy - 1) / wgy,
                      (virtual_img->sz.z + wgz - 1) / wgz);
    return output;
}

#endif
