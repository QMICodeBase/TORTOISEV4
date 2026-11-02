#ifndef _MTL_COMPOSEFIELDS_CXX
#define _MTL_COMPOSEFIELDS_CXX

#include "compose_fields.h"
#include "metal_context.h"
#include "compose_fields.metal.h"

struct ComposeParams
{
    int32_t sz[4];
    float   spc[4];
    float   dir[3][4];
};

CUDAIMAGE::Pointer ComposeFields(CUDAIMAGE::Pointer main_field, CUDAIMAGE::Pointer update_field)
{
    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz   = main_field->sz;
    output->dir  = main_field->dir;
    output->orig = main_field->orig;
    output->spc  = main_field->spc;
    output->components_per_voxel = main_field->components_per_voxel;
    output->Allocate();

    ComposeParams p{};
    p.sz[0] = main_field->sz.x; p.sz[1] = main_field->sz.y; p.sz[2] = main_field->sz.z;
    p.sz[3] = main_field->components_per_voxel;
    p.spc[0] = main_field->spc.x; p.spc[1] = main_field->spc.y; p.spc[2] = main_field->spc.z;
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            p.dir[r][c] = (float)main_field->dir(r, c);

    mtlctx::Buffer params = mtlctx::CreateUniform(&p, sizeof(p));
    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    mtlctx::ComputePipeline pipe =
        mtlctx::Pipeline("compose_fields", kcompose_fieldsMSL, "main_compose", wgx, wgy, wgz);
    mtlctx::Dispatch(pipe,
                      {main_field->getFloatdata().buf, update_field->getFloatdata().buf,
                       output->getFloatdata().buf, params},
                      (main_field->sz.x + wgx - 1) / wgx,
                      (main_field->sz.y + wgy - 1) / wgy,
                      (main_field->sz.z + wgz - 1) / wgz);
    return output;
}

#endif
