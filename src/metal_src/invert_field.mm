#ifndef _MTL_INVERTFIELD_CXX
#define _MTL_INVERTFIELD_CXX

#include "invert_field.h"
#include "compose_fields.h"
#include "reductions.h"
#include "metal_context.h"
#include "invert_field.metal.h"
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace
{
struct InvertParams
{
    int32_t sz[4];
    float   spc[4];
    float   ctl[4];      // epsilon, max_error_norm
};
}

CUDAIMAGE::Pointer InvertField(CUDAIMAGE::Pointer field, CUDAIMAGE::Pointer initial_estimate)
{
    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz   = field->sz;
    output->dir  = field->dir;
    output->orig = field->orig;
    output->spc  = field->spc;
    output->components_per_voxel = field->components_per_voxel;
    output->Allocate();          // zero-filled, matching cudaMemset3D

    // The reference seeds from an initial estimate when one is supplied.
    if(initial_estimate)
    {
        // Device-to-device, matching the reference's cudaMemcpy3D with
        // cudaMemcpyDeviceToDevice (cuda_image_utilities.cxx:356-361). This was a
        // Download+Upload pair - the last host round trip in the backend.
        // Sized from `output` (i.e. `field`) as the reference's extent is, not from
        // initial_estimate, so the two cannot disagree if the sizes ever differ.
        mtlctx::CopyBuffer(output->getFloatdata().buf,
                            initial_estimate->getFloatdata().buf, output->Bytes());
    }

    CUDAIMAGE::Pointer scale = CUDAIMAGE::New();
    scale->sz   = field->sz;
    scale->dir  = field->dir;
    scale->orig = field->orig;
    scale->spc  = field->spc;
    scale->components_per_voxel = 1;
    scale->Allocate();

    // Convergence controls, verbatim from InvertField_cuda.
    const float max_tol  = 0.0005f;
    const float mean_tol = 0.00005f;
    const int   Niter    = 200;
    const size_t npix    = field->NumVoxels();

    float max_error  = 1E10f;
    float mean_error = 1E10f;
    int iteration = 0;

    InvertParams p{};
    p.sz[0] = field->sz.x; p.sz[1] = field->sz.y; p.sz[2] = field->sz.z;
    p.sz[3] = field->components_per_voxel;
    p.spc[0] = field->spc.x; p.spc[1] = field->spc.y; p.spc[2] = field->spc.z;

    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    const uint32_t gx = (field->sz.x + wgx - 1) / wgx;
    const uint32_t gy = (field->sz.y + wgy - 1) / wgy;
    const uint32_t gz = (field->sz.z + wgz - 1) / wgz;

    while(iteration++ < Niter && max_error > max_tol && mean_error > mean_tol)
    {
        CUDAIMAGE::Pointer composed = ComposeFields(output, field);

        mtlctx::ComputePipeline p1 =
            mtlctx::Pipeline("invert_field", kinvert_fieldMSL, "local_norm_and_negate",
                              wgx, wgy, wgz);
        mtlctx::Buffer params = mtlctx::CreateUniform(&p, sizeof(p));
        // this entry point does not touch `outp`, so binding 2 must be omitted
        mtlctx::DispatchAt(p1, {{0u, composed->getFloatdata().buf},
                                 {1u, scale->getFloatdata().buf},
                                 {3u, params}}, gx, gy, gz);

        max_error  = Reduce(scale->getFloatdata().buf, npix, ReduceOp::Max);
        mean_error = Reduce(scale->getFloatdata().buf, npix, ReduceOp::Sum) / (float)npix;
        if(std::getenv("TORTOISE_METAL_DEBUG"))
        {
            const float cmax = Reduce(composed->getFloatdata().buf, npix,
                                      ReduceOp::MaxFieldNorm, field->spc);
            const float fmax = Reduce(field->getFloatdata().buf, npix,
                                      ReduceOp::MaxFieldNorm, field->spc);
            const float omax = Reduce(output->getFloatdata().buf, npix,
                                      ReduceOp::MaxFieldNorm, field->spc);
            std::fprintf(stderr,
                "  iter %3d  max=%.6g mean=%.6g | |composed|=%.6g |field|=%.6g |out|=%.6g\n",
                iteration, max_error, mean_error, cmax, fmax, omax);
        }

        float eps = 0.5f;
        if(iteration == 1)
            eps = 0.75f;

        p.ctl[0] = eps;
        p.ctl[1] = max_error;
        mtlctx::Buffer params2 = mtlctx::CreateUniform(&p, sizeof(p));
        mtlctx::ComputePipeline p2 =
            mtlctx::Pipeline("invert_field", kinvert_fieldMSL, "update_invert_field",
                              wgx, wgy, wgz);
        mtlctx::Dispatch(p2, {composed->getFloatdata().buf, scale->getFloatdata().buf,
                               output->getFloatdata().buf, params2}, gx, gy, gz);
    }
    return output;
}

#endif
