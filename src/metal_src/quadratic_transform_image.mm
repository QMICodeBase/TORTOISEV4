#ifndef _MTL_QUADTRANSFORM_CXX
#define _MTL_QUADTRANSFORM_CXX

#include "quadratic_transform_image.h"
#include "metal_context.h"
#include "quadratic_transform_image.metal.h"
#include <cmath>
#include <cstdlib>

struct QuadParams
{
    int32_t tsz[4];
    int32_t isz[4];
    float   smat[3][4];
    float   smat_inv[3][4];
    float   rot[3][4];
    float   par[6][4];
    int32_t flags[4];      // phase, do_cubic
};

CUDAIMAGE::Pointer QuadraticTransformImageC(CUDAIMAGE::Pointer main_image, TransformType::Pointer tp,
                                            CUDAIMAGE::Pointer target_img)
{
    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz   = target_img->sz;
    output->dir  = target_img->dir;
    output->orig = target_img->orig;
    output->spc  = target_img->spc;
    output->components_per_voxel = target_img->components_per_voxel;
    output->Allocate();

    TransformType::MatrixType mat = tp->GetMatrix();
    TransformType::ParametersType params = tp->GetParameters();

    QuadParams p{};
    p.tsz[0] = target_img->sz.x; p.tsz[1] = target_img->sz.y; p.tsz[2] = target_img->sz.z;
    p.isz[0] = main_image->sz.x; p.isz[1] = main_image->sz.y; p.isz[2] = main_image->sz.z;

    // Same construction as QuadraticTransformImage_cuda: target index -> world,
    // and world -> source index (including the origin shift folded into col 3).
    // The reference narrows the direction matrix to float[9] BEFORE multiplying
    // (quadratic_transform_image.cxx:26-27 then .cu:101-123), so every product and
    // quotient below is float*float / float/float. Multiplying in double and
    // narrowing afterwards gives a different result and can flip a voxel across
    // the domain guard or a floor() step.
    float tdir[9], idir[9];
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
        {
            tdir[3 * r + c] = (float)target_img->dir(r, c);
            idir[3 * r + c] = (float)main_image->dir(r, c);
        }

    p.smat[0][0] = tdir[0] * target_img->spc.x;
    p.smat[0][1] = tdir[1] * target_img->spc.y;
    p.smat[0][2] = tdir[2] * target_img->spc.z;
    p.smat[1][0] = tdir[3] * target_img->spc.x;
    p.smat[1][1] = tdir[4] * target_img->spc.y;
    p.smat[1][2] = tdir[5] * target_img->spc.z;
    p.smat[2][0] = tdir[6] * target_img->spc.x;
    p.smat[2][1] = tdir[7] * target_img->spc.y;
    p.smat[2][2] = tdir[8] * target_img->spc.z;
    p.smat[0][3] = target_img->orig.x;
    p.smat[1][3] = target_img->orig.y;
    p.smat[2][3] = target_img->orig.z;

    p.smat_inv[0][0] = idir[0] / main_image->spc.x;
    p.smat_inv[0][1] = idir[3] / main_image->spc.x;
    p.smat_inv[0][2] = idir[6] / main_image->spc.x;
    p.smat_inv[1][0] = idir[1] / main_image->spc.y;
    p.smat_inv[1][1] = idir[4] / main_image->spc.y;
    p.smat_inv[1][2] = idir[7] / main_image->spc.y;
    p.smat_inv[2][0] = idir[2] / main_image->spc.z;
    p.smat_inv[2][1] = idir[5] / main_image->spc.z;
    p.smat_inv[2][2] = idir[8] / main_image->spc.z;
    p.smat_inv[0][3] = -(p.smat_inv[0][0]*main_image->orig.x + p.smat_inv[0][1]*main_image->orig.y
                       + p.smat_inv[0][2]*main_image->orig.z);
    p.smat_inv[1][3] = -(p.smat_inv[1][0]*main_image->orig.x + p.smat_inv[1][1]*main_image->orig.y
                       + p.smat_inv[1][2]*main_image->orig.z);
    p.smat_inv[2][3] = -(p.smat_inv[2][0]*main_image->orig.x + p.smat_inv[2][1]*main_image->orig.y
                       + p.smat_inv[2][2]*main_image->orig.z);

    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            p.rot[r][c] = (float)mat(r, c);

    for(int i = 0; i < TransformType::NQUADPARAMS; i++)
        p.par[i / 4][i % 4] = (float)params[i];

    // Phase axis and cubic-term activation, decided exactly as the CUDA host does.
    // Compared as floats, matching the reference's params_arr[] copy.
    const float p6 = (float)params[6], p7 = (float)params[7], p8 = (float)params[8];
    int phase = 0;
    if(std::fabs(p7) > std::fabs(p6) && std::fabs(p7) > std::fabs(p8))
        phase = 1;
    if(std::fabs(p8) > std::fabs(p6) && std::fabs(p8) > std::fabs(p7))
        phase = 2;
    int do_cubic = 0;
    // The reference tests the FLOAT copy against 1e-10, not the double against 0
    // (quadratic_transform_image.cu:150-156); a parameter of magnitude <= 1e-10
    // would otherwise enable the cubic term here and not in CUDA.
    //
    // The literal must stay DOUBLE. `params_arr[p]` is float and fabs() promotes
    // it exactly to double, so the reference compares against 1E-10. Writing
    // 1E-10f here compares against float(1E-10) = 1.0000000134e-10 instead, and a
    // parameter landing in that window would flip the branch - a different
    // transform, not an ulp. This mirror-image of the trap the comment above
    // describes was present until 2026-08-20.
    for(int q = 14; q <= 20; q++)
        if(std::fabs((float)params[q]) > 1E-10)
            do_cubic = 1;
    p.flags[0] = phase;
    p.flags[1] = do_cubic;

    mtlctx::Buffer params_buf = mtlctx::CreateUniform(&p, sizeof(p));

    const bool quantise = std::getenv("TORTOISE_METAL_CUDA_TEXFILTER") != nullptr;
    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    mtlctx::ComputePipeline pipe =
        mtlctx::Pipeline("quadratic_transform_image", kquadratic_transform_imageMSL, "main_quadratic",
                          wgx, wgy, wgz, {{"TEXFILTER_QUANTISE", quantise ? 1.0 : 0.0}});
    mtlctx::Dispatch(pipe,
                      {main_image->getFloatdata().buf, output->getFloatdata().buf, params_buf},
                      (target_img->sz.x + wgx - 1) / wgx,
                      (target_img->sz.y + wgy - 1) / wgy,
                      (target_img->sz.z + wgz - 1) / wgz);
    return output;
}

#endif
