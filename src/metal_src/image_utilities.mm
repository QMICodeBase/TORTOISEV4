#ifndef _MTL_IMAGE_UTILITIES_CXX
#define _MTL_IMAGE_UTILITIES_CXX

#include "image_utilities.h"
#include "metal_context.h"
#include "image_utilities.metal.h"
#include "reductions.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace
{
struct UtilParams
{
    int32_t sz[4];
    float   fa[4];
    float   fb[4];
};

UtilParams MakeParams(CUDAIMAGE::Pointer img)
{
    UtilParams p{};
    p.sz[0] = img->sz.x; p.sz[1] = img->sz.y; p.sz[2] = img->sz.z;
    p.sz[3] = img->components_per_voxel;
    return p;
}

CUDAIMAGE::Pointer MakeLike(CUDAIMAGE::Pointer src, int ncomp = -1)
{
    CUDAIMAGE::Pointer o = CUDAIMAGE::New();
    o->sz   = src->sz;
    o->dir  = src->dir;
    o->orig = src->orig;
    o->spc  = src->spc;
    o->components_per_voxel = (ncomp < 0) ? src->components_per_voxel : ncomp;
    o->Allocate();
    return o;
}

void Run(const char *entry, const UtilParams &p, CUDAIMAGE::Pointer shape,
         const std::vector<std::pair<uint32_t, mtlctx::Buffer> > &bufs)
{
    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    mtlctx::ComputePipeline pipe =
        mtlctx::Pipeline("image_utilities", kimage_utilitiesMSL, entry, wgx, wgy, wgz);
    std::vector<std::pair<uint32_t, mtlctx::Buffer> > all = bufs;
    all.push_back(std::make_pair(3u, mtlctx::CreateUniform(&p, sizeof(p))));
    mtlctx::DispatchAt(pipe, all,
                        (shape->sz.x + wgx - 1) / wgx,
                        (shape->sz.y + wgy - 1) / wgy,
                        (shape->sz.z + wgz - 1) / wgz);
}
} // namespace

CUDAIMAGE::Pointer AddImages(CUDAIMAGE::Pointer im1, CUDAIMAGE::Pointer im2)
{
    CUDAIMAGE::Pointer o = MakeLike(im1);
    Run("add_images", MakeParams(im1), im1,
        {{0u, im1->getFloatdata().buf}, {1u, im2->getFloatdata().buf},
         {2u, o->getFloatdata().buf}});
    return o;
}

CUDAIMAGE::Pointer MultiplyImages(CUDAIMAGE::Pointer im1, CUDAIMAGE::Pointer im2)
{
    CUDAIMAGE::Pointer o = MakeLike(im1);
    Run("multiply_images", MakeParams(im1), im1,
        {{0u, im1->getFloatdata().buf}, {1u, im2->getFloatdata().buf},
         {2u, o->getFloatdata().buf}});
    return o;
}

CUDAIMAGE::Pointer MultiplyImage(CUDAIMAGE::Pointer im1, float factor)
{
    CUDAIMAGE::Pointer o = MakeLike(im1);
    UtilParams p = MakeParams(im1);
    p.fa[0] = factor;
    Run("multiply_image", p, im1,
        {{0u, im1->getFloatdata().buf}, {2u, o->getFloatdata().buf}});
    return o;
}

CUDAIMAGE::Pointer PreprocessImage(CUDAIMAGE::Pointer img, float low_val, float up_val)
{
    // Extrema on the DEVICE, as the reference does (ScalarFindMin/ScalarFindMax).
    // This used to download the whole image and reduce on the host - exact, but a
    // full blocking readback of every voxel, and it dominated this op's cost. min/max are order-independent and involve no rounding, so the
    // device reduction returns bit-identical values; only the transfer disappears.
    //
    // The seeds are asymmetric in the reference and that is reproduced: ScalarFindMax
    // seeds -1 (cuda_image_utilities.cu:105), so the maximum is max(-1, true_max);
    // ScalarFindMin seeds 1E100, which overflows f32 to +inf, so the minimum is the
    // true minimum.
    const size_t nfloat = img->NumFloats();
    const float img_min = Reduce(img->getFloatdata().buf, nfloat, ReduceOp::Min);
    const float img_max = Reduce(img->getFloatdata().buf, nfloat, ReduceOp::Max);

    CUDAIMAGE::Pointer o = MakeLike(img, 1);
    UtilParams p = MakeParams(img);
    p.sz[3] = 1;
    p.fa[0] = low_val; p.fa[1] = up_val; p.fa[2] = img_min; p.fa[3] = img_max;
    Run("preprocess_image", p, img,
        {{0u, img->getFloatdata().buf}, {2u, o->getFloatdata().buf}});
    return o;
}

void RestrictPhase(CUDAIMAGE::Pointer field, float3 phase)
{
    UtilParams p = MakeParams(field);
    p.fa[0] = phase.x; p.fa[1] = phase.y; p.fa[2] = phase.z;
    Run("restrict_phase", p, field, {{2u, field->getFloatdata().buf}});
}

void ContrainDefFields(CUDAIMAGE::Pointer ufield, CUDAIMAGE::Pointer dfield)
{
    UtilParams p = MakeParams(ufield);
    Run("constrain_def_fields", p, ufield,
        {{2u, dfield->getFloatdata().buf}, {4u, ufield->getFloatdata().buf}});
}

void ScaleUpdateField(CUDAIMAGE::Pointer field, float scale_factor)
{
    // Reference: max local norm over the field (spacing-normalised), then scale
    // by scale_factor/magnitude - and skip entirely if the field is degenerate.
    const float magnitude = Reduce(field->getFloatdata().buf, field->NumVoxels(),
                                   ReduceOp::MaxFieldNorm, field->spc);
    if(!(magnitude > 1E-20))
        return;

    UtilParams p = MakeParams(field);
    p.fa[0] = scale_factor / magnitude;
    Run("multiply_image_inplace", p, field, {{2u, field->getFloatdata().buf}});
}

void AddToUpdateField(CUDAIMAGE::Pointer updateField, CUDAIMAGE::Pointer updateField_temp,
                      float weight, bool normalize)
{
    float magnitude = 1;
    if(normalize)
        magnitude = std::sqrt(Reduce(updateField_temp->getFloatdata().buf,
                                     updateField_temp->NumFloats(), ReduceOp::SumSq));
    if(magnitude == 0)
        return;

    UtilParams p = MakeParams(updateField);
    p.fa[0] = weight / magnitude;      // divided host-side, as the reference does
    Run("add_to_update_field", p, updateField,
        {{0u, updateField_temp->getFloatdata().buf}, {2u, updateField->getFloatdata().buf}});
}

float SumImage(CUDAIMAGE::Pointer im1)
{
    return Reduce(im1->getFloatdata().buf, im1->NumFloats(), ReduceOp::Sum);
}

#endif
