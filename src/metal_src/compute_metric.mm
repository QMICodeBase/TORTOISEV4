#ifndef _MTL_COMPUTEMETRIC_CXX
#define _MTL_COMPUTEMETRIC_CXX

#include "compute_metric.h"
#include "reductions.h"
#include "metal_context.h"
#include "metric_cc.metal.h"
#include "metric_ccsk.metal.h"
#include "metric_msjac.metal.h"
#include "metric_ccjacs.metal.h"
#include <cmath>
#include <stdexcept>

namespace
{
struct MetricParams
{
    int32_t sz[4];
    float   spc[4];
    float   dir[3][4];
    float   phase[4];
    float   newph[4];
    int32_t axes[4];
    float   taps[8][4];
};

MetricParams MakeMetricParams(CUDAIMAGE::Pointer img)
{
    MetricParams p{};
    p.sz[0] = img->sz.x; p.sz[1] = img->sz.y; p.sz[2] = img->sz.z;
    p.spc[0] = img->spc.x; p.spc[1] = img->spc.y; p.spc[2] = img->spc.z;
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            p.dir[r][c] = (float)img->dir(r, c);
    return p;
}

CUDAIMAGE::Pointer MakeField(CUDAIMAGE::Pointer like)
{
    CUDAIMAGE::Pointer f = CUDAIMAGE::New();
    f->sz = like->sz; f->dir = like->dir; f->orig = like->orig; f->spc = like->spc;
    f->components_per_voxel = 3;
    f->Allocate();
    return f;
}
}

float ComputeMetric_CC(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                       CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM)
{
    updateFieldF = MakeField(up_img);
    updateFieldM = MakeField(up_img);

    CUDAIMAGE::Pointer metric = CUDAIMAGE::New();
    metric->sz = up_img->sz; metric->dir = up_img->dir;
    metric->orig = up_img->orig; metric->spc = up_img->spc;
    metric->components_per_voxel = 1;
    metric->Allocate();

    MetricParams p = MakeMetricParams(up_img);
    mtlctx::Buffer par = mtlctx::CreateUniform(&p, sizeof(p));

    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    mtlctx::ComputePipeline pipe =
        mtlctx::Pipeline("metric_cc", kmetric_ccMSL, "main_cc", wgx, wgy, wgz);
    mtlctx::Dispatch(pipe,
                      {up_img->getFloatdata().buf, down_img->getFloatdata().buf,
                       updateFieldF->getFloatdata().buf, updateFieldM->getFloatdata().buf,
                       metric->getFloatdata().buf, par},
                      (up_img->sz.x + wgx - 1) / wgx,
                      (up_img->sz.y + wgy - 1) / wgy,
                      (up_img->sz.z + wgz - 1) / wgz);

    const float sum = Reduce(metric->getFloatdata().buf, metric->NumVoxels(), ReduceOp::Sum);
    return sum / up_img->sz.x / up_img->sz.y / up_img->sz.z;
}

float ComputeMetric_CCSK(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                         const CUDAIMAGE::Pointer str_img,
                         CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                         float t)
{
    updateFieldF = MakeField(up_img);
    updateFieldM = MakeField(up_img);

    CUDAIMAGE::Pointer metric = CUDAIMAGE::New();
    metric->sz = up_img->sz; metric->dir = up_img->dir;
    metric->orig = up_img->orig; metric->spc = up_img->spc;
    metric->components_per_voxel = 1;
    metric->Allocate();

    CUDAIMAGE::Pointer K = CUDAIMAGE::New();
    K->sz = up_img->sz; K->dir = up_img->dir;
    K->orig = up_img->orig; K->spc = up_img->spc;
    K->components_per_voxel = 1;
    K->Allocate();                      // zeroed, matching cudaMemset3D

    MetricParams p = MakeMetricParams(up_img);
    p.phase[3] = t;
    mtlctx::Buffer par = mtlctx::CreateUniform(&p, sizeof(p));

    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    const uint32_t gx = (up_img->sz.x + wgx - 1) / wgx;
    const uint32_t gy = (up_img->sz.y + wgy - 1) / wgy;
    const uint32_t gz = (up_img->sz.z + wgz - 1) / wgz;

    // 1. build the K image
    mtlctx::ComputePipeline kpipe =
        mtlctx::Pipeline("metric_ccsk", kmetric_ccskMSL, "compute_k", wgx, wgy, wgz);
    mtlctx::DispatchAt(kpipe, {{0u, up_img->getFloatdata().buf},
                                {1u, down_img->getFloatdata().buf},
                                {2u, K->getFloatdata().buf},
                                {7u, par}}, gx, gy, gz);

    // 2. correlate it against the structural image
    mtlctx::ComputePipeline pipe =
        mtlctx::Pipeline("metric_ccsk", kmetric_ccskMSL, "main_ccsk", wgx, wgy, wgz);
    mtlctx::DispatchAt(pipe, {{0u, up_img->getFloatdata().buf},
                               {1u, down_img->getFloatdata().buf},
                               {2u, K->getFloatdata().buf},
                               {3u, str_img->getFloatdata().buf},
                               {4u, updateFieldF->getFloatdata().buf},
                               {5u, updateFieldM->getFloatdata().buf},
                               {6u, metric->getFloatdata().buf},
                               {7u, par}}, gx, gy, gz);

    const float sum = Reduce(metric->getFloatdata().buf, metric->NumVoxels(), ReduceOp::Sum);
    return sum / up_img->sz.x / up_img->sz.y / up_img->sz.z;
}

namespace
{
// The wrappers take the ITK operator (as the CUDA ones do) and pull the taps out
// here, so main/ calls are identical across backends.
std::vector<float> TapsOf(itk::GaussianOperator<float, 3> &oper)
{
    auto aa = oper.GetBufferReference();
    std::vector<float> t(aa.size());
    for(size_t m = 0; m < aa.size(); m++)
        t[m] = aa[m];
    return t;
}

void FillPhase(MetricParams &p, float3 phase_vector, const std::vector<float> &taps)
{
    p.phase[0] = phase_vector.x; p.phase[1] = phase_vector.y; p.phase[2] = phase_vector.z;
    for(int r = 0; r < 3; r++)
        p.newph[r] = p.dir[r][0] * phase_vector.x + p.dir[r][1] * phase_vector.y
                   + p.dir[r][2] * phase_vector.z;

    int phase = 2;
    if(std::fabs(phase_vector.x) > std::fabs(phase_vector.y) &&
       std::fabs(phase_vector.x) > std::fabs(phase_vector.z)) phase = 0;
    else if(std::fabs(phase_vector.y) > std::fabs(phase_vector.x) &&
            std::fabs(phase_vector.y) > std::fabs(phase_vector.z)) phase = 1;

    int phase_xyz = 2;
    if(std::fabs(p.newph[0]) > std::fabs(p.newph[1]) &&
       std::fabs(p.newph[0]) > std::fabs(p.newph[2])) phase_xyz = 0;
    else if(std::fabs(p.newph[1]) > std::fabs(p.newph[0]) &&
            std::fabs(p.newph[1]) > std::fabs(p.newph[2])) phase_xyz = 1;

    p.axes[0] = phase;
    p.axes[1] = phase_xyz;
    // The uniform block holds 32 taps; ITK bounds this via SetMaximumKernelWidth(31),
    // but the *WithTaps entry points are public, so defend the invariant instead of
    // silently truncating and letting the shader index past the array.
    // CUDA's c_Kernel is __constant__ float[31], so it aborts above 31.
    if(taps.size() > 31)
        throw std::runtime_error("kernel has " + std::to_string(taps.size()) +
                                 " taps; CUDA's c_Kernel holds at most 31");
    p.sz[3] = (int32_t)taps.size();
    for(size_t m = 0; m < taps.size(); m++)
        p.taps[m / 4][m % 4] = taps[m];
}
}

float ComputeMetric_MSJac(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                          const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV,
                          CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                          float3 phase_vector, itk::GaussianOperator<float, 3> &oper)
{
    return ComputeMetric_MSJacWithTaps(up_img, down_img, def_FINV, def_MINV,
                                       updateFieldF, updateFieldM, phase_vector, TapsOf(oper));
}

float ComputeMetric_MSJacWithTaps(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                                  const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV,
                                  CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                                  float3 phase_vector, const std::vector<float> &kernel_taps)
{
    updateFieldF = MakeField(up_img);
    updateFieldM = MakeField(up_img);

    CUDAIMAGE::Pointer metric = CUDAIMAGE::New();
    metric->sz = up_img->sz; metric->dir = up_img->dir;
    metric->orig = up_img->orig; metric->spc = up_img->spc;
    metric->components_per_voxel = 1;
    metric->Allocate();

    MetricParams p = MakeMetricParams(up_img);

    FillPhase(p, phase_vector, kernel_taps);

    mtlctx::Buffer par = mtlctx::CreateUniform(&p, sizeof(p));

    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    mtlctx::ComputePipeline pipe =
        mtlctx::Pipeline("metric_msjac", kmetric_msjacMSL, "main_msjac", wgx, wgy, wgz);
    mtlctx::Dispatch(pipe,
                      {up_img->getFloatdata().buf, down_img->getFloatdata().buf,
                       def_FINV->getFloatdata().buf, def_MINV->getFloatdata().buf,
                       updateFieldF->getFloatdata().buf, updateFieldM->getFloatdata().buf,
                       metric->getFloatdata().buf, par},
                      (up_img->sz.x + wgx - 1) / wgx,
                      (up_img->sz.y + wgy - 1) / wgy,
                      (up_img->sz.z + wgz - 1) / wgz);

    const float sum = Reduce(metric->getFloatdata().buf, metric->NumVoxels(), ReduceOp::Sum);
    return sum / up_img->sz.x / up_img->sz.y / up_img->sz.z;
}

float ComputeMetric_CCJacS(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                           const CUDAIMAGE::Pointer str_img,
                           const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV,
                           CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                           float3 phase_vector, itk::GaussianOperator<float, 3> &oper)
{
    return ComputeMetric_CCJacSWithTaps(up_img, down_img, str_img, def_FINV, def_MINV,
                                        updateFieldF, updateFieldM, phase_vector, TapsOf(oper));
}

float ComputeMetric_CCJacSWithTaps(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                                   const CUDAIMAGE::Pointer str_img,
                                   const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV,
                                   CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                                   float3 phase_vector, const std::vector<float> &kernel_taps)
{
    // metric_ccjacs.metal's `main` binds 9 storage buffers; the WebGPU default limit
    // is 8. Checked here rather than at startup so that runs which never compute
    // this metric (any dataset without a structural image) are unaffected.
    mtlctx::RequireStorageBuffers(9, "the CCJacS metric (metric_ccjacs.metal main)");

    updateFieldF = MakeField(up_img);
    updateFieldM = MakeField(up_img);

    auto scalarLike = [&](void) {
        CUDAIMAGE::Pointer a = CUDAIMAGE::New();
        a->sz = up_img->sz; a->dir = up_img->dir; a->orig = up_img->orig; a->spc = up_img->spc;
        a->components_per_voxel = 1;
        a->Allocate();
        return a;
    };

    CUDAIMAGE::Pointer metric = scalarLike();
    MetricParams p = MakeMetricParams(up_img);
    FillPhase(p, phase_vector, kernel_taps);
    mtlctx::Buffer par = mtlctx::CreateUniform(&p, sizeof(p));

    // wgx=32 is one coalesced run along a row; 4x4x4 made a 32-wide subgroup span 8
    // pitch-separated rows. Shape-agnostic - every elementwise shader indexes by
    // global id with an inside() guard. NOT for reductions, which fix their geometry.
    // Full measurement: PERF_NOTES.md 14.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    const uint32_t gx = (up_img->sz.x + wgx - 1) / wgx;
    const uint32_t gy = (up_img->sz.y + wgy - 1) / wgy;
    const uint32_t gz = (up_img->sz.z + wgz - 1) / wgz;

    // The reference runs the whole three-kernel chain twice: (up, def_FINV) then
    // (down, def_MINV), the second overwriting the metric image.
    auto pass = [&](CUDAIMAGE::Pointer img, CUDAIMAGE::Pointer fld, CUDAIMAGE::Pointer upd) {
        CUDAIMAGE::Pointer det  = scalarLike();
        CUDAIMAGE::Pointer sKS  = scalarLike();
        CUDAIMAGE::Pointer sSS  = scalarLike();
        CUDAIMAGE::Pointer sKK  = scalarLike();
        CUDAIMAGE::Pointer valS = scalarLike();
        CUDAIMAGE::Pointer valK = scalarLike();

        // Each entry point must be given exactly the bindings it declares.
        const std::vector<std::pair<uint32_t, mtlctx::Buffer> > b_det = {
            {0u, img->getFloatdata().buf}, {2u, fld->getFloatdata().buf},
            {3u, det->getFloatdata().buf}, {11u, par}};
        const std::vector<std::pair<uint32_t, mtlctx::Buffer> > b_str = {
            {1u, str_img->getFloatdata().buf}, {3u, det->getFloatdata().buf},
            {4u, sKS->getFloatdata().buf}, {5u, sSS->getFloatdata().buf},
            {6u, sKK->getFloatdata().buf}, {7u, valS->getFloatdata().buf},
            {8u, valK->getFloatdata().buf}, {11u, par}};
        const std::vector<std::pair<uint32_t, mtlctx::Buffer> > b_main = {
            {0u, img->getFloatdata().buf}, {2u, fld->getFloatdata().buf},
            {4u, sKS->getFloatdata().buf}, {5u, sSS->getFloatdata().buf},
            {6u, sKK->getFloatdata().buf}, {7u, valS->getFloatdata().buf},
            {8u, valK->getFloatdata().buf}, {9u, upd->getFloatdata().buf},
            {10u, metric->getFloatdata().buf}, {11u, par}};

        struct { const char *entry; const std::vector<std::pair<uint32_t, mtlctx::Buffer> > *b; }
            steps[3] = {{"compute_det_img", &b_det},
                        {"compute_structs", &b_str},
                        {"main_ccjacs", &b_main}};
        for(int s = 0; s < 3; s++)
        {
            mtlctx::ComputePipeline pipe =
                mtlctx::Pipeline("metric_ccjacs", kmetric_ccjacsMSL, steps[s].entry, wgx, wgy, wgz);
            mtlctx::DispatchAt(pipe, *steps[s].b, gx, gy, gz);
        }
    };

    pass(up_img,   def_FINV, updateFieldF);
    pass(down_img, def_MINV, updateFieldM);

    const float sum = Reduce(metric->getFloatdata().buf, metric->NumVoxels(), ReduceOp::Sum);
    return sum / up_img->sz.x / up_img->sz.y / up_img->sz.z;
}

#endif
