#ifndef _WGPU_REDUCTIONS_CXX
#define _WGPU_REDUCTIONS_CXX

#include "reductions.h"
#include "webgpu_context.h"
#include "reductions.wgsl.h"
#include <stdexcept>
#include <vector>

namespace
{
// Same constants as the CUDA side (cuda_image_utilities.cu:14-15). Keeping them
// identical keeps the partial-sum pairing identical, which is what makes the
// float result reproducible.

struct ReduceParams
{
    int32_t n[4];
    float   spc[4];
};
}

float Reduce(const wgpu::Buffer &buf, size_t n_elements, ReduceOp op, float3 spc,
             uint32_t kGrid, uint32_t kBlock)
{
    // reductions.wgsl's workgroup array is a fixed array<f32,1024> and the tree
    // halves from block/2, so the block must be a power of two and <= 1024. A bad
    // value would silently drop elements rather than fail.
    if(kBlock == 0 || kBlock > 1024 || (kBlock & (kBlock - 1)) != 0)
        throw std::runtime_error("Reduce(): block size must be a power of two <= 1024");

    ReduceParams p{};
    p.n[0] = (int32_t)n_elements;
    p.spc[0] = spc.x; p.spc[1] = spc.y; p.spc[2] = spc.z;

    wgpu::Buffer partials = wgpuctx::CreateStorage(kGrid * sizeof(float));
    wgpu::Buffer params   = wgpuctx::CreateUniform(&p, sizeof(p));

    wgpu::ComputePipeline pipe =
        wgpuctx::Pipeline("reductions", kreductionsWGSL, "reduce", kBlock, 1, 1,
                          {{"OP", (double)(int)op}});
    wgpuctx::Dispatch(pipe, {buf, partials, params}, kGrid, 1, 1);

    // Second stage over the 24 partials, mirroring ScalarFindSum<<<1,bSize>>>.
    ReduceParams p2{};
    p2.n[0] = (int32_t)kGrid;
    wgpu::Buffer out2    = wgpuctx::CreateStorage(kGrid * sizeof(float));
    wgpu::Buffer params2 = wgpuctx::CreateUniform(&p2, sizeof(p2));
    const ReduceOp op2 = (op == ReduceOp::Sum || op == ReduceOp::SumSq) ? ReduceOp::Sum
                       : (op == ReduceOp::Min)                          ? ReduceOp::Min
                                                                        : ReduceOp::Max;
    wgpu::ComputePipeline pipe2 =
        wgpuctx::Pipeline("reductions", kreductionsWGSL, "reduce", kBlock, 1, 1,
                          {{"OP", (double)(int)op2}});
    wgpuctx::Dispatch(pipe2, {partials, out2, params2}, 1, 1, 1);

    float result = 0;
    wgpuctx::Download(out2, &result, sizeof(float));
    return result;
}

#endif
