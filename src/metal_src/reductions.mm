#ifndef _MTL_REDUCTIONS_CXX
#define _MTL_REDUCTIONS_CXX

#include "reductions.h"
#include "metal_context.h"
#include "reductions.metal.h"
#include <stdexcept>
#include <vector>

namespace
{

struct ReduceParams
{
    int32_t n[4];
    float   spc[4];
};
}

float Reduce(const mtlctx::Buffer &buf, size_t n_elements, ReduceOp op, float3 spc,
             uint32_t kGrid, uint32_t kBlock)
{
    mtlctx::Buffer out = ReduceToBuffer(buf, n_elements, op, spc, kGrid, kBlock);
    float result = 0;
    mtlctx::Download(out, &result, sizeof(float));
    return result;
}

mtlctx::Buffer ReduceToBuffer(const mtlctx::Buffer &buf, size_t n_elements, ReduceOp op,
                              float3 spc, uint32_t kGrid, uint32_t kBlock)
{
    // reductions.metal's workgroup array is a fixed array<f32,1024> and the tree
    // halves from block/2, so the block must be a power of two and <= 1024. A bad
    // value would silently drop elements rather than fail.
    if(kBlock == 0 || kBlock > 1024 || (kBlock & (kBlock - 1)) != 0)
        throw std::runtime_error("Reduce(): block size must be a power of two <= 1024");

    ReduceParams p{};
    p.n[0] = (int32_t)n_elements;
    p.spc[0] = spc.x; p.spc[1] = spc.y; p.spc[2] = spc.z;

    mtlctx::Buffer partials = mtlctx::CreateStorage(kGrid * sizeof(float));
    mtlctx::Buffer params   = mtlctx::CreateUniform(&p, sizeof(p));

    mtlctx::ComputePipeline pipe =
        mtlctx::Pipeline("reductions", kreductionsMSL, "reduce", kBlock, 1, 1,
                          {{"OP", (double)(int)op}});
    mtlctx::Dispatch(pipe, {buf, partials, params}, kGrid, 1, 1);

    // Second stage over the 24 partials, mirroring ScalarFindSum<<<1,bSize>>>.
    ReduceParams p2{};
    p2.n[0] = (int32_t)kGrid;
    mtlctx::Buffer out2    = mtlctx::CreateStorage(kGrid * sizeof(float));
    mtlctx::Buffer params2 = mtlctx::CreateUniform(&p2, sizeof(p2));
    const ReduceOp op2 = (op == ReduceOp::Sum || op == ReduceOp::SumSq) ? ReduceOp::Sum
                       : (op == ReduceOp::Min)                          ? ReduceOp::Min
                                                                        : ReduceOp::Max;
    mtlctx::ComputePipeline pipe2 =
        mtlctx::Pipeline("reductions", kreductionsMSL, "reduce", kBlock, 1, 1,
                          {{"OP", (double)(int)op2}});
    mtlctx::Dispatch(pipe2, {partials, out2, params2}, 1, 1, 1);

    return out2;
}

#endif
