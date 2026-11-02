#ifndef _MTL_COMPUTEENTROPY_CXX
#define _MTL_COMPUTEENTROPY_CXX

#include "compute_entropy.h"
#include "reductions.h"
#include "metal_context.h"
#include "compute_entropy.metal.h"
#include <vector>

namespace
{
struct EntropyParams
{
    int32_t sz[4];       // x, y, z, Nbins
    float   lims[4];     // low1, high1, low2, high2
    float   misc[4];     // hist_sum
};


}

void ComputeJointEntropy(CUDAIMAGE::Pointer img1, float low_lim1, float high_lim1,
                         CUDAIMAGE::Pointer img2, float low_lim2, float high_lim2,
                         int Nbins, float &entropy_j, float &entropy_img1, float &entropy_img2)
{
    const size_t nb  = (size_t)Nbins;
    const size_t nb2 = nb * nb;

    EntropyParams p{};
    p.sz[0] = img1->sz.x; p.sz[1] = img1->sz.y; p.sz[2] = img1->sz.z; p.sz[3] = Nbins;
    p.lims[0] = low_lim1; p.lims[1] = high_lim1;
    p.lims[2] = low_lim2; p.lims[3] = high_lim2;

    mtlctx::Buffer histu = mtlctx::CreateStorage(nb2 * sizeof(uint32_t));
    mtlctx::Buffer histf = mtlctx::CreateStorage(nb2 * sizeof(float));
    mtlctx::Buffer marg  = mtlctx::CreateStorage(nb * sizeof(float));
    mtlctx::ZeroFresh(histu, nb2 * sizeof(uint32_t));

    // 1. joint histogram, global u32 atomics
    {
        const uint32_t wg = 4;
        mtlctx::Buffer par = mtlctx::CreateUniform(&p, sizeof(p));
        mtlctx::ComputePipeline pipe =
            mtlctx::Pipeline("compute_entropy", kcompute_entropyMSL, "joint_histogram",
                              wg, wg, wg);
        mtlctx::DispatchAt(pipe, {{0u, img1->getFloatdata().buf}, {1u, img2->getFloatdata().buf},
                                   {2u, histu}, {3u, par}},
                            (img1->sz.x + wg - 1) / wg, (img1->sz.y + wg - 1) / wg,
                            (img1->sz.z + wg - 1) / wg);
    }

    const uint32_t lin = 64;                    // 1-D entry points
    auto Run1D = [&](const char *entry, uint32_t n,
                     const std::vector<std::pair<uint32_t, mtlctx::Buffer> > &bufs) {
        mtlctx::ComputePipeline pipe =
            mtlctx::Pipeline("compute_entropy", kcompute_entropyMSL, entry, lin, 1, 1);
        mtlctx::DispatchAt(pipe, bufs, (n + lin - 1) / lin, 1, 1);
    };

    // 2. counts -> float histogram
    {
        mtlctx::Buffer par = mtlctx::CreateUniform(&p, sizeof(p));
        Run1D("hist_to_float", (uint32_t)nb2, {{2u, histu}, {3u, par}, {4u, histf}});
    }

    // 3. moving marginal (column sums), then its entropy - reference order
    auto MarginalEntropy = [&](uint32_t axis) -> mtlctx::Buffer {
        mtlctx::ComputePipeline pipe =
            mtlctx::Pipeline("compute_entropy", kcompute_entropyMSL, "joint_to_marginal",
                              lin, 1, 1, {{"MARGINAL_AXIS", (double)axis}});
        mtlctx::Buffer par = mtlctx::CreateUniform(&p, sizeof(p));
        mtlctx::DispatchAt(pipe, {{3u, par}, {4u, histf}, {5u, marg}},
                            ((uint32_t)nb + lin - 1) / lin, 1, 1);

        // The divisor stays on the device - see ReduceToBuffer's comment. Same float,
        // same division, one fewer queue drain per marginal.
        mtlctx::Buffer sum = ReduceToBuffer(marg, nb, ReduceOp::Sum, float3{1,1,1}, 1, 256);
        mtlctx::Buffer par2 = mtlctx::CreateUniform(&p, sizeof(p));
        Run1D("binwise_entropy_marginal", (uint32_t)nb,
              {{3u, par2}, {5u, marg}, {6u, sum}});
        // Leave the result on the device too: all three entropies are read together
        // below, so only the FIRST read pays a drain and the other two find an empty
        // queue. `marg` is reused by the next axis, but this reduction's output buffer
        // is its own and the queue is ordered, so its value is already fixed.
        return ReduceToBuffer(marg, nb, ReduceOp::Sum, float3{1,1,1}, 1, 256);
    };

    mtlctx::Buffer b_img2 = MarginalEntropy(0);      // moving
    mtlctx::Buffer b_img1 = MarginalEntropy(1);      // fixed

    // 4. joint entropy
    {
        mtlctx::Buffer sum = ReduceToBuffer(histf, nb2, ReduceOp::Sum, float3{1,1,1}, 1, 256);
        mtlctx::Buffer par = mtlctx::CreateUniform(&p, sizeof(p));
        Run1D("binwise_entropy_joint", (uint32_t)nb2, {{3u, par}, {4u, histf}, {6u, sum}});
        mtlctx::Buffer b_j = ReduceToBuffer(histf, nb2, ReduceOp::Sum, float3{1,1,1}, 1, 256);

        // One drain for the three scalars instead of one each.
        mtlctx::Download(b_j,    &entropy_j,    sizeof(float));
        mtlctx::Download(b_img2, &entropy_img2, sizeof(float));
        mtlctx::Download(b_img1, &entropy_img1, sizeof(float));
    }
}

#endif
