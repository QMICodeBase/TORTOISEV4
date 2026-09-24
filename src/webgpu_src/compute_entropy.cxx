#ifndef _WGPU_COMPUTEENTROPY_CXX
#define _WGPU_COMPUTEENTROPY_CXX

#include "compute_entropy.h"
#include "reductions.h"
#include "webgpu_context.h"
#include "compute_entropy.wgsl.h"
#include <vector>

namespace
{
struct EntropyParams
{
    int32_t sz[4];       // x, y, z, Nbins
    float   lims[4];     // low1, high1, low2, high2
    float   misc[4];     // hist_sum
};

// compute_entropy.cu's ScalarFindSum2 is one block of 256 (bSize there is 256,
// not the 1024 used in cuda_image_utilities.cu). Matching the shape matches the
// summation order.
const uint32_t kSumGrid  = 1;
const uint32_t kSumBlock = 256;

float SumOf(const wgpu::Buffer &b, size_t n)
{
    return Reduce(b, n, ReduceOp::Sum, float3{1.f, 1.f, 1.f}, kSumGrid, kSumBlock);
}
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

    wgpu::Buffer histu = wgpuctx::CreateStorage(nb2 * sizeof(uint32_t));
    wgpu::Buffer histf = wgpuctx::CreateStorage(nb2 * sizeof(float));
    wgpu::Buffer marg  = wgpuctx::CreateStorage(nb * sizeof(float));
    wgpuctx::Zero(histu, nb2 * sizeof(uint32_t));

    // 1. joint histogram, global u32 atomics
    {
        const uint32_t wg = 4;
        wgpu::Buffer par = wgpuctx::CreateUniform(&p, sizeof(p));
        wgpu::ComputePipeline pipe =
            wgpuctx::Pipeline("compute_entropy", kcompute_entropyWGSL, "joint_histogram",
                              wg, wg, wg);
        wgpuctx::DispatchAt(pipe, {{0u, img1->getFloatdata().buf}, {1u, img2->getFloatdata().buf},
                                   {2u, histu}, {3u, par}},
                            (img1->sz.x + wg - 1) / wg, (img1->sz.y + wg - 1) / wg,
                            (img1->sz.z + wg - 1) / wg);
    }

    const uint32_t lin = 64;                    // 1-D entry points
    auto Run1D = [&](const char *entry, uint32_t n,
                     const std::vector<std::pair<uint32_t, wgpu::Buffer> > &bufs) {
        wgpu::ComputePipeline pipe =
            wgpuctx::Pipeline("compute_entropy", kcompute_entropyWGSL, entry, lin, 1, 1);
        wgpuctx::DispatchAt(pipe, bufs, (n + lin - 1) / lin, 1, 1);
    };

    // 2. counts -> float histogram
    {
        wgpu::Buffer par = wgpuctx::CreateUniform(&p, sizeof(p));
        Run1D("hist_to_float", (uint32_t)nb2, {{2u, histu}, {3u, par}, {4u, histf}});
    }

    // 3. moving marginal (column sums), then its entropy - reference order
    auto MarginalEntropy = [&](uint32_t axis) {
        wgpu::ComputePipeline pipe =
            wgpuctx::Pipeline("compute_entropy", kcompute_entropyWGSL, "joint_to_marginal",
                              lin, 1, 1, {{"MARGINAL_AXIS", (double)axis}});
        wgpu::Buffer par = wgpuctx::CreateUniform(&p, sizeof(p));
        wgpuctx::DispatchAt(pipe, {{3u, par}, {4u, histf}, {5u, marg}},
                            ((uint32_t)nb + lin - 1) / lin, 1, 1);

        EntropyParams q = p;
        q.misc[0] = SumOf(marg, nb);
        wgpu::Buffer par2 = wgpuctx::CreateUniform(&q, sizeof(q));
        Run1D("binwise_entropy_marginal", (uint32_t)nb, {{3u, par2}, {5u, marg}});
        return SumOf(marg, nb);
    };

    entropy_img2 = MarginalEntropy(0);      // moving
    entropy_img1 = MarginalEntropy(1);      // fixed

    // 4. joint entropy
    {
        EntropyParams q = p;
        q.misc[0] = SumOf(histf, nb2);
        wgpu::Buffer par = wgpuctx::CreateUniform(&q, sizeof(q));
        Run1D("binwise_entropy_joint", (uint32_t)nb2, {{3u, par}, {4u, histf}});
        entropy_j = SumOf(histf, nb2);
    }
}

#endif
