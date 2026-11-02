// webgpu_probe - M2 smoke test for the WebGPU backend.
//
//   1. selects an adapter under the CLAUDE.md 3 policy and logs its identity
//   2. allocates a buffer the size of the largest validation volume
//   3. runs a trivial WGSL kernel and verifies the result on the host
//
// Exit 0 only if all three succeed. TORTOISE_WEBGPU_PROBE_ONLY=1 makes a failed
// adapter selection return non-zero instead of aborting, so the rejection path
// is testable on a machine with no NVIDIA card.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "webgpu_context.h"
#include "saxpy.wgsl.h"   // generated from shaders/saxpy.wgsl


int main()
{
    if(!wgpuctx::Init())
    {
        std::printf("PROBE: adapter selection refused (expected under "
                    "TORTOISE_WEBGPU_PROBE_ONLY on a non-conforming host)\n");
        return 2;
    }
    std::printf("PROBE: adapter = %s\n", wgpuctx::Info().Describe().c_str());

    // 1. Allocation at the scale the pipeline actually needs. `slow` is
    //    140x140x92 with 3 components -> the largest single field in play.
    const size_t nvox  = (size_t)140 * 140 * 92;
    const size_t nfloat = nvox * 3;
    wgpu::Buffer big = wgpuctx::CreateStorage(nfloat * sizeof(float));
    if(!big)
    {
        std::printf("PROBE: FAILED to allocate %.1f MiB\n",
                    nfloat * sizeof(float) / 1048576.0);
        return 1;
    }
    std::printf("PROBE: allocated %.1f MiB (largest validation field)\n",
                nfloat * sizeof(float) / 1048576.0);

    // 2. Trivial kernel over a smaller buffer, verified on the host.
    const size_t n = 1u << 20;
    std::vector<float> host(n);
    for(size_t i = 0; i < n; i++)
        host[i] = (float)(i % 977) * 0.5f;

    wgpu::Buffer src = wgpuctx::CreateStorage(n * sizeof(float));
    wgpu::Buffer dst = wgpuctx::CreateStorage(n * sizeof(float));
    wgpuctx::Upload(src, host.data(), n * sizeof(float));

    const uint32_t wg = 64;
    wgpu::ComputePipeline p = wgpuctx::Pipeline("saxpy", ksaxpyWGSL, "main", wg, 1, 1);
    wgpuctx::Dispatch(p, {src, dst}, (uint32_t)((n + wg - 1) / wg), 1, 1);

    std::vector<float> got(n);
    wgpuctx::Download(dst, got.data(), n * sizeof(float));

    size_t bad = 0;
    double maxerr = 0.0;
    for(size_t i = 0; i < n; i++)
    {
        const float want = host[i] * 2.0f + 1.0f;
        const double e = std::fabs(got[i] - want);
        if(e != 0.0) { bad++; maxerr = e > maxerr ? e : maxerr; }
    }
    std::printf("PROBE: kernel %zu elements, %zu mismatches, max err %.3g\n", n, bad, maxerr);
    std::printf("PROBE: cumulative device allocation %.1f MiB (not live usage)\n",
                wgpuctx::BytesAllocatedCumulative() / 1048576.0);

    if(bad)
    {
        std::printf("PROBE: FAILED\n");
        return 1;
    }
    std::printf("PROBE: OK\n");
    return 0;
}
