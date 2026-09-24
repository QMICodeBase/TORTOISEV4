// metal_probe - M1 smoke test for the Metal backend (CLAUDE.md 5.1).
//
//   1. selects the Metal device and logs its identity
//   2. allocates a buffer the size of the largest validation volume
//   3. runs a trivial MSL kernel and verifies the result on the host
//   4. reports the device allocation checkpoints the loop protocol records
//
// Exit 0 only if all succeed. TORTOISE_METAL_PROBE_ONLY=1 makes a failed device
// selection return non-zero instead of aborting, so the rejection path is testable.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "metal_context.h"
#include "saxpy.metal.h"   // generated from shaders/saxpy.metal

int main()
{
    if(!mtlctx::Init())
    {
        std::printf("PROBE: device selection refused (expected under "
                    "TORTOISE_METAL_PROBE_ONLY on a host with no usable Metal device)\n");
        return 2;
    }
    std::printf("PROBE: device = %s\n", mtlctx::Info().Describe().c_str());
    std::printf("PROBE: allocation after setup %.1f MiB\n",
                mtlctx::PeakDeviceBytes() / 1048576.0);
    mtlctx::ReportLimits();

    // 1. Allocation at the scale the pipeline needs: `slow` is 140x140x92 with
    //    3 components, the largest single field in play.
    const size_t nvox   = (size_t)140 * 140 * 92;
    const size_t nfloat = nvox * 3;
    mtlctx::Buffer big = mtlctx::CreateStorage(nfloat * sizeof(float));
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

    mtlctx::Buffer src = mtlctx::CreateStorage(n * sizeof(float));
    mtlctx::Buffer dst = mtlctx::CreateStorage(n * sizeof(float));
    mtlctx::Upload(src, host.data(), n * sizeof(float));
    const uint32_t count = (uint32_t)n;
    mtlctx::Buffer nbuf = mtlctx::CreateUniform(&count, sizeof(count));

    // Negative path: a bad entry point must abort with a diagnostic, not return a
    // pipeline that silently leaves its outputs untouched (the failure mode that
    // looks exactly like a porting bug). Checked by the caller's exit code.
    if(std::getenv("TORTOISE_METAL_NEGATIVE"))
    {
        std::printf("PROBE: requesting a nonexistent entry point, expecting abort\n");
        std::fflush(stdout);
        mtlctx::Pipeline("saxpy", ksaxpyMSL, "no_such_entry_point", 64, 1, 1);
        std::printf("PROBE: FAILED - bad entry point was accepted\n");
        return 1;
    }

    const uint32_t wg = 64;
    mtlctx::ComputePipeline p = mtlctx::Pipeline("saxpy", ksaxpyMSL, "main_saxpy", wg, 1, 1);
    mtlctx::Dispatch(p, {src, dst, nbuf}, (uint32_t)((n + wg - 1) / wg), 1, 1);

    std::vector<float> got(n);
    mtlctx::Download(dst, got.data(), n * sizeof(float));

    size_t bad = 0;
    double maxerr = 0.0;
    for(size_t i = 0; i < n; i++)
    {
        const float want = host[i] * 2.0f + 1.0f;
        const double e = std::fabs(got[i] - want);
        if(e != 0.0) { bad++; maxerr = e > maxerr ? e : maxerr; }
    }
    std::printf("PROBE: kernel %zu elements, %zu mismatches, max err %.3g\n", n, bad, maxerr);
    std::printf("PROBE: cumulative allocation %.1f MiB, peak device allocation %.1f MiB\n",
                mtlctx::BytesAllocatedCumulative() / 1048576.0,
                mtlctx::PeakDeviceBytes() / 1048576.0);

    if(bad || mtlctx::ErrorCount())
    {
        std::printf("PROBE: FAILED\n");
        return 1;
    }
    std::printf("PROBE: OK\n");
    return 0;
}
