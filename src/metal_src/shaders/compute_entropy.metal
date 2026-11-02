// Port of compute_entropy.cu : the ComputeJointEntropy pipeline.
// Translated from shaders/compute_entropy.wgsl.
//
// Design difference, and why it is still exact:
//
// CUDA builds a per-block joint histogram in shared memory (u32 atomics), writes
// one Nbins x Nbins slice per block, then sums the slices into a float histogram.
// With Nbins=100 that slice is 100*100*4 = 40 KB of workgroup storage, which
// exceeds WebGPU's portable limit (16 KB) and Metal's on much hardware. So this
// port accumulates a SINGLE histogram with global atomics instead.
//
// That is not an approximation: the counts are integers, integer addition is
// associative, and the per-block-then-sum total equals the direct total exactly.
// The subsequent u32 -> f32 conversion is exact as long as no bin exceeds 2^24
// (16.7 M); the largest validation volume has 1.8 M voxels, so a single bin
// cannot come close.
//
// Everything downstream - the marginal projections, the p*log(p) transform and
// the sum order - follows the reference exactly.

#include <metal_stdlib>
using namespace metal;

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];
constant uint MARGINAL_AXIS [[function_constant(3)]];   // 0 = moving (column sums), 1 = fixed (row sums)

constant int PADDING = 2;

struct Params {
    int4   sz;      // x, y, z, Nbins
    float4 lims;    // low1, high1, low2, high2
    float4 misc;    // hist_sum
};

// Parzen bin index, matching ITK's convention as the reference implements it.
static inline int parzen_index(float val, float low, float high, int nbins) {
    const float bin_size = (high - low) / float(nbins - 2 * PADDING);
    const float norm_min = low / bin_size - float(PADDING);
    const float term = val / bin_size - norm_min;
    int idx = int(term);
    if(idx < 2) {
        idx = 2;
    } else if(idx > nbins - 3) {
        idx = nbins - 3;
    }
    return idx;
}

kernel void joint_histogram(device const float  *img1  [[buffer(0)]],
                            device const float  *img2  [[buffer(1)]],
                            device atomic_uint  *histu [[buffer(2)]],
                            constant Params     &P     [[buffer(3)]],
                            uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x); const int j = int(gid.y); const int k = int(gid.z);
    if(i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    const int n = (k * P.sz.y + j) * P.sz.x + i;
    const float v1 = img1[n];
    const float v2 = img2[n];

    const float low1 = P.lims.x; const float high1 = P.lims.y;
    const float low2 = P.lims.z; const float high2 = P.lims.w;
    const int nbins = P.sz.w;

    if(v1 >= low1 && v2 >= low2 && v1 <= high1 && v2 <= high2) {
        const int fixed_idx  = parzen_index(v1, low1, high1, nbins);
        const int moving_idx = parzen_index(v2, low2, high2, nbins);
        atomic_fetch_add_explicit(&histu[fixed_idx * nbins + moving_idx], 1u,
                                  memory_order_relaxed);
    }
}

// u32 counts -> f32 histogram (AccumulateJointPartialHistogram_kernel's result).
kernel void hist_to_float(device atomic_uint *histu [[buffer(2)]],
                          constant Params    &P     [[buffer(3)]],
                          device float       *histf [[buffer(4)]],
                          uint3 gid [[thread_position_in_grid]])
{
    const int n = int(gid.x);
    const int nbins = P.sz.w;
    if(n >= nbins * nbins) { return; }
    histf[n] = float(atomic_load_explicit(&histu[n], memory_order_relaxed));
}

// Project the joint histogram onto a marginal:
//   moving (axis 0) sums down columns, fixed (axis 1) sums across rows.
kernel void joint_to_marginal(constant Params &P     [[buffer(3)]],
                              device float    *histf [[buffer(4)]],
                              device float    *marg  [[buffer(5)]],
                              uint3 gid [[thread_position_in_grid]])
{
    const int idx = int(gid.x);
    const int nbins = P.sz.w;
    if(idx >= nbins) { return; }

    float sm = 0.0f;
    for(int t = 0; t < nbins; t = t + 1) {
        if(MARGINAL_AXIS == 0u) {
            sm = sm + histf[t * nbins + idx];      // column sum -> moving
        } else {
            sm = sm + histf[idx * nbins + t];      // row sum    -> fixed
        }
    }
    marg[idx] = sm;
}

// The histogram sum arrives at buffer(6) element 0, straight from the reduction that
// produced it, rather than through P.misc.x. It is the SAME float32 - the host used to
// download it and copy it into the uniform - so the division is bit-identical. What
// disappears is the readback, which cost a full queue drain per call.
kernel void binwise_entropy_marginal(constant Params    &P        [[buffer(3)]],
                                     device float       *marg     [[buffer(5)]],
                                     device const float *hist_sum [[buffer(6)]],
                                     uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x);
    if(i >= P.sz.w) { return; }
    const float prob = marg[i] / hist_sum[0];
    if(prob > 1e-10f) { marg[i] = prob * log(prob); } else { marg[i] = 0.0f; }
}

kernel void binwise_entropy_joint(constant Params    &P        [[buffer(3)]],
                                  device float       *histf    [[buffer(4)]],
                                  device const float *hist_sum [[buffer(6)]],
                                  uint3 gid [[thread_position_in_grid]])
{
    const int n = int(gid.x);
    const int nbins = P.sz.w;
    if(n >= nbins * nbins) { return; }
    const float prob = histf[n] / hist_sum[0];
    if(prob > 1e-10f) { histf[n] = prob * log(prob); } else { histf[n] = 0.0f; }
}
