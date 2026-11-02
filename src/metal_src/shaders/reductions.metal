// Port of shaders/reductions.wgsl, itself a port of the reduction kernels in
// cuda_image_utilities.cu / compute_metric.cu: ScalarFindSum, ScalarFindSumSq,
// ScalarFindMax and FieldFindMaxLocalNorm.
//
// Summation is NOT associative in floating point, so unlike the elementwise
// kernels the *order* here is part of the specification. CUDA uses bSize=1024
// threads across gSize=24 blocks: thread t of block b starts at t + b*1024 and
// strides by 24*1024, then the block tree-reduces by halving from 512. That exact
// shape is reproduced (WGX defaults to 1024, dispatched as 24 threadgroups) so the
// partial sums pair up identically. mtlctx::Pipeline aborts rather than shrink a
// threadgroup, so this geometry cannot be silently lost.

#include <metal_stdlib>
using namespace metal;

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];
constant uint OP  [[function_constant(3)]];  // 0 sum, 1 sumsq, 2 max, 3 max local field norm, 4 min

struct Params {
    int4   n;      // element count (n.x), stride in elements (n.y)
    float4 spc;    // voxel spacing, for the field-norm variant
};

kernel void reduce(device const float *src [[buffer(0)]],
                   device       float *dst [[buffer(1)]],
                   constant Params    &P   [[buffer(2)]],
                   uint3 lid [[thread_position_in_threadgroup]],
                   uint3 wid [[threadgroup_position_in_grid]],
                   uint3 nwg [[threadgroups_per_grid]])
{
    threadgroup float sh[1024];

    const uint t      = lid.x;
    const int  gth    = int(t + wid.x * WGX);
    const int  stride = int(WGX * nwg.x);
    const int  n      = P.n.x;

    // CUDA seeds the max variants with -1, not -inf; a norm is never negative so
    // this only matters for an empty range, where it must be reproduced.
    //
    // ScalarFindMin seeds `float mn = 1E100` (cuda_image_utilities.cu:139), which
    // overflows f32 and is therefore +inf. The WGSL port had to write FLT_MAX
    // because WGSL cannot express inf in a const-expression; that value is kept
    // here rather than "corrected" to INFINITY, so the two ports agree with each
    // other as well as with the reference on every real input. min(FLT_MAX, x) == x
    // for every finite x, so this differs from CUDA only for an empty range or an
    // all-+inf input, neither of which occurs (the caller always passes n > 0).
    float acc = 0.0f;
    if(OP == 4u)      { acc = 3.4028234663852886e38f; }   // FLT_MAX
    else if(OP >= 2u) { acc = -1.0f; }

    for(int i = gth; i < n; i = i + stride) {
        if(OP == 0u) {
            acc = acc + src[i];
        } else if(OP == 1u) {
            acc = acc + src[i] * src[i];
        } else if(OP == 2u) {
            const float v = src[i];
            if(v > acc) { acc = v; }
        } else if(OP == 4u) {
            const float v = src[i];
            if(v < acc) { acc = v; }
        } else {
            const float x = src[3 * i]     / P.spc.x;
            const float y = src[3 * i + 1] / P.spc.y;
            const float z = src[3 * i + 2] / P.spc.z;
            const float v = sqrt(x * x + y * y + z * z);
            if(v > acc) { acc = v; }
        }
    }

    sh[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The barrier is outside the `t < size` test, as in the WGSL and the reference:
    // every thread in the threadgroup must reach it.
    for(uint size = WGX / 2u; size > 0u; size = size / 2u) {
        if(t < size) {
            if(OP < 2u) {
                sh[t] = sh[t] + sh[t + size];
            } else if(OP == 4u) {
                if(sh[t + size] < sh[t]) { sh[t] = sh[t + size]; }
            } else if(sh[t + size] > sh[t]) {
                sh[t] = sh[t + size];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(t == 0u) { dst[wid.x] = sh[0]; }
}
