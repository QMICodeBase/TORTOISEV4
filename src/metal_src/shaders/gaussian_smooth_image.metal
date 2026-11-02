// Port of shaders/gaussian_smooth_image.wgsl, itself a port of
// gaussian_smooth_image.cu : the three separable passes plus
// AdjustFieldBoundary_kernel.
//
// Two faithfulness points:
//
//  * The CUDA kernels stage each line in shared memory, but that is a
//    performance detail - the arithmetic is a plain 1D convolution. Reading taps
//    straight from the buffer gives identical results and avoids tying the
//    workgroup size to the image dimension, which would break for any axis
//    longer than the portable invocation limit. The accumulation order
//    (tap 0 upward) is preserved, which is what actually affects rounding.
//
//  * Out-of-range taps are SKIPPED, not clamped and not zero-padded, and the
//    kernel is NOT renormalised at the boundary (gaussian_smooth_image.cu:60).
//    So edge voxels are convolved with a truncated kernel summing to less than
//    one, and get systematically darker. That is the reference behaviour and is
//    reproduced exactly.

#include <metal_stdlib>
using namespace metal;

constant uint WGX  [[function_constant(0)]];
constant uint WGY  [[function_constant(1)]];
constant uint WGZ  [[function_constant(2)]];
constant uint AXIS [[function_constant(3)]];   // 0 = x, 1 = y, 2 = z

struct Params {
    int4   sz;            // x, y, z, Ncomponents
    int4   ksz;           // kernel_sz, (weights below are for the boundary pass)
    float4 wts;           // weight1, weight2
    float4 kern[8];       // up to 32 taps
};

static inline float tap(constant Params &P, int i) {
    const float4 v = P.kern[i / 4];
    switch(i % 4) {
        case 0:  return v.x;
        case 1:  return v.y;
        case 2:  return v.z;
        default: return v.w;
    }
}

static inline int idx(constant Params &P, int i, int j, int k, int c) {
    return ((k * P.sz.y + j) * P.sz.x + i) * P.sz.w + c;
}

kernel void main_smooth(device const float *src [[buffer(0)]],
                        device       float *dst [[buffer(1)]],
                        constant Params    &P   [[buffer(2)]],
                        uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x);
    const int j = int(gid.y);
    const int k = int(gid.z);
    if(i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    const int mask = P.ksz.x;
    const int halfk = mask / 2;

    // Length of the axis being convolved, and this voxel's position along it.
    int extent = P.sz.x;
    int pos    = i;
    if(AXIS == 1u) { extent = P.sz.y; pos = j; }
    if(AXIS == 2u) { extent = P.sz.z; pos = k; }

    for(int c = 0; c < P.sz.w; c = c + 1) {
        float val = 0.0f;
        for(int t = 0; t < mask; t = t + 1) {
            const int p = pos + t - halfk;
            if(p >= 0 && p < extent) {          // out-of-range taps contribute nothing
                float s = 0.0f;
                if(AXIS == 0u)      { s = src[idx(P, p, j, k, c)]; }
                else if(AXIS == 1u) { s = src[idx(P, i, p, k, c)]; }
                else                { s = src[idx(P, i, j, p, c)]; }
                val = val + s * tap(P, t);
            }
        }
        dst[idx(P, i, j, k, c)] = val;
    }
}

// AdjustFieldBoundary_kernel: zero the outer shell, blend the interior.
// Runs in place on the smoothed field, reading the original for the blend.
kernel void adjust_boundary(device const float *src [[buffer(0)]],
                            device       float *dst [[buffer(1)]],
                            constant Params    &P   [[buffer(2)]],
                            uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x);
    const int j = int(gid.y);
    const int k = int(gid.z);
    if(i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    const bool edge = (i == 0 || i == P.sz.x - 1 ||
                       j == 0 || j == P.sz.y - 1 ||
                       k == 0 || k == P.sz.z - 1);

    for(int c = 0; c < P.sz.w; c = c + 1) {
        const int n = idx(P, i, j, k, c);
        if(edge) {
            dst[n] = 0.0f;
        } else {
            dst[n] = dst[n] * P.wts.x + src[n] * P.wts.y;
        }
    }
}
