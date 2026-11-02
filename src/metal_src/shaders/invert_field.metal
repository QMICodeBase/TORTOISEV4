// Port of shaders/invert_field.wgsl: ComputeFieldLocalNormImage and
// UpdateInvertField_kernel. The composition step reuses compose_fields.metal and
// the convergence reductions reuse reductions.metal, so only these two are new.
//
// NegateImage_kernel is folded into the norm pass: the reference computes the norm
// image from the composed field, then negates the composed field in place, and
// nothing reads the un-negated field in between.

#include <metal_stdlib>
using namespace metal;

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];

struct Params {
    int4   sz;
    float4 spc;
    float4 ctl;      // epsilon, max_error_norm
};

static inline int vidx(constant Params &P, int i, int j, int k) {
    return ((k * P.sz.y + j) * P.sz.x + i) * 3;
}
static inline int sidx(constant Params &P, int i, int j, int k) {
    return (k * P.sz.y + j) * P.sz.x + i;
}

// ComputeFieldLocalNormImage, then NegateImage_kernel on the composed field.
// Note the reference divides by the spacing twice per term rather than squaring
// it once; kept as written because the rounding differs.
kernel void local_norm_and_negate(device float    *composed [[buffer(0)]],
                                  device float    *scale    [[buffer(1)]],
                                  constant Params &P        [[buffer(3)]],
                                  uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x); const int j = int(gid.y); const int k = int(gid.z);
    if(i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    const int n = vidx(P, i, j, k);
    const float cx = composed[n]; const float cy = composed[n + 1]; const float cz = composed[n + 2];

    float nrm = cx * cx / P.spc.x / P.spc.x +
                cy * cy / P.spc.y / P.spc.y +
                cz * cz / P.spc.z / P.spc.z;
    nrm = sqrt(nrm);
    scale[sidx(P, i, j, k)] = nrm;

    composed[n]     = -cx;
    composed[n + 1] = -cy;
    composed[n + 2] = -cz;
}

// UpdateInvertField_kernel: clamp the step where the local norm is large, take an
// epsilon-scaled step, and force the outer shell to zero.
kernel void update_invert_field(device float    *composed [[buffer(0)]],
                                device float    *scale    [[buffer(1)]],
                                device float    *outp     [[buffer(2)]],
                                constant Params &P        [[buffer(3)]],
                                uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x); const int j = int(gid.y); const int k = int(gid.z);
    if(i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    const int n = vidx(P, i, j, k);
    const float eps    = P.ctl.x;
    const float maxerr = P.ctl.y;

    float ux = composed[n];
    float uy = composed[n + 1];
    float uz = composed[n + 2];

    const float scaledNorm = scale[sidx(P, i, j, k)];
    if(scaledNorm > eps * maxerr) {
        const float f = eps * maxerr / scaledNorm;
        ux = ux * f;
        uy = uy * f;
        uz = uz * f;
    }

    ux = outp[n]     + ux * eps;
    uy = outp[n + 1] + uy * eps;
    uz = outp[n + 2] + uz * eps;

    if(i == 0 || i == P.sz.x - 1 ||
       j == 0 || j == P.sz.y - 1 ||
       k == 0 || k == P.sz.z - 1) {
        ux = 0.0f; uy = 0.0f; uz = 0.0f;
    }

    outp[n]     = ux;
    outp[n + 1] = uy;
    outp[n + 2] = uz;
}
