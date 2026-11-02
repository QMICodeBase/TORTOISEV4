// Port of shaders/image_utilities.wgsl, itself a port of the elementwise kernels
// in cuda_image_utilities.cu. Translated statement for statement: the arithmetic
// grouping IS the specification (see preprocess_image below), so nothing here is
// tidied, factored or reassociated.
//
// One module, several entry points, sharing the indexing helper and parameter
// block exactly as the WGSL does.

#include <metal_stdlib>
using namespace metal;

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];

struct Params {
    int4   sz;     // x, y, z, Ncomponents
    float4 fa;     // scalar arguments (meaning is per-kernel)
    float4 fb;
};

static inline int base(constant Params &P, int i, int j, int k) {
    return ((k * P.sz.y + j) * P.sz.x + i) * P.sz.w;
}

static inline bool inside(constant Params &P, uint3 gid) {
    return int(gid.x) < P.sz.x && int(gid.y) < P.sz.y && int(gid.z) < P.sz.z;
}

// Binding numbers match the WGSL @binding() values one for one:
//   0 a, 1 b, 2 outp, 3 P, 4 uf (second in-place field, constrain_def_fields only)

kernel void add_images(device const float *a    [[buffer(0)]],
                       device const float *b    [[buffer(1)]],
                       device       float *outp [[buffer(2)]],
                       constant Params    &P    [[buffer(3)]],
                       uint3 gid [[thread_position_in_grid]])
{
    if(!inside(P, gid)) { return; }
    const int n = base(P, int(gid.x), int(gid.y), int(gid.z));
    for(int m = 0; m < P.sz.w; m = m + 1) { outp[n + m] = a[n + m] + b[n + m]; }
}

kernel void multiply_images(device const float *a    [[buffer(0)]],
                            device const float *b    [[buffer(1)]],
                            device       float *outp [[buffer(2)]],
                            constant Params    &P    [[buffer(3)]],
                            uint3 gid [[thread_position_in_grid]])
{
    if(!inside(P, gid)) { return; }
    const int n = base(P, int(gid.x), int(gid.y), int(gid.z));
    for(int m = 0; m < P.sz.w; m = m + 1) { outp[n + m] = a[n + m] * b[n + m]; }
}

// fa.x = factor
kernel void multiply_image(device const float *a    [[buffer(0)]],
                           device       float *outp [[buffer(2)]],
                           constant Params    &P    [[buffer(3)]],
                           uint3 gid [[thread_position_in_grid]])
{
    if(!inside(P, gid)) { return; }
    const int n = base(P, int(gid.x), int(gid.y), int(gid.z));
    for(int m = 0; m < P.sz.w; m = m + 1) { outp[n + m] = a[n + m] * P.fa.x; }
}

// Rescale [img_min,img_max] -> [low_val,up_val]. Written in the reference's own
// three-term form rather than the algebraically tidier one, because the grouping
// changes the rounding. fa = (low_val, up_val, img_min, img_max)
kernel void preprocess_image(device const float *a    [[buffer(0)]],
                             device       float *outp [[buffer(2)]],
                             constant Params    &P    [[buffer(3)]],
                             uint3 gid [[thread_position_in_grid]])
{
    if(!inside(P, gid)) { return; }
    const int n = base(P, int(gid.x), int(gid.y), int(gid.z));
    const float low = P.fa.x; const float up = P.fa.y;
    const float imin = P.fa.z; const float imax = P.fa.w;
    outp[n] = (up - low) / (imax - imin) * a[n]
              - imin * (up - low) / (imax - imin)
              + low;
}

// Project each vector onto the phase-encode direction. fa.xyz = phase
kernel void restrict_phase(device float    *outp [[buffer(2)]],
                           constant Params &P    [[buffer(3)]],
                           uint3 gid [[thread_position_in_grid]])
{
    if(!inside(P, gid)) { return; }
    const int n = base(P, int(gid.x), int(gid.y), int(gid.z));
    const float vx = outp[n]; const float vy = outp[n + 1]; const float vz = outp[n + 2];
    float nrm = vx * vx + vy * vy + vz * vz;
    nrm = sqrt(nrm);
    if(nrm != 0.0f) {
        const float ux = vx / nrm; const float uy = vy / nrm; const float uz = vz / nrm;
        const float d = ux * P.fa.x + uy * P.fa.y + uz * P.fa.z;
        outp[n]     = P.fa.x * nrm * d;
        outp[n + 1] = P.fa.y * nrm * d;
        outp[n + 2] = P.fa.z * nrm * d;
    }
}

// Make the two fields exact opposites: u = (u-d)/2, d = -u. In place on both:
// `uf` (binding 4) is the up field and `outp` (binding 2) the down field.
kernel void constrain_def_fields(device float    *outp [[buffer(2)]],
                                 constant Params &P    [[buffer(3)]],
                                 device float    *uf   [[buffer(4)]],
                                 uint3 gid [[thread_position_in_grid]])
{
    if(!inside(P, gid)) { return; }
    const int n = base(P, int(gid.x), int(gid.y), int(gid.z));
    for(int v = 0; v < 3; v = v + 1) {
        const float val = (uf[n + v] - outp[n + v]) * 0.5f;
        uf[n + v]   = val;
        outp[n + v] = -val;
    }
}

// In-place scale, used by ScaleUpdateField after its max reduction. fa.x = factor
kernel void multiply_image_inplace(device float    *outp [[buffer(2)]],
                                   constant Params &P    [[buffer(3)]],
                                   uint3 gid [[thread_position_in_grid]])
{
    if(!inside(P, gid)) { return; }
    const int n = base(P, int(gid.x), int(gid.y), int(gid.z));
    for(int m = 0; m < P.sz.w; m = m + 1) { outp[n + m] = outp[n + m] * P.fa.x; }
}

// total += to_add * factor. The reference computes factor = weight/magnitude on
// the HOST and passes one float (cuda_image_utilities.cu:328); doing the divide
// here instead would regroup the arithmetic and change the last bits.
kernel void add_to_update_field(device const float *a    [[buffer(0)]],
                                device       float *outp [[buffer(2)]],
                                constant Params    &P    [[buffer(3)]],
                                uint3 gid [[thread_position_in_grid]])
{
    if(!inside(P, gid)) { return; }
    const int n = base(P, int(gid.x), int(gid.y), int(gid.z));
    for(int m = 0; m < P.sz.w; m = m + 1) {
        outp[n + m] = outp[n + m] + a[n + m] * P.fa.x;
    }
}
