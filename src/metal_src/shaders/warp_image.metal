// Port of shaders/warp_image.wgsl, itself a port of warp_image.cu : warp_image_kernel
//
// This is the one operation where CUDA's cudaAddressModeBorder is load-bearing:
// the kernel samples tex3D with NO domain guard, so neighbours outside
// [0, N-1] contribute 0 while in-range neighbours still contribute. That is
// different from ResampleImage/QuadraticTransformImage, which zero the whole
// voxel when the sample point leaves the domain (CLAUDE.md 5.2).
//
// Neither WebGPU nor this port uses a sampler: the reference needs clamp-to-border,
// so the trilinear filter is done by hand over a plain storage buffer.
//
// On weight precision: the CUDA Programming Guide says the texture unit stores
// interpolation weights in 1.8 fixed point (8 fractional bits), which would make
// exact fp32 weights diverge from the reference. TEXFILTER_QUANTISE implements
// that documented model - and MEASUREMENT SAYS IT IS NOT WHAT THE HARDWARE DOES:
//
//   WarpImage.0 vs the CUDA golden vector, exact fp32 weights : rel 1.45e-6
//   WarpImage.0 vs the same vector, floor(w*256)/256          : rel 1.37e-3
//
// Emulating the documented quantisation is ~1000x FURTHER from CUDA. A synthetic
// linear ramp (where trilinear interpolation is exact) confirms the direction:
// CUDA deviates from exact trilinear by 3.32e-5, so its sampler is lossy, but not
// in the way floor(w*256)/256 models. The hardware's actual rounding is not
// established.
//
// So the default is exact fp32 weights because that is empirically the CLOSEST
// AVAILABLE MATCH to the reference - not because it is more accurate. Accuracy is
// not the goal here; agreement is. TEXFILTER_QUANTISE is retained as a diagnostic
// for re-testing this on other hardware. RESULT ON APPLE SILICON: not needed. Exact
// fp32 weights pass the same gates here as on the NVIDIA/Vulkan backend, which is
// expected - this port never touches a texture unit, so there is no hardware filter
// whose rounding could differ. Enable with TORTOISE_METAL_CUDA_TEXFILTER.

#include <metal_stdlib>
using namespace metal;

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];
// uint rather than bool: every function constant in this backend is uint, so one
// rule covers them all and the host needs no name->type table (CLAUDE.md 5.1).
constant uint TEXFILTER_QUANTISE [[function_constant(3)]];

struct Params {
    int4   sz;      // x, y, z, unused
    float4 res;
    float4 dir0, dir1, dir2;
};

static inline float fetch(device const float *img, constant Params &P, int i, int j, int k) {
    // border: anything outside the array reads as zero
    if(i < 0 || j < 0 || k < 0 || i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) {
        return 0.0f;
    }
    return img[(k * P.sz.y + j) * P.sz.x + i];
}

static inline float weight(float a) {
    if(TEXFILTER_QUANTISE != 0u) {
        // CUDA stores the filter weight in 1.8 fixed point.
        return floor(a * 256.0f) / 256.0f;
    }
    return a;
}

// Trilinear sample at unnormalised coordinates, zero outside the volume.
// CUDA is handed (iw+0.5) and internally subtracts 0.5, so the base index is
// floor(iw) and the weight is frac(iw) - no half-texel juggling needed here.
static inline float sample_border(device const float *img, constant Params &P,
                                  float iw, float jw, float kw) {
    const float fx = floor(iw); const float fy = floor(jw); const float fz = floor(kw);
    const float ax = weight(iw - fx); const float ay = weight(jw - fy); const float az = weight(kw - fz);
    const int ix = int(fx); const int iy = int(fy); const int iz = int(fz);

    const float c000 = fetch(img, P, ix,     iy,     iz);
    const float c100 = fetch(img, P, ix + 1, iy,     iz);
    const float c010 = fetch(img, P, ix,     iy + 1, iz);
    const float c110 = fetch(img, P, ix + 1, iy + 1, iz);
    const float c001 = fetch(img, P, ix,     iy,     iz + 1);
    const float c101 = fetch(img, P, ix + 1, iy,     iz + 1);
    const float c011 = fetch(img, P, ix,     iy + 1, iz + 1);
    const float c111 = fetch(img, P, ix + 1, iy + 1, iz + 1);

    const float x00 = c000 * (1.0f - ax) + c100 * ax;
    const float x10 = c010 * (1.0f - ax) + c110 * ax;
    const float x01 = c001 * (1.0f - ax) + c101 * ax;
    const float x11 = c011 * (1.0f - ax) + c111 * ax;

    const float y0 = x00 * (1.0f - ay) + x10 * ay;
    const float y1 = x01 * (1.0f - ay) + x11 * ay;

    return y0 * (1.0f - az) + y1 * az;
}

kernel void main_warp(device const float *img   [[buffer(0)]],
                      device const float *field [[buffer(1)]],
                      device       float *outp  [[buffer(2)]],
                      constant Params    &P     [[buffer(3)]],
                      uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x);
    const int j = int(gid.y);
    const int k = int(gid.z);
    if(i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    const int vox = (k * P.sz.y + j) * P.sz.x + i;

    const float d0 = P.dir0.x; const float d1 = P.dir0.y; const float d2 = P.dir0.z;
    const float d3 = P.dir1.x; const float d4 = P.dir1.y; const float d5 = P.dir1.z;
    const float d6 = P.dir2.x; const float d7 = P.dir2.y; const float d8 = P.dir2.z;

    const float fi = float(i); const float fj = float(j); const float fk = float(k);

    const float x = d0 * P.res.x * fi + d1 * P.res.y * fj + d2 * P.res.z * fk;
    const float y = d3 * P.res.x * fi + d4 * P.res.y * fj + d5 * P.res.z * fk;
    const float z = d6 * P.res.x * fi + d7 * P.res.y * fj + d8 * P.res.z * fk;

    const float xw = x + field[3 * vox + 0];
    const float yw = y + field[3 * vox + 1];
    const float zw = z + field[3 * vox + 2];

    const float iw = (d0 * xw + d3 * yw + d6 * zw) / P.res.x;
    const float jw = (d1 * xw + d4 * yw + d7 * zw) / P.res.y;
    const float kw = (d2 * xw + d5 * yw + d8 * zw) / P.res.z;

    outp[vox] = sample_border(img, P, iw, jw, kw);
}
