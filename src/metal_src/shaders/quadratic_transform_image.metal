// Port of shaders/quadratic_transform_image.wgsl, itself a port of
// quadratic_transform_image.cu : QuadraticTransformImage_kernel
//
// Unlike WarpImage, this kernel guards the domain BEFORE sampling
// (quadratic_transform_image.cu:79): if the sample point leaves [0, N-1] on any
// axis the output voxel is left at its memset zero. So the border address mode
// never materially engages, and an all-or-nothing guard is the faithful rule
// here - see CLAUDE.md 5.2.

#include <metal_stdlib>
using namespace metal;

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];
// See warp_image.metal for the measurement behind this default: exact fp32 weights
// are empirically closer to CUDA than the documented 1/256 quantisation model.
constant uint TEXFILTER_QUANTISE [[function_constant(3)]];

struct Params {
    int4   tsz;            // target size
    int4   isz;            // source image size
    float4 smat[3];        // rows 0..2 of the target index->world matrix
    float4 smat_inv[3];    // rows 0..2 of the world->source index matrix
    float4 rot[3];         // rotation matrix rows
    float4 par[6];         // 24 quadratic parameters
    int4   flags;          // phase, do_cubic
};

static inline float par(constant Params &P, int i) {
    const float4 v = P.par[i / 4];
    switch(i % 4) {
        case 0:  return v.x;
        case 1:  return v.y;
        case 2:  return v.z;
        default: return v.w;
    }
}

static inline float fetch(device const float *img, constant Params &P, int i, int j, int k) {
    if(i < 0 || j < 0 || k < 0 || i >= P.isz.x || j >= P.isz.y || k >= P.isz.z) {
        return 0.0f;
    }
    return img[(k * P.isz.y + j) * P.isz.x + i];
}

static inline float weight(float a) {
    if(TEXFILTER_QUANTISE != 0u) { return floor(a * 256.0f) / 256.0f; }
    return a;
}

static inline float sample_trilinear(device const float *img, constant Params &P,
                                     float iw, float jw, float kw) {
    const float fx = floor(iw); const float fy = floor(jw); const float fz = floor(kw);
    const float ax = weight(iw - fx); const float ay = weight(jw - fy); const float az = weight(kw - fz);
    const int ix = int(fx); const int iy = int(fy); const int iz = int(fz);

    const float x00 = fetch(img, P, ix, iy, iz)         * (1.0f - ax) + fetch(img, P, ix + 1, iy, iz)         * ax;
    const float x10 = fetch(img, P, ix, iy + 1, iz)     * (1.0f - ax) + fetch(img, P, ix + 1, iy + 1, iz)     * ax;
    const float x01 = fetch(img, P, ix, iy, iz + 1)     * (1.0f - ax) + fetch(img, P, ix + 1, iy, iz + 1)     * ax;
    const float x11 = fetch(img, P, ix, iy + 1, iz + 1) * (1.0f - ax) + fetch(img, P, ix + 1, iy + 1, iz + 1) * ax;

    const float y0 = x00 * (1.0f - ay) + x10 * ay;
    const float y1 = x01 * (1.0f - ay) + x11 * ay;
    return y0 * (1.0f - az) + y1 * az;
}

kernel void main_quadratic(device const float *img  [[buffer(0)]],
                           device       float *outp [[buffer(1)]],
                           constant Params    &P    [[buffer(2)]],
                           uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x);
    const int j = int(gid.y);
    const int k = int(gid.z);
    if(i >= P.tsz.x || j >= P.tsz.y || k >= P.tsz.z) { return; }

    const float fi = float(i); const float fj = float(j); const float fk = float(k);

    float x = P.smat[0].x * fi + P.smat[0].y * fj + P.smat[0].z * fk + P.smat[0].w;
    float y = P.smat[1].x * fi + P.smat[1].y * fj + P.smat[1].z * fk + P.smat[1].w;
    float z = P.smat[2].x * fi + P.smat[2].y * fj + P.smat[2].z * fk + P.smat[2].w;

    x = x - par(P, 21);
    y = y - par(P, 22);
    z = z - par(P, 23);

    float x1 = P.rot[0].x * x + P.rot[0].y * y + P.rot[0].z * z + par(P, 0);
    float y1 = P.rot[1].x * x + P.rot[1].y * y + P.rot[1].z * z + par(P, 1);
    float z1 = P.rot[2].x * x + P.rot[2].y * y + P.rot[2].z * z + par(P, 2);

    const float new_phase_coord =
        par(P, 6) * x1 + par(P, 7) * y1 + par(P, 8) * z1 +
        par(P, 9) * x1 * y1 + par(P, 10) * x1 * z1 + par(P, 11) * y1 * z1 +
        par(P, 12) * (x1 * x1 - y1 * y1) + par(P, 13) * (2.0f * z1 * z1 - x1 * x1 - y1 * y1);

    float cubic_change = 0.0f;
    if(P.flags.y != 0) {
        cubic_change =
            par(P, 14) * x1 * y1 * z1 +
            par(P, 15) * z1 * (x1 * x1 - y1 * y1) +
            par(P, 16) * x1 * (4.0f * z1 * z1 - x1 * x1 - y1 * y1) +
            par(P, 17) * y1 * (4.0f * z1 * z1 - x1 * x1 - y1 * y1) +
            par(P, 18) * x1 * (x1 * x1 - 3.0f * y1 * y1) +
            par(P, 19) * y1 * (3.0f * x1 * x1 - y1 * y1) +
            par(P, 20) * z1 * (2.0f * z1 * z1 - 3.0f * x1 * x1 - 3.0f * y1 * y1);
    }

    if(P.flags.x == 0) { x1 = new_phase_coord + cubic_change; }
    if(P.flags.x == 1) { y1 = new_phase_coord + cubic_change; }
    if(P.flags.x == 2) { z1 = new_phase_coord + cubic_change; }

    const float iw = P.smat_inv[0].x * x1 + P.smat_inv[0].y * y1 + P.smat_inv[0].z * z1 + P.smat_inv[0].w;
    const float jw = P.smat_inv[1].x * x1 + P.smat_inv[1].y * y1 + P.smat_inv[1].z * z1 + P.smat_inv[1].w;
    const float kw = P.smat_inv[2].x * x1 + P.smat_inv[2].y * y1 + P.smat_inv[2].z * z1 + P.smat_inv[2].w;

    if(iw >= 0.0f && iw <= float(P.isz.x - 1) &&
       jw >= 0.0f && jw <= float(P.isz.y - 1) &&
       kw >= 0.0f && kw <= float(P.isz.z - 1)) {
        outp[(k * P.tsz.y + j) * P.tsz.x + i] = sample_trilinear(img, P, iw, jw, kw);
    }
}
