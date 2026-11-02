// Shared pieces of compute_metric.cu, textually included into each metric shader
// by the build step (this file is prepended to each metric_*.metal before
// embedding, exactly as metric_common.wgsl is on the WebGPU side).
//
// CONTRACT, and it differs from the WGSL one: WGSL let the prelude reference the
// uniform block `MP` as a module-scope binding, so every metric shader had to name
// its uniform exactly that. MSL has no module-scope kernel arguments, so the block
// is passed EXPLICITLY to every helper - `sidx(MP, ...)`, `image_gradient(img, MP, ...)`.
// The naming requirement is gone; the argument is not optional.
//
// Window radii and thresholds are the reference's #defines:
//   WIN_RAD 5, WIN_RAD_Z 3, LIMCC 1e-10

#include <metal_stdlib>
using namespace metal;

constant int   WIN_RAD   = 5;
constant int   WIN_RAD_Z = 3;
constant float LIMCC     = 1e-10f;
constant float LIMCCSK   = 1e-5f;
constant float LIMCCJAC  = 1e-5f;

struct MetricParams {
    int4   sz;      // x, y, z, kernel_sz
    float4 spc;
    float4 dir0, dir1, dir2;
    float4 phase;   // phase vector; .w carries a scalar (e.g. CCSK's t)
    float4 newph;   // phase vector rotated by the direction matrix
    int4   axes;    // .x = phase axis, .y = phase_xyz
    float4 taps[8]; // smoothing kernel taps, up to 32
};

static inline int sidx(constant MetricParams &MP, int i, int j, int k) {
    return (k * MP.sz.y + j) * MP.sz.x + i;
}
static inline int vidx3(constant MetricParams &MP, int i, int j, int k, int c) {
    return ((k * MP.sz.y + j) * MP.sz.x + i) * 3 + c;
}

// ComputeImageGradient: central differences in index space, then rotated into
// world space by the direction matrix. Returns zero on the outer shell, exactly
// as the reference does.
static inline float3 image_gradient(device const float *img, constant MetricParams &MP,
                                    int i, int j, int k) {
    if(i == 0 || i == MP.sz.x - 1 ||
       j == 0 || j == MP.sz.y - 1 ||
       k == 0 || k == MP.sz.z - 1) {
        return float3(0.0f, 0.0f, 0.0f);
    }

    const float gx = 0.5f * (img[sidx(MP, i + 1, j, k)] - img[sidx(MP, i - 1, j, k)]) / MP.spc.x;

    // The reference loads the +1 neighbour, then subtracts the -1 neighbour and
    // scales in one expression; kept in that shape.
    float gy = img[sidx(MP, i, j + 1, k)];
    gy = 0.5f * (gy - img[sidx(MP, i, j - 1, k)]) / MP.spc.y;

    float gz = img[sidx(MP, i, j, k + 1)];
    gz = 0.5f * (gz - img[sidx(MP, i, j, k - 1)]) / MP.spc.z;

    return float3(
        MP.dir0.x * gx + MP.dir0.y * gy + MP.dir0.z * gz,
        MP.dir1.x * gx + MP.dir1.y * gy + MP.dir1.z * gz,
        MP.dir2.x * gx + MP.dir2.y * gy + MP.dir2.z * gz);
}
