// Port of shaders/resample_image.wgsl, itself a port of
// resample_image.cu : ResampleImage_kernel
//
// Faithfulness notes:
//  * boundary rule is all-or-nothing: if the sample point falls outside
//    [0, N-1] on ANY axis the whole output voxel is zero (resample_image.cu:82).
//    This differs from WarpImage, which blends per-neighbour against a zero
//    border - see CLAUDE.md 5.2.
//  * the interpolation is done in CUDA's exact order (lerp z, then y, then x)
//    and the coordinate expression divides each of the three terms by the
//    spacing separately, as the original does. Both matter for bit-exactness:
//    reassociating float adds changes the last ulp.
//  * no hardware filtering is involved here, so this kernel has no sampler
//    inaccuracy to account for. It is NOT bit-exact even so: nvcc defaults to
//    -fmad=true and contracts a*b+c into a single-rounded FMA, while neither WGSL nor MSL
//    exposes contraction control. Measured residual 1.45e-6; CUDA compiled with
//    -fmad=false disagrees with itself by 2.67e-6, so contraction - not this port -
//    sets the floor. Do not restore any claim of bit-exactness here.

#include <metal_stdlib>
using namespace metal;

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];

struct Params {
    int4   dsz;     // data size x,y,z + Ncomponents
    int4   vsz;     // virtual (output) size x,y,z
    float4 dspc;
    float4 vspc;
    float4 dorig;
    float4 vorig;
    float4 ddir0, ddir1, ddir2;   // rows of data direction
    float4 vdir0, vdir1, vdir2;   // rows of virtual direction
};

static inline int didx(constant Params &P, int i, int j, int k, int m) {
    const int nc = P.dsz.w;
    return ((k * P.dsz.y + j) * P.dsz.x + i) * nc + m;
}

kernel void main_resample(device const float *data [[buffer(0)]],
                          device       float *outp [[buffer(1)]],
                          constant Params    &P    [[buffer(2)]],
                          uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x);
    const int j = int(gid.y);
    const int k = int(gid.z);
    if(i >= P.vsz.x || j >= P.vsz.y || k >= P.vsz.z) { return; }

    const int nc = P.dsz.w;
    const int oidx = ((k * P.vsz.y + j) * P.vsz.x + i) * nc;

    const float D00 = P.ddir0.x; const float D01 = P.ddir0.y; const float D02 = P.ddir0.z;
    const float D10 = P.ddir1.x; const float D11 = P.ddir1.y; const float D12 = P.ddir1.z;
    const float D20 = P.ddir2.x; const float D21 = P.ddir2.y; const float D22 = P.ddir2.z;
    const float V00 = P.vdir0.x; const float V01 = P.vdir0.y; const float V02 = P.vdir0.z;
    const float V10 = P.vdir1.x; const float V11 = P.vdir1.y; const float V12 = P.vdir1.z;
    const float V20 = P.vdir2.x; const float V21 = P.vdir2.y; const float V22 = P.vdir2.z;

    const float fi = float(i); const float fj = float(j); const float fk = float(k);

    const float b1 = P.vorig.x - P.dorig.x + V00 * P.vspc.x * fi + V01 * P.vspc.y * fj + V02 * P.vspc.z * fk;
    const float b2 = P.vorig.y - P.dorig.y + V10 * P.vspc.x * fi + V11 * P.vspc.y * fj + V12 * P.vspc.z * fk;
    const float b3 = P.vorig.z - P.dorig.z + V20 * P.vspc.x * fi + V21 * P.vspc.y * fj + V22 * P.vspc.z * fk;

    // Per-term division, exactly as the CUDA source writes it.
    const float iw = (D00 * b1) / P.dspc.x + (D10 * b2) / P.dspc.x + (D20 * b3) / P.dspc.x;
    const float jw = (D01 * b1) / P.dspc.y + (D11 * b2) / P.dspc.y + (D21 * b3) / P.dspc.y;
    const float kw = (D02 * b1) / P.dspc.z + (D12 * b2) / P.dspc.z + (D22 * b3) / P.dspc.z;

    if(iw < 0.0f || iw > float(P.dsz.x - 1) ||
       jw < 0.0f || jw > float(P.dsz.y - 1) ||
       kw < 0.0f || kw > float(P.dsz.z - 1)) {
        for(int m = 0; m < nc; m = m + 1) { outp[oidx + m] = 0.0f; }
        return;
    }

    const int fx = int(floor(iw)); const int fy = int(floor(jw)); const int fz = int(floor(kw));
    const int cx = int(ceil(iw));  const int cy = int(ceil(jw));  const int cz = int(ceil(kw));

    const float xd = iw - float(fx);
    const float yd = jw - float(fy);
    const float zd = kw - float(fz);

    for(int m = 0; m < nc; m = m + 1) {
        const float ia1 = data[didx(P, fx, fy, fz, m)];
        const float ja1 = data[didx(P, cx, fy, fz, m)];
        const float ia2 = data[didx(P, fx, cy, fz, m)];
        const float ja2 = data[didx(P, cx, cy, fz, m)];
        const float ib1 = data[didx(P, fx, fy, cz, m)];
        const float jb1 = data[didx(P, cx, fy, cz, m)];
        const float ib2 = data[didx(P, fx, cy, cz, m)];
        const float jb2 = data[didx(P, cx, cy, cz, m)];

        const float i1 = ia1 * (1.0f - zd) + ib1 * zd;
        const float i2 = ia2 * (1.0f - zd) + ib2 * zd;
        const float j1 = ja1 * (1.0f - zd) + jb1 * zd;
        const float j2 = ja2 * (1.0f - zd) + jb2 * zd;

        const float w1 = i1 * (1.0f - yd) + i2 * yd;
        const float w2 = j1 * (1.0f - yd) + j2 * yd;

        outp[oidx + m] = w1 * (1.0f - xd) + w2 * xd;
    }
}
