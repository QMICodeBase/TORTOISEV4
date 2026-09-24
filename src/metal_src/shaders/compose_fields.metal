// Port of shaders/compose_fields.wgsl, itself a port of
// cuda_image_utilities.cu : ComposeFields_kernel
//
// Composition of two displacement fields: walk to the point the main field maps
// to, sample the update field there, and return main + update expressed as a
// displacement from the original grid point.
//
// Boundary rule is a third distinct behaviour, different from both WarpImage and
// ResampleImage (CLAUDE.md 5.2): when the sampling point leaves [0, N-1] the output
// falls back to the MAIN field's displacement, it is neither zeroed nor blended.
//
// Interpolation order (lerp z, then y, then x) and the per-term division by
// spacing follow the reference exactly - both affect the last bits.

#include <metal_stdlib>
using namespace metal;

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];

struct Params {
    int4   sz;
    float4 spc;
    float4 dir0, dir1, dir2;
};

static inline int uidx(constant Params &P, int i, int j, int k, int m) {
    return ((k * P.sz.y + j) * P.sz.x + i) * 3 + m;
}

kernel void main_compose(device const float *mainf   [[buffer(0)]],
                         device const float *updatef [[buffer(1)]],
                         device       float *outp    [[buffer(2)]],
                         constant Params    &P       [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x);
    const int j = int(gid.y);
    const int k = int(gid.z);
    if(i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    const int n = uidx(P, i, j, k, 0);

    const float d0 = P.dir0.x; const float d1 = P.dir0.y; const float d2 = P.dir0.z;
    const float d3 = P.dir1.x; const float d4 = P.dir1.y; const float d5 = P.dir1.z;
    const float d6 = P.dir2.x; const float d7 = P.dir2.y; const float d8 = P.dir2.z;
    const float sx = P.spc.x;  const float sy = P.spc.y;  const float sz = P.spc.z;

    const float fi = float(i); const float fj = float(j); const float fk = float(k);

    float x[3];
    x[0] = d0 * sx * fi + d1 * sy * fj + d2 * sz * fk;
    x[1] = d3 * sx * fi + d4 * sy * fj + d5 * sz * fk;
    x[2] = d6 * sx * fi + d7 * sy * fj + d8 * sz * fk;

    float xp[3];
    xp[0] = x[0] + mainf[n];
    xp[1] = x[1] + mainf[n + 1];
    xp[2] = x[2] + mainf[n + 2];

    const float iw = (d0 * xp[0] + d3 * xp[1] + d6 * xp[2]) / sx;
    const float jw = (d1 * xp[0] + d4 * xp[1] + d7 * xp[2]) / sy;
    const float kw = (d2 * xp[0] + d5 * xp[1] + d8 * xp[2]) / sz;

    if(iw < 0.0f || iw > float(P.sz.x - 1) ||
       jw < 0.0f || jw > float(P.sz.y - 1) ||
       kw < 0.0f || kw > float(P.sz.z - 1)) {
        // outside: keep the main field's displacement unchanged
        outp[n]     = mainf[n];
        outp[n + 1] = mainf[n + 1];
        outp[n + 2] = mainf[n + 2];
        return;
    }

    const int fx = int(floor(iw)); const int fy = int(floor(jw)); const int fz = int(floor(kw));
    const int cx = int(ceil(iw));  const int cy = int(ceil(jw));  const int cz = int(ceil(kw));
    const float xd = iw - float(fx);
    const float yd = jw - float(fy);
    const float zd = kw - float(fz);

    for(int m = 0; m < 3; m = m + 1) {
        const float ia1 = updatef[uidx(P, fx, fy, fz, m)];
        const float ja1 = updatef[uidx(P, cx, fy, fz, m)];
        const float ia2 = updatef[uidx(P, fx, cy, fz, m)];
        const float ja2 = updatef[uidx(P, cx, cy, fz, m)];
        const float ib1 = updatef[uidx(P, fx, fy, cz, m)];
        const float jb1 = updatef[uidx(P, cx, fy, cz, m)];
        const float ib2 = updatef[uidx(P, fx, cy, cz, m)];
        const float jb2 = updatef[uidx(P, cx, cy, cz, m)];

        const float i1 = ia1 * (1.0f - zd) + ib1 * zd;
        const float i2 = ia2 * (1.0f - zd) + ib2 * zd;
        const float j1 = ja1 * (1.0f - zd) + jb1 * zd;
        const float j2 = ja2 * (1.0f - zd) + jb2 * zd;

        const float w1 = i1 * (1.0f - yd) + i2 * yd;
        const float w2 = j1 * (1.0f - yd) + j2 * yd;

        const float update = w1 * (1.0f - xd) + w2 * xd;
        outp[n + m] = xp[m] + update - x[m];
    }
}
