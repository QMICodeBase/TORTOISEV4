// Port of resample_image.cu : ResampleImage_kernel
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
//    -fmad=true and contracts a*b+c into a single-rounded FMA, while WGSL exposes
//    no contraction control. Measured residual 1.45e-6; CUDA compiled with
//    -fmad=false disagrees with itself by 2.67e-6, so contraction - not this port -
//    sets the floor. Do not restore any claim of bit-exactness here.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;

struct Params {
    dsz   : vec4<i32>,   // data size x,y,z + Ncomponents
    vsz   : vec4<i32>,   // virtual (output) size x,y,z
    dspc  : vec4<f32>,
    vspc  : vec4<f32>,
    dorig : vec4<f32>,
    vorig : vec4<f32>,
    ddir0 : vec4<f32>, ddir1 : vec4<f32>, ddir2 : vec4<f32>,   // rows of data direction
    vdir0 : vec4<f32>, vdir1 : vec4<f32>, vdir2 : vec4<f32>,   // rows of virtual direction
};

@group(0) @binding(0) var<storage, read>       data : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
@group(0) @binding(2) var<uniform>             P    : Params;

fn didx(i : i32, j : i32, k : i32, m : i32) -> i32 {
    let nc = P.dsz.w;
    return ((k * P.dsz.y + j) * P.dsz.x + i) * nc + m;
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);
    if (i >= P.vsz.x || j >= P.vsz.y || k >= P.vsz.z) { return; }

    let nc = P.dsz.w;
    let oidx = ((k * P.vsz.y + j) * P.vsz.x + i) * nc;

    let D00 = P.ddir0.x; let D01 = P.ddir0.y; let D02 = P.ddir0.z;
    let D10 = P.ddir1.x; let D11 = P.ddir1.y; let D12 = P.ddir1.z;
    let D20 = P.ddir2.x; let D21 = P.ddir2.y; let D22 = P.ddir2.z;
    let V00 = P.vdir0.x; let V01 = P.vdir0.y; let V02 = P.vdir0.z;
    let V10 = P.vdir1.x; let V11 = P.vdir1.y; let V12 = P.vdir1.z;
    let V20 = P.vdir2.x; let V21 = P.vdir2.y; let V22 = P.vdir2.z;

    let fi = f32(i); let fj = f32(j); let fk = f32(k);

    let b1 = P.vorig.x - P.dorig.x + V00 * P.vspc.x * fi + V01 * P.vspc.y * fj + V02 * P.vspc.z * fk;
    let b2 = P.vorig.y - P.dorig.y + V10 * P.vspc.x * fi + V11 * P.vspc.y * fj + V12 * P.vspc.z * fk;
    let b3 = P.vorig.z - P.dorig.z + V20 * P.vspc.x * fi + V21 * P.vspc.y * fj + V22 * P.vspc.z * fk;

    // Per-term division, exactly as the CUDA source writes it.
    let iw = (D00 * b1) / P.dspc.x + (D10 * b2) / P.dspc.x + (D20 * b3) / P.dspc.x;
    let jw = (D01 * b1) / P.dspc.y + (D11 * b2) / P.dspc.y + (D21 * b3) / P.dspc.y;
    let kw = (D02 * b1) / P.dspc.z + (D12 * b2) / P.dspc.z + (D22 * b3) / P.dspc.z;

    if (iw < 0.0 || iw > f32(P.dsz.x - 1) ||
        jw < 0.0 || jw > f32(P.dsz.y - 1) ||
        kw < 0.0 || kw > f32(P.dsz.z - 1)) {
        for (var m = 0; m < nc; m = m + 1) { outp[oidx + m] = 0.0; }
        return;
    }

    let fx = i32(floor(iw)); let fy = i32(floor(jw)); let fz = i32(floor(kw));
    let cx = i32(ceil(iw));  let cy = i32(ceil(jw));  let cz = i32(ceil(kw));

    let xd = iw - f32(fx);
    let yd = jw - f32(fy);
    let zd = kw - f32(fz);

    for (var m = 0; m < nc; m = m + 1) {
        let ia1 = data[didx(fx, fy, fz, m)];
        let ja1 = data[didx(cx, fy, fz, m)];
        let ia2 = data[didx(fx, cy, fz, m)];
        let ja2 = data[didx(cx, cy, fz, m)];
        let ib1 = data[didx(fx, fy, cz, m)];
        let jb1 = data[didx(cx, fy, cz, m)];
        let ib2 = data[didx(fx, cy, cz, m)];
        let jb2 = data[didx(cx, cy, cz, m)];

        let i1 = ia1 * (1.0 - zd) + ib1 * zd;
        let i2 = ia2 * (1.0 - zd) + ib2 * zd;
        let j1 = ja1 * (1.0 - zd) + jb1 * zd;
        let j2 = ja2 * (1.0 - zd) + jb2 * zd;

        let w1 = i1 * (1.0 - yd) + i2 * yd;
        let w2 = j1 * (1.0 - yd) + j2 * yd;

        outp[oidx + m] = w1 * (1.0 - xd) + w2 * xd;
    }
}
