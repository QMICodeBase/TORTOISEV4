// Port of quadratic_transform_image.cu : QuadraticTransformImage_kernel
//
// Unlike WarpImage, this kernel guards the domain BEFORE sampling
// (quadratic_transform_image.cu:79): if the sample point leaves [0, N-1] on any
// axis the output voxel is left at its memset zero. So the border address mode
// never materially engages, and an all-or-nothing guard is the faithful rule
// here - see CLAUDE.md 5.2.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;
// See warp_image.wgsl for the measurement behind this default: exact fp32 weights
// are empirically closer to CUDA than the documented 1/256 quantisation model.
override TEXFILTER_QUANTISE : bool = false;

struct Params {
    tsz      : vec4<i32>,      // target size
    isz      : vec4<i32>,      // source image size
    smat     : array<vec4<f32>, 3>,      // rows 0..2 of the target index->world matrix
    smat_inv : array<vec4<f32>, 3>,      // rows 0..2 of the world->source index matrix
    rot      : array<vec4<f32>, 3>,      // rotation matrix rows
    par      : array<vec4<f32>, 6>,      // 24 quadratic parameters
    flags    : vec4<i32>,                // phase, do_cubic
};

@group(0) @binding(0) var<storage, read>       img  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
@group(0) @binding(2) var<uniform>             P    : Params;

fn par(i : i32) -> f32 {
    let v = P.par[i / 4];
    switch (i % 4) {
        case 0: { return v.x; }
        case 1: { return v.y; }
        case 2: { return v.z; }
        default: { return v.w; }
    }
}

fn fetch(i : i32, j : i32, k : i32) -> f32 {
    if (i < 0 || j < 0 || k < 0 || i >= P.isz.x || j >= P.isz.y || k >= P.isz.z) {
        return 0.0;
    }
    return img[(k * P.isz.y + j) * P.isz.x + i];
}

fn weight(a : f32) -> f32 {
    if (TEXFILTER_QUANTISE) { return floor(a * 256.0) / 256.0; }
    return a;
}

fn sample_trilinear(iw : f32, jw : f32, kw : f32) -> f32 {
    let fx = floor(iw); let fy = floor(jw); let fz = floor(kw);
    let ax = weight(iw - fx); let ay = weight(jw - fy); let az = weight(kw - fz);
    let ix = i32(fx); let iy = i32(fy); let iz = i32(fz);

    let x00 = fetch(ix, iy, iz)         * (1.0 - ax) + fetch(ix + 1, iy, iz)         * ax;
    let x10 = fetch(ix, iy + 1, iz)     * (1.0 - ax) + fetch(ix + 1, iy + 1, iz)     * ax;
    let x01 = fetch(ix, iy, iz + 1)     * (1.0 - ax) + fetch(ix + 1, iy, iz + 1)     * ax;
    let x11 = fetch(ix, iy + 1, iz + 1) * (1.0 - ax) + fetch(ix + 1, iy + 1, iz + 1) * ax;

    let y0 = x00 * (1.0 - ay) + x10 * ay;
    let y1 = x01 * (1.0 - ay) + x11 * ay;
    return y0 * (1.0 - az) + y1 * az;
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);
    if (i >= P.tsz.x || j >= P.tsz.y || k >= P.tsz.z) { return; }

    let fi = f32(i); let fj = f32(j); let fk = f32(k);

    var x = P.smat[0].x * fi + P.smat[0].y * fj + P.smat[0].z * fk + P.smat[0].w;
    var y = P.smat[1].x * fi + P.smat[1].y * fj + P.smat[1].z * fk + P.smat[1].w;
    var z = P.smat[2].x * fi + P.smat[2].y * fj + P.smat[2].z * fk + P.smat[2].w;

    x = x - par(21);
    y = y - par(22);
    z = z - par(23);

    var x1 = P.rot[0].x * x + P.rot[0].y * y + P.rot[0].z * z + par(0);
    var y1 = P.rot[1].x * x + P.rot[1].y * y + P.rot[1].z * z + par(1);
    var z1 = P.rot[2].x * x + P.rot[2].y * y + P.rot[2].z * z + par(2);

    let new_phase_coord =
        par(6) * x1 + par(7) * y1 + par(8) * z1 +
        par(9) * x1 * y1 + par(10) * x1 * z1 + par(11) * y1 * z1 +
        par(12) * (x1 * x1 - y1 * y1) + par(13) * (2.0 * z1 * z1 - x1 * x1 - y1 * y1);

    var cubic_change = 0.0;
    if (P.flags.y != 0) {
        cubic_change =
            par(14) * x1 * y1 * z1 +
            par(15) * z1 * (x1 * x1 - y1 * y1) +
            par(16) * x1 * (4.0 * z1 * z1 - x1 * x1 - y1 * y1) +
            par(17) * y1 * (4.0 * z1 * z1 - x1 * x1 - y1 * y1) +
            par(18) * x1 * (x1 * x1 - 3.0 * y1 * y1) +
            par(19) * y1 * (3.0 * x1 * x1 - y1 * y1) +
            par(20) * z1 * (2.0 * z1 * z1 - 3.0 * x1 * x1 - 3.0 * y1 * y1);
    }

    if (P.flags.x == 0) { x1 = new_phase_coord + cubic_change; }
    if (P.flags.x == 1) { y1 = new_phase_coord + cubic_change; }
    if (P.flags.x == 2) { z1 = new_phase_coord + cubic_change; }

    let iw = P.smat_inv[0].x * x1 + P.smat_inv[0].y * y1 + P.smat_inv[0].z * z1 + P.smat_inv[0].w;
    let jw = P.smat_inv[1].x * x1 + P.smat_inv[1].y * y1 + P.smat_inv[1].z * z1 + P.smat_inv[1].w;
    let kw = P.smat_inv[2].x * x1 + P.smat_inv[2].y * y1 + P.smat_inv[2].z * z1 + P.smat_inv[2].w;

    if (iw >= 0.0 && iw <= f32(P.isz.x - 1) &&
        jw >= 0.0 && jw <= f32(P.isz.y - 1) &&
        kw >= 0.0 && kw <= f32(P.isz.z - 1)) {
        outp[(k * P.tsz.y + j) * P.tsz.x + i] = sample_trilinear(iw, jw, kw);
    }
}
