// Port of the two kernels InvertField_cuda drives that are not already ported:
// ComputeFieldLocalNormImage and UpdateInvertField_kernel. The composition step
// reuses compose_fields.wgsl and the convergence reductions reuse
// reductions.wgsl, so only these two are new.
//
// NegateImage_kernel is folded into the norm pass: the reference computes the
// norm image from the composed field, then negates the composed field in place,
// and nothing reads the un-negated field in between.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;

struct Params {
    sz  : vec4<i32>,
    spc : vec4<f32>,
    ctl : vec4<f32>,      // epsilon, max_error_norm
};

@group(0) @binding(0) var<storage, read_write> composed : array<f32>;   // 3-component
@group(0) @binding(1) var<storage, read_write> scale    : array<f32>;   // scalar norm image
@group(0) @binding(2) var<storage, read_write> outp     : array<f32>;   // 3-component, in place
@group(0) @binding(3) var<uniform>             P        : Params;

fn vidx(i : i32, j : i32, k : i32) -> i32 {
    return ((k * P.sz.y + j) * P.sz.x + i) * 3;
}
fn sidx(i : i32, j : i32, k : i32) -> i32 {
    return (k * P.sz.y + j) * P.sz.x + i;
}

// ComputeFieldLocalNormImage, then NegateImage_kernel on the composed field.
// Note the reference divides by the spacing twice per term rather than squaring
// it once; kept as written because the rounding differs.
@compute @workgroup_size(WGX, WGY, WGZ)
fn local_norm_and_negate(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
    if (i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    let n = vidx(i, j, k);
    let cx = composed[n]; let cy = composed[n + 1]; let cz = composed[n + 2];

    var nrm = cx * cx / P.spc.x / P.spc.x +
              cy * cy / P.spc.y / P.spc.y +
              cz * cz / P.spc.z / P.spc.z;
    nrm = sqrt(nrm);
    scale[sidx(i, j, k)] = nrm;

    composed[n]     = -cx;
    composed[n + 1] = -cy;
    composed[n + 2] = -cz;
}

// UpdateInvertField_kernel: clamp the step where the local norm is large, take an
// epsilon-scaled step, and force the outer shell to zero.
@compute @workgroup_size(WGX, WGY, WGZ)
fn update_invert_field(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
    if (i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    let n = vidx(i, j, k);
    let eps = P.ctl.x;
    let maxerr = P.ctl.y;

    var ux = composed[n];
    var uy = composed[n + 1];
    var uz = composed[n + 2];

    let scaledNorm = scale[sidx(i, j, k)];
    if (scaledNorm > eps * maxerr) {
        let f = eps * maxerr / scaledNorm;
        ux = ux * f;
        uy = uy * f;
        uz = uz * f;
    }

    ux = outp[n]     + ux * eps;
    uy = outp[n + 1] + uy * eps;
    uz = outp[n + 2] + uz * eps;

    if (i == 0 || i == P.sz.x - 1 ||
        j == 0 || j == P.sz.y - 1 ||
        k == 0 || k == P.sz.z - 1) {
        ux = 0.0; uy = 0.0; uz = 0.0;
    }

    outp[n]     = ux;
    outp[n + 1] = uy;
    outp[n + 2] = uz;
}
