// Port of the elementwise kernels in cuda_image_utilities.cu.
//
// One module, several entry points: they share the indexing helper and the
// parameter block, and the pipeline cache keys on the entry point, so this costs
// nothing at runtime and keeps the per-kernel boilerplate down.
//
// Every kernel here is a pure map over voxels, so the CUDA tiling (which is the
// swapped-launch grid/block described in CLAUDE.md 0.1) has no bearing on results.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;

struct Params {
    sz    : vec4<i32>,     // x, y, z, Ncomponents
    fa    : vec4<f32>,     // scalar arguments (meaning is per-kernel)
    fb    : vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       a    : array<f32>;
@group(0) @binding(1) var<storage, read>       b    : array<f32>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
@group(0) @binding(3) var<uniform>             P    : Params;
// Second in-place field, used only by constrain_def_fields.
@group(0) @binding(4) var<storage, read_write> uf   : array<f32>;

fn base(i : i32, j : i32, k : i32) -> i32 {
    return ((k * P.sz.y + j) * P.sz.x + i) * P.sz.w;
}

fn inside(gid : vec3<u32>) -> bool {
    return i32(gid.x) < P.sz.x && i32(gid.y) < P.sz.y && i32(gid.z) < P.sz.z;
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn add_images(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (!inside(gid)) { return; }
    let n = base(i32(gid.x), i32(gid.y), i32(gid.z));
    for (var m = 0; m < P.sz.w; m = m + 1) { outp[n + m] = a[n + m] + b[n + m]; }
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn multiply_images(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (!inside(gid)) { return; }
    let n = base(i32(gid.x), i32(gid.y), i32(gid.z));
    for (var m = 0; m < P.sz.w; m = m + 1) { outp[n + m] = a[n + m] * b[n + m]; }
}

// fa.x = factor
@compute @workgroup_size(WGX, WGY, WGZ)
fn multiply_image(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (!inside(gid)) { return; }
    let n = base(i32(gid.x), i32(gid.y), i32(gid.z));
    for (var m = 0; m < P.sz.w; m = m + 1) { outp[n + m] = a[n + m] * P.fa.x; }
}

// Rescale [img_min,img_max] -> [low_val,up_val]. Written in the reference's own
// three-term form rather than the algebraically tidier one, because the grouping
// changes the rounding. fa = (low_val, up_val, img_min, img_max)
@compute @workgroup_size(WGX, WGY, WGZ)
fn preprocess_image(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (!inside(gid)) { return; }
    let n = base(i32(gid.x), i32(gid.y), i32(gid.z));
    let low = P.fa.x; let up = P.fa.y; let imin = P.fa.z; let imax = P.fa.w;
    outp[n] = (up - low) / (imax - imin) * a[n]
              - imin * (up - low) / (imax - imin)
              + low;
}

// Project each vector onto the phase-encode direction. fa.xyz = phase
@compute @workgroup_size(WGX, WGY, WGZ)
fn restrict_phase(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (!inside(gid)) { return; }
    let n = base(i32(gid.x), i32(gid.y), i32(gid.z));
    let vx = outp[n]; let vy = outp[n + 1]; let vz = outp[n + 2];
    var nrm = vx * vx + vy * vy + vz * vz;
    nrm = sqrt(nrm);
    if (nrm != 0.0) {
        let ux = vx / nrm; let uy = vy / nrm; let uz = vz / nrm;
        let d = ux * P.fa.x + uy * P.fa.y + uz * P.fa.z;
        outp[n]     = P.fa.x * nrm * d;
        outp[n + 1] = P.fa.y * nrm * d;
        outp[n + 2] = P.fa.z * nrm * d;
    }
}

// Make the two fields exact opposites: u = (u-d)/2, d = -u. In place on both:
// `uf` (binding 4) is the up field and `outp` (binding 2) the down field.
@compute @workgroup_size(WGX, WGY, WGZ)
fn constrain_def_fields(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (!inside(gid)) { return; }
    let n = base(i32(gid.x), i32(gid.y), i32(gid.z));
    for (var v = 0; v < 3; v = v + 1) {
        let val = (uf[n + v] - outp[n + v]) * 0.5;
        uf[n + v]   = val;
        outp[n + v] = -val;
    }
}

// In-place scale, used by ScaleUpdateField after its max reduction. fa.x = factor
@compute @workgroup_size(WGX, WGY, WGZ)
fn multiply_image_inplace(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (!inside(gid)) { return; }
    let n = base(i32(gid.x), i32(gid.y), i32(gid.z));
    for (var m = 0; m < P.sz.w; m = m + 1) { outp[n + m] = outp[n + m] * P.fa.x; }
}

// total += to_add * factor. The reference computes factor = weight/magnitude on
// the HOST and passes one float (cuda_image_utilities.cu:328); doing the divide
// here instead would regroup the arithmetic and change the last bits.
@compute @workgroup_size(WGX, WGY, WGZ)
fn add_to_update_field(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (!inside(gid)) { return; }
    let n = base(i32(gid.x), i32(gid.y), i32(gid.z));
    for (var m = 0; m < P.sz.w; m = m + 1) {
        outp[n + m] = outp[n + m] + a[n + m] * P.fa.x;
    }
}
