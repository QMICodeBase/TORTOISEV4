// Port of gaussian_smooth_image.cu : the three separable passes plus
// AdjustFieldBoundary_kernel.
//
// Two faithfulness points:
//
//  * The CUDA kernels stage each line in shared memory, but that is a
//    performance detail - the arithmetic is a plain 1D convolution. Reading taps
//    straight from the buffer gives identical results and avoids tying the
//    workgroup size to the image dimension, which would break for any axis
//    longer than the 256-invocation portable limit. The accumulation order
//    (tap 0 upward) is preserved, which is what actually affects rounding.
//
//  * Out-of-range taps are SKIPPED, not clamped and not zero-padded, and the
//    kernel is NOT renormalised at the boundary (gaussian_smooth_image.cu:60).
//    So edge voxels are convolved with a truncated kernel summing to less than
//    one, and get systematically darker. That is the reference behaviour and is
//    reproduced exactly.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;
override AXIS : u32 = 0u;          // 0 = x, 1 = y, 2 = z

struct Params {
    sz     : vec4<i32>,            // x, y, z, Ncomponents
    ksz    : vec4<i32>,            // kernel_sz, (weights below are for the boundary pass)
    wts    : vec4<f32>,            // weight1, weight2
    kernel : array<vec4<f32>, 8>,  // up to 32 taps
};

@group(0) @binding(0) var<storage, read>       src  : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst  : array<f32>;
@group(0) @binding(2) var<uniform>             P    : Params;

fn tap(i : i32) -> f32 {
    let v = P.kernel[i / 4];
    switch (i % 4) {
        case 0: { return v.x; }
        case 1: { return v.y; }
        case 2: { return v.z; }
        default: { return v.w; }
    }
}

fn idx(i : i32, j : i32, k : i32, c : i32) -> i32 {
    return ((k * P.sz.y + j) * P.sz.x + i) * P.sz.w + c;
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);
    if (i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    let mask = P.ksz.x;
    let half = mask / 2;

    // Length of the axis being convolved, and this voxel's position along it.
    var extent = P.sz.x;
    var pos    = i;
    if (AXIS == 1u) { extent = P.sz.y; pos = j; }
    if (AXIS == 2u) { extent = P.sz.z; pos = k; }

    for (var c = 0; c < P.sz.w; c = c + 1) {
        var val = 0.0;
        for (var t = 0; t < mask; t = t + 1) {
            let p = pos + t - half;
            if (p >= 0 && p < extent) {          // out-of-range taps contribute nothing
                var s = 0.0;
                if (AXIS == 0u)      { s = src[idx(p, j, k, c)]; }
                else if (AXIS == 1u) { s = src[idx(i, p, k, c)]; }
                else                 { s = src[idx(i, j, p, c)]; }
                val = val + s * tap(t);
            }
        }
        dst[idx(i, j, k, c)] = val;
    }
}

// AdjustFieldBoundary_kernel: zero the outer shell, blend the interior.
// Runs in place on the smoothed field, reading the original for the blend.
@compute @workgroup_size(WGX, WGY, WGZ)
fn adjust_boundary(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);
    if (i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    let edge = (i == 0 || i == P.sz.x - 1 ||
                j == 0 || j == P.sz.y - 1 ||
                k == 0 || k == P.sz.z - 1);

    for (var c = 0; c < P.sz.w; c = c + 1) {
        let n = idx(i, j, k, c);
        if (edge) {
            dst[n] = 0.0;
        } else {
            dst[n] = dst[n] * P.wts.x + src[n] * P.wts.y;
        }
    }
}
