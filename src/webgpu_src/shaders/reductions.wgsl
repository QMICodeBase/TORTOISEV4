// Port of the reduction kernels in cuda_image_utilities.cu / compute_metric.cu:
// ScalarFindSum, ScalarFindSumSq, ScalarFindMax and FieldFindMaxLocalNorm.
//
// Summation is NOT associative in floating point, so unlike the elementwise
// kernels the *order* here is part of the specification. CUDA uses bSize=1024
// threads across gSize=24 blocks: thread t of block b starts at t + b*1024 and
// strides by 24*1024, then the block tree-reduces by halving from 512. That
// exact shape is reproduced below (WGX defaults to 1024, dispatched as 24
// workgroups) so the partial sums pair up identically.
//
// Max reductions are order-independent and exact, so those would match under any
// tiling; they keep the same shape only for consistency.

override WGX : u32 = 1024u;
override WGY : u32 = 1u;      // declared so the shared pipeline helper can always
override WGZ : u32 = 1u;      // supply the three workgroup dims
override OP  : u32 = 0u;      // 0 = sum, 1 = sum of squares, 2 = max, 3 = max local field norm, 4 = min

struct Params {
    n     : vec4<i32>,        // element count (n.x), stride in elements (n.y)
    spc   : vec4<f32>,        // voxel spacing, for the field-norm variant
};

@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             P   : Params;

var<workgroup> sh : array<f32, 1024>;

@compute @workgroup_size(WGX, WGY, WGZ)
fn reduce(@builtin(local_invocation_id) lid : vec3<u32>,
          @builtin(workgroup_id) wid : vec3<u32>,
          @builtin(num_workgroups) nwg : vec3<u32>) {
    let t   = lid.x;
    let gth = i32(t + wid.x * WGX);
    let stride = i32(WGX * nwg.x);
    let n = P.n.x;

    // CUDA seeds the max variants with -1, not -inf; a norm is never negative so
    // this only matters for an empty range, where it must be reproduced.
    //
    // ScalarFindMin seeds `float mn = 1E100` (cuda_image_utilities.cu:139), which
    // overflows f32 and is therefore +inf. WGSL cannot express inf in a
    // const-expression ("value inf cannot be represented as 'f32'"), so the seed is
    // the largest finite f32 instead (written as its exact value - 3.40282347e38
    // rounds UP past f32 max and Tint rejects it). min(FLT_MAX, x) == x for every
    // finite x, so
    // the result is identical for all real data; it would differ from the reference
    // only for an empty range or an all-+inf input, neither of which occurs (the
    // caller always passes n_elements > 0).
    var acc = 0.0;
    if (OP == 4u) { acc = 3.4028234663852886e38; }   // FLT_MAX
    else if (OP >= 2u) { acc = -1.0; }

    var i = gth;
    loop {
        if (i >= n) { break; }
        if (OP == 0u) {
            acc = acc + src[i];
        } else if (OP == 1u) {
            acc = acc + src[i] * src[i];
        } else if (OP == 2u) {
            let v = src[i];
            if (v > acc) { acc = v; }
        } else if (OP == 4u) {
            let v = src[i];
            if (v < acc) { acc = v; }
        } else {
            let x = src[3 * i]     / P.spc.x;
            let y = src[3 * i + 1] / P.spc.y;
            let z = src[3 * i + 2] / P.spc.z;
            let v = sqrt(x * x + y * y + z * z);
            if (v > acc) { acc = v; }
        }
        i = i + stride;
    }

    sh[t] = acc;
    workgroupBarrier();

    var size = WGX / 2u;
    loop {
        if (size == 0u) { break; }
        if (t < size) {
            if (OP < 2u) {
                sh[t] = sh[t] + sh[t + size];
            } else if (OP == 4u) {
                if (sh[t + size] < sh[t]) { sh[t] = sh[t + size]; }
            } else if (sh[t + size] > sh[t]) {
                sh[t] = sh[t + size];
            }
        }
        workgroupBarrier();
        size = size / 2u;
    }

    if (t == 0u) { dst[wid.x] = sh[0]; }
}
