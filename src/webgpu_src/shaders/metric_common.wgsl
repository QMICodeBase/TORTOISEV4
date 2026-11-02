// Shared pieces of compute_metric.cu, textually included into each metric
// shader by the build step (WGSL has no #include, so this file is prepended to
// each metric_*.wgsl before embedding).
//
// CONTRACT: this prelude references `MP` for its uniform block, so every metric
// shader MUST declare its uniform with exactly that name. A new metric shader
// naming it otherwise fails with an "unresolved identifier" pointing inside this
// file rather than at the real cause.
//
// Window radii and thresholds are the reference's #defines:
//   WIN_RAD 5, WIN_RAD_Z 3, LIMCC 1e-10

const WIN_RAD   : i32 = 5;
const WIN_RAD_Z : i32 = 3;
const LIMCC     : f32 = 1e-10;
const LIMCCSK   : f32 = 1e-5;
const LIMCCJAC  : f32 = 1e-5;

struct MetricParams {
    sz    : vec4<i32>,     // x, y, z, kernel_sz
    spc   : vec4<f32>,
    dir0  : vec4<f32>, dir1 : vec4<f32>, dir2 : vec4<f32>,
    phase : vec4<f32>,     // phase vector; .w carries a scalar (e.g. CCSK's t)
    newph : vec4<f32>,     // phase vector rotated by the direction matrix
    axes  : vec4<i32>,     // .x = phase axis, .y = phase_xyz
    taps  : array<vec4<f32>, 8>,   // smoothing kernel taps, up to 32
};

fn sidx(i : i32, j : i32, k : i32) -> i32 {
    return (k * MP.sz.y + j) * MP.sz.x + i;
}
fn vidx3(i : i32, j : i32, k : i32, c : i32) -> i32 {
    return ((k * MP.sz.y + j) * MP.sz.x + i) * 3 + c;
}

// ComputeImageGradient: central differences in index space, then rotated into
// world space by the direction matrix. Returns zero on the outer shell, exactly
// as the reference does.
fn image_gradient(img : ptr<storage, array<f32>, read>, i : i32, j : i32, k : i32) -> vec3<f32> {
    if (i == 0 || i == MP.sz.x - 1 ||
        j == 0 || j == MP.sz.y - 1 ||
        k == 0 || k == MP.sz.z - 1) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let gx = 0.5 * ((*img)[sidx(i + 1, j, k)] - (*img)[sidx(i - 1, j, k)]) / MP.spc.x;

    // The reference loads the +1 neighbour, then subtracts the -1 neighbour and
    // scales in one expression; kept in that shape.
    var gy = (*img)[sidx(i, j + 1, k)];
    gy = 0.5 * (gy - (*img)[sidx(i, j - 1, k)]) / MP.spc.y;

    var gz = (*img)[sidx(i, j, k + 1)];
    gz = 0.5 * (gz - (*img)[sidx(i, j, k - 1)]) / MP.spc.z;

    return vec3<f32>(
        MP.dir0.x * gx + MP.dir0.y * gy + MP.dir0.z * gz,
        MP.dir1.x * gx + MP.dir1.y * gy + MP.dir1.z * gz,
        MP.dir2.x * gx + MP.dir2.y * gy + MP.dir2.z * gz);
}
