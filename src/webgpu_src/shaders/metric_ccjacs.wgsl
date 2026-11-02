// Port of compute_metric.cu : computeDetImg + computeFiniteDiffStructs +
// ComputeMetric_CCJacS_kernel.
//
// CCJacS correlates a Jacobian-modulated image against the structural image. The
// host runs this three-kernel chain TWICE - once for (up, def_FINV) producing
// updateFieldF, once for (down, def_MINV) producing updateFieldM - sharing one
// metric image.
//
// Note on the metric image: the "at x" block assigns `row_M[i] = -1` before
// accumulating, so the second pass OVERWRITES the first pass's contribution
// rather than adding to it. The reported metric therefore comes from the down
// pass alone. That is the reference's behaviour and is reproduced as-is.
//
// Unlike MSJac, `a_b` here is NOT zeroed - it is the ratio of adjacent smoothing
// kernel taps and multiplies the gradient term at the neighbour offsets.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;

// NOTE: `main` below binds 9 storage buffers, exceeding WebGPU's default limit of
// 8. compute_metric.cxx's ComputeMetric_CCJacSWithTaps calls
// wgpuctx::RequireStorageBuffers(9, ...) to check for it. If you add or remove a
// storage binding used by `main`, update that call.
@group(0) @binding(0) var<storage, read>       b0_img  : array<f32>;   // up or down
@group(0) @binding(1) var<storage, read>       str_img : array<f32>;
@group(0) @binding(2) var<storage, read>       field   : array<f32>;   // def_FINV or def_MINV
@group(0) @binding(3) var<storage, read_write> detimg  : array<f32>;
@group(0) @binding(4) var<storage, read_write> sKS     : array<f32>;
@group(0) @binding(5) var<storage, read_write> sSS     : array<f32>;
@group(0) @binding(6) var<storage, read_write> sKK     : array<f32>;
@group(0) @binding(7) var<storage, read_write> valS    : array<f32>;
@group(0) @binding(8) var<storage, read_write> valK    : array<f32>;
@group(0) @binding(9) var<storage, read_write> upd     : array<f32>;
@group(0) @binding(10) var<storage, read_write> metric : array<f32>;
@group(0) @binding(11) var<uniform>            MP      : MetricParams;

fn phase_axis() -> i32 { return MP.axes.x; }
fn phase_xyz()  -> i32 { return MP.axes.y; }

fn dir_row(r : i32) -> vec3<f32> {
    if (r == 0) { return vec3<f32>(MP.dir0.x, MP.dir0.y, MP.dir0.z); }
    if (r == 1) { return vec3<f32>(MP.dir1.x, MP.dir1.y, MP.dir1.z); }
    return vec3<f32>(MP.dir2.x, MP.dir2.y, MP.dir2.z);
}
fn comp(v : vec3<f32>, c : i32) -> f32 {
    if (c == 0) { return v.x; }
    if (c == 1) { return v.y; }
    return v.z;
}
fn szc(c : i32) -> i32 {
    if (c == 0) { return MP.sz.x; }
    if (c == 1) { return MP.sz.y; }
    return MP.sz.z;
}
fn spcc(c : i32) -> f32 {
    if (c == 0) { return MP.spc.x; }
    if (c == 1) { return MP.spc.y; }
    return MP.spc.z;
}

fn jacobian_at(i : i32, j : i32, k : i32, h : i32) -> f32 {
    let ph = phase_axis();
    var pos = i;
    if (ph == 1) { pos = j; } else if (ph == 2) { pos = k; }
    if (pos < h || pos > szc(ph) - h - 1) { return 1.0; }

    var grad = 0.0;
    if (ph == 0) {
        grad = 0.5 * (field[vidx3(i + h, j, k, 0)] - field[vidx3(i - h, j, k, 0)])
               / MP.spc.x / f32(h);
    } else if (ph == 1) {
        grad = 0.5 * (field[vidx3(i, j + h, k, 1)] - field[vidx3(i, j - h, k, 1)])
               / MP.spc.y / f32(h);
    } else {
        grad = 0.5 * (field[vidx3(i, j, k + h, 2)] - field[vidx3(i, j, k - h, 2)])
               / MP.spc.z / f32(h);
    }

    var temp = vec3<f32>(0.0, 0.0, 0.0);
    if (ph == 0) { temp.x = grad; } else if (ph == 1) { temp.y = grad; } else { temp.z = grad; }
    let t2 = vec3<f32>(dot(dir_row(0), temp), dot(dir_row(1), temp), dot(dir_row(2), temp));
    return comp(t2, phase_xyz());
}

// ---- pass 1: det-modulated image ------------------------------------------
@compute @workgroup_size(WGX, WGY, WGZ)
fn compute_det_img(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
    if (i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    var det = jacobian_at(i, j, k, 1);
    if (det <= -1.0) { det = -1.0 + 1e-5; }
    detimg[sidx(i, j, k)] = (1.0 + det) * b0_img[sidx(i, j, k)];
}

// ---- pass 2: windowed correlation structures (WIN_RAD_JAC) -----------------
const WIN_RAD_JAC   : i32 = 9;
const WIN_RAD_JAC_Z : i32 = 4;

@compute @workgroup_size(WGX, WGY, WGZ)
fn compute_structs(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
    if (i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    let start = vec3<i32>(max(i - WIN_RAD_JAC, 0), max(j - WIN_RAD_JAC, 0),
                          max(k - WIN_RAD_JAC_Z, 0));
    let end   = vec3<i32>(min(i + WIN_RAD_JAC + 1, MP.sz.x),
                          min(j + WIN_RAD_JAC + 1, MP.sz.y),
                          min(k + WIN_RAD_JAC_Z + 1, MP.sz.z));

    var suma2 = 0.0; var suma = 0.0; var sumac = 0.0;
    var sumc2 = 0.0; var sumc = 0.0;
    var N = 0;
    var vald_center = 0.0;
    var valS_center = 0.0;

    for (var z = start.z; z < end.z; z = z + 1) {
        for (var y = start.y; y < end.y; y = y + 1) {
            for (var x = start.x; x < end.x; x = x + 1) {
                let Kim = detimg[sidx(x, y, z)];
                let c   = str_img[sidx(x, y, z)];
                if (z == k && y == j && x == i) {
                    vald_center = Kim;
                    valS_center = c;
                }
                suma2 = suma2 + Kim * Kim;
                suma  = suma  + Kim;
                sumc2 = sumc2 + c * c;
                sumc  = sumc  + c;
                sumac = sumac + Kim * c;
                N = N + 1;
            }
        }
    }

    let Umean = suma / f32(N);
    let Smean = sumc / f32(N);
    let n = sidx(i, j, k);
    sSS[n]  = sumc2 - Smean * sumc;
    sKS[n]  = sumac - Umean * sumc;
    sKK[n]  = suma2 - Umean * suma;
    valS[n] = valS_center - Smean;
    valK[n] = vald_center - Umean;
}

// ---- pass 3: metric value and update field ---------------------------------
@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
    if (i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    var update = array<f32, 3>(0.0, 0.0, 0.0);

    if (i >= 1 && j >= 1 && k >= 1 &&
        i <= MP.sz.x - 2 && j <= MP.sz.y - 2 && k <= MP.sz.z - 2) {

        let ph  = phase_axis();
        let pxz = phase_xyz();
        let newph = comp(vec3<f32>(MP.newph.x, MP.newph.y, MP.newph.z), pxz);

        // a_b = ratio of the adjacent smoothing taps (NOT zeroed here)
        let mid = (MP.sz.w - 1) / 2;
        let b = MP.taps[mid / 4][mid % 4];
        var a = 0.0;
        if (mid > 0) { a = MP.taps[(mid - 1) / 4][(mid - 1) % 4]; }
        let a_b = a / b;

        // ---- at x ----
        {
            let n = sidx(i, j, k);
            let sSS_val = sSS[n]; let sKS_val = sKS[n]; let sKK_val = sKK[n];
            let valS_val = valS[n]; let valK_val = valK[n];
            let sSS_sKK = sSS_val * sKK_val;

            var m = -1.0;
            if (abs(sSS_sKK) > LIMCCJAC && abs(sKK_val) > LIMCCJAC) {
                m = m + (-sKS_val * sKS_val / sSS_sKK);
                let first_term = -2.0 * sKS_val / sSS_sKK;

                var detF2 = jacobian_at(i, j, k, 1) + 1.0;
                if (detF2 <= 0.0) { detF2 = 1e-5; }
                let detF = detF2;                     // mf() identity

                let M1t = image_gradient(&b0_img, i, j, k);
                var M1 = array<f32, 3>(M1t.x * detF, M1t.y * detF, M1t.z * detF);
                // M2 is zero in the centre block
                let second_term = valS_val - sKS_val / sKK_val * valK_val;
                update[0] = first_term * second_term * M1[0];
                update[1] = first_term * second_term * M1[1];
                update[2] = first_term * second_term * M1[2];
            }
            metric[n] = m;
        }

        // ---- at x +/- 1 along the phase axis ----
        for (var s = 0; s < 2; s = s + 1) {
            let h = 1;
            let step = select(-h, h, s == 0);
            var n0 = i; var n1 = j; var n2 = k;
            if (ph == 0) { n0 = n0 + step; } else if (ph == 1) { n1 = n1 + step; } else { n2 = n2 + step; }

            if (n0 >= h && n1 >= h && n2 >= h &&
                n0 <= MP.sz.x - h - 1 && n1 <= MP.sz.y - h - 1 && n2 <= MP.sz.z - h - 1) {

                let n = sidx(n0, n1, n2);
                let sSS_val = sSS[n]; let sKS_val = sKS[n]; let sKK_val = sKK[n];
                let valS_val = valS[n]; let valK_val = valK[n];
                let valb_center = b0_img[n];
                let sSS_sKK = sSS_val * sKK_val;

                if (abs(sSS_sKK) > LIMCCJAC && abs(sKK_val) > LIMCCJAC) {
                    let first_term = -2.0 * sKS_val / sSS_sKK;

                    var detF2 = jacobian_at(n0, n1, n2, 1) + 1.0;
                    if (detF2 <= 0.0) { detF2 = 1e-5; }
                    let detF = detF2;

                    let M1t = image_gradient(&b0_img, n0, n1, n2);
                    // Reference groups as M1 *= (detF*a_b) (compute_metric.cu:1179-1182),
                    // not ((M1*detF)*a_b); the grouping changes the rounding.
                    let da = detF * a_b;
                    var M1 = array<f32, 3>(M1t.x * da, M1t.y * da, M1t.z * da);
                    let sgn = select(0.5, -0.5, s == 0);
                    let M2 = newph * 1.0 * valb_center * sgn / spcc(ph) / f32(h);
                    M1[pxz] = M1[pxz] + M2;

                    let second_term = valS_val - sKS_val / sKK_val * valK_val;
                    update[0] = update[0] + first_term * second_term * M1[0];
                    update[1] = update[1] + first_term * second_term * M1[1];
                    update[2] = update[2] + first_term * second_term * M1[2];
                }
            }
        }

        upd[vidx3(i, j, k, 0)] = update[0];
        upd[vidx3(i, j, k, 1)] = update[1];
        upd[vidx3(i, j, k, 2)] = update[2];
    }
}
