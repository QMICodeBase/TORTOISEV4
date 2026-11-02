// Port of compute_metric.cu : ComputeMetric_MSJac_kernel
//
// Mean-squares metric with Jacobian modulation along the phase-encode axis: the
// intensity at each voxel is scaled by the local stretch of the deformation
// field, so compression brightens and expansion dims, matching EPI physics.
//
// Faithfulness notes:
//  * mf(det) = det and dmf(x) = 1 in the reference (both are identity stubs with
//    the interesting versions commented out), so they are inlined as such here.
//  * `a_b` is computed from the smoothing kernel taps and then unconditionally
//    overwritten with 0 in all three blocks, which kills the gradient term at the
//    neighbour offsets. That is dead-but-live code: it is reproduced exactly,
//    including the multiply-by-zero, rather than simplified away.
//  * only h=1 is ever used (`for(int h=1;h<2;h++)`), so the loop is unrolled.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;

@group(0) @binding(0) var<storage, read>       up_img  : array<f32>;
@group(0) @binding(1) var<storage, read>       down_img: array<f32>;
@group(0) @binding(2) var<storage, read>       def_F   : array<f32>;
@group(0) @binding(3) var<storage, read>       def_M   : array<f32>;
@group(0) @binding(4) var<storage, read_write> updF    : array<f32>;
@group(0) @binding(5) var<storage, read_write> updM    : array<f32>;
@group(0) @binding(6) var<storage, read_write> metric  : array<f32>;
@group(0) @binding(7) var<uniform>             MP      : MetricParams;

fn phase_axis() -> i32 { return MP.axes.x; }
fn phase_xyz()  -> i32 { return MP.axes.y; }

fn field_at(field : ptr<storage, array<f32>, read>, i : i32, j : i32, k : i32, c : i32) -> f32 {
    return (*field)[vidx3(i, j, k, c)];
}

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

// ComputeSingleJacobianMatrixAtIndex: derivative of the field's phase component
// along the phase axis, rotated into world space, returning the phase_xyz entry.
fn jacobian_at(field : ptr<storage, array<f32>, read>,
               i : i32, j : i32, k : i32, h : i32) -> f32 {
    let ph = phase_axis();
    var pos = i;
    if (ph == 1) { pos = j; } else if (ph == 2) { pos = k; }
    if (pos < h || pos > szc(ph) - h - 1) { return 1.0; }

    var grad = 0.0;
    if (ph == 0) {
        grad = 0.5 * (field_at(field, i + h, j, k, 0) - field_at(field, i - h, j, k, 0))
               / MP.spc.x / f32(h);
    } else if (ph == 1) {
        grad = 0.5 * (field_at(field, i, j + h, k, 1) - field_at(field, i, j - h, k, 1))
               / MP.spc.y / f32(h);
    } else {
        grad = 0.5 * (field_at(field, i, j, k + h, 2) - field_at(field, i, j, k - h, 2))
               / MP.spc.z / f32(h);
    }

    // temp has `grad` in the phase slot and zero elsewhere, so the rotation
    // reduces to one column of the direction matrix.
    var temp = vec3<f32>(0.0, 0.0, 0.0);
    if (ph == 0) { temp.x = grad; } else if (ph == 1) { temp.y = grad; } else { temp.z = grad; }

    let t2 = vec3<f32>(dot(dir_row(0), temp), dot(dir_row(1), temp), dot(dir_row(2), temp));
    return comp(t2, phase_xyz());
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
    if (i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    var updateF = array<f32, 3>(0.0, 0.0, 0.0);
    var updateM = array<f32, 3>(0.0, 0.0, 0.0);

    if (i >= 1 && j >= 1 && k >= 1 &&
        i <= MP.sz.x - 2 && j <= MP.sz.y - 2 && k <= MP.sz.z - 2) {

        let gradI2 = image_gradient(&up_img, i, j, k);
        let gradJ2 = image_gradient(&down_img, i, j, k);

        // The reference derives a_b from the kernel taps and then zeroes it.
        let a_b = 0.0;

        let ph  = phase_axis();
        let pxz = phase_xyz();
        let newph = comp(vec3<f32>(MP.newph.x, MP.newph.y, MP.newph.z), pxz);

        // ---- at x ----
        {
            var detf = jacobian_at(&def_F, i, j, k, 1) + 1.0;
            var detm = jacobian_at(&def_M, i, j, k, 1) + 1.0;
            if (detf <= 0.0) { detf = 1e-5; }
            if (detm <= 0.0) { detm = 1e-5; }

            let valf = up_img[sidx(i, j, k)] * detf;
            let valm = down_img[sidx(i, j, k)] * detm;
            let K = valf - valm;
            metric[sidx(i, j, k)] = K * K;

            updateF[0] = 2.0 * K * gradI2.x * detf;
            updateF[1] = 2.0 * K * gradI2.y * detf;
            updateF[2] = 2.0 * K * gradI2.z * detf;
            updateM[0] = -2.0 * K * gradJ2.x * detm;
            updateM[1] = -2.0 * K * gradJ2.y * detm;
            updateM[2] = -2.0 * K * gradJ2.z * detm;
        }

        // ---- at x +/- h along the phase axis (only h = 1 is used) ----
        for (var s = 0; s < 2; s = s + 1) {
            let h = 1;
            let step = select(-h, h, s == 0);
            var n0 = i; var n1 = j; var n2 = k;
            if (ph == 0) { n0 = n0 + step; } else if (ph == 1) { n1 = n1 + step; } else { n2 = n2 + step; }

            if (n0 >= h && n1 >= h && n2 >= h &&
                n0 <= MP.sz.x - h - 1 && n1 <= MP.sz.y - h - 1 && n2 <= MP.sz.z - h - 1) {

                var detf2 = jacobian_at(&def_F, n0, n1, n2, h) + 1.0;
                var detm2 = jacobian_at(&def_M, n0, n1, n2, h) + 1.0;
                if (detf2 <= 0.0) { detf2 = 1e-5; }
                if (detm2 <= 0.0) { detm2 = 1e-5; }
                let detf = detf2;      // mf() is the identity
                let detm = detm2;

                let gI = image_gradient(&up_img, n0, n1, n2);
                let gJ = image_gradient(&down_img, n0, n1, n2);

                let fval = up_img[sidx(n0, n1, n2)];
                let mval = down_img[sidx(n0, n1, n2)];
                let K = fval * detf - mval * detm;

                // dmf() is 1; the sign of the 0.5 term flips between +h and -h
                let sgn = select(0.5, -0.5, s == 0);
                updateF[pxz] = updateF[pxz] +
                    2.0 * K * (comp(gI, pxz) * detf * a_b + newph * 1.0 * fval * sgn / spcc(ph) / f32(h));
                updateM[pxz] = updateM[pxz] -
                    2.0 * K * (comp(gJ, pxz) * detm * a_b + newph * 1.0 * mval * sgn / spcc(ph) / f32(h));
            }
        }

        updF[vidx3(i, j, k, 0)] = updateF[0];
        updF[vidx3(i, j, k, 1)] = updateF[1];
        updF[vidx3(i, j, k, 2)] = updateF[2];
        updM[vidx3(i, j, k, 0)] = updateM[0];
        updM[vidx3(i, j, k, 1)] = updateM[1];
        updM[vidx3(i, j, k, 2)] = updateM[2];
    }
}
