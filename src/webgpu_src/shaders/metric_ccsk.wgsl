// Port of compute_metric.cu : Compute_K_image + ComputeMetric_CCSK_kernel
//
// CCSK correlates a "K image" - a t-weighted harmonic-style blend of the up and
// down images - against the structural image. Two entry points, run in sequence
// exactly as the reference does: build K, then correlate.
//
// MP.phase.w carries the blend parameter t.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;

@group(0) @binding(0) var<storage, read>       up_img  : array<f32>;
@group(0) @binding(1) var<storage, read>       down_img: array<f32>;
@group(0) @binding(2) var<storage, read_write> K_img   : array<f32>;
@group(0) @binding(3) var<storage, read>       str_img : array<f32>;
@group(0) @binding(4) var<storage, read_write> updF    : array<f32>;
@group(0) @binding(5) var<storage, read_write> updM    : array<f32>;
@group(0) @binding(6) var<storage, read_write> metric  : array<f32>;
@group(0) @binding(7) var<uniform>             MP      : MetricParams;

@compute @workgroup_size(WGX, WGY, WGZ)
fn compute_k(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
    if (i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    let n = sidx(i, j, k);
    let t = MP.phase.w;
    let a = up_img[n];
    let b = down_img[n];

    let a_b = a * t + b * (1.0 - t);
    // K_img was memset to zero; the reference only writes above the threshold.
    if (a_b > LIMCCSK) {
        K_img[n] = a * b / a_b;
    }
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
    if (i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    let t = MP.phase.w;
    var updateF = vec3<f32>(0.0, 0.0, 0.0);
    var updateM = vec3<f32>(0.0, 0.0, 0.0);

    let start = vec3<i32>(max(i - WIN_RAD, 0), max(j - WIN_RAD, 0), max(k - WIN_RAD_Z, 0));
    let end   = vec3<i32>(min(i + WIN_RAD + 1, MP.sz.x),
                          min(j + WIN_RAD + 1, MP.sz.y),
                          min(k + WIN_RAD_Z + 1, MP.sz.z));

    var suma2 = 0.0; var suma = 0.0; var sumac = 0.0;
    var sumc2 = 0.0; var sumc = 0.0;
    var N = 0;
    var valK_center = 0.0;
    var valS_center = 0.0;

    for (var z = start.z; z < end.z; z = z + 1) {
        for (var y = start.y; y < end.y; y = y + 1) {
            for (var x = start.x; x < end.x; x = x + 1) {
                let Kim = K_img[sidx(x, y, z)];
                let c   = str_img[sidx(x, y, z)];
                if (z == k && y == j && x == i) {
                    valK_center = Kim;
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

    let Kmean = suma / f32(N);
    let Smean = sumc / f32(N);

    let valK = valK_center - Kmean;
    let valS = valS_center - Smean;

    let sKK = suma2 - Kmean * suma;
    let sSS = sumc2 - Smean * sumc;
    let sKS = sumac - Kmean * sumc;

    var mval = -1.0;
    let sSS_sKK = sSS * sKK;
    if (abs(sSS_sKK) > LIMCCSK && abs(sKK) > LIMCCSK) {
        mval = -sKS * sKS / sSS_sKK;

        let first_term = -2.0 * sKS / sSS_sKK * (valS - sKS / sKK * valK);
        let fval = up_img[sidx(i, j, k)];
        let mvalue = down_img[sidx(i, j, k)];

        let sm = fval * t + mvalue * (1.0 - t);
        if (sm * sm > LIMCCSK) {
            {
                let grad_term = mvalue / sm - fval * mvalue * t / sm / sm;
                let g = image_gradient(&up_img, i, j, k);
                updateF = vec3<f32>(first_term * grad_term * g.x,
                                    first_term * grad_term * g.y,
                                    first_term * grad_term * g.z);
            }
            {
                let grad_term = fval / sm - fval * mvalue * (1.0 - t) / sm / sm;
                let g = image_gradient(&down_img, i, j, k);
                updateM = vec3<f32>(first_term * grad_term * g.x,
                                    first_term * grad_term * g.y,
                                    first_term * grad_term * g.z);
            }
        }
    }

    metric[sidx(i, j, k)] = mval;

    updF[vidx3(i, j, k, 0)] = updateF.x;
    updF[vidx3(i, j, k, 1)] = updateF.y;
    updF[vidx3(i, j, k, 2)] = updateF.z;
    updM[vidx3(i, j, k, 0)] = updateM.x;
    updM[vidx3(i, j, k, 1)] = updateM.y;
    updM[vidx3(i, j, k, 2)] = updateM.z;
}
