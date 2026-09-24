// Port of compute_metric.cu : ComputeMetric_CC_kernel
//
// Local (windowed) normalised cross-correlation between the up and down images,
// producing a per-voxel metric image plus the two update fields. The host sums
// the metric image and divides by the voxel count.
//
// The window is clipped at the volume edge rather than padded, and N counts only
// the voxels actually visited - so edge windows are smaller, which is the
// reference behaviour and changes the means.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;

@group(0) @binding(0) var<storage, read>       up_img  : array<f32>;
@group(0) @binding(1) var<storage, read>       down_img: array<f32>;
@group(0) @binding(2) var<storage, read_write> updF    : array<f32>;
@group(0) @binding(3) var<storage, read_write> updM    : array<f32>;
@group(0) @binding(4) var<storage, read_write> metric  : array<f32>;
@group(0) @binding(5) var<uniform>             MP      : MetricParams;

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
    if (i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    var updateF = vec3<f32>(0.0, 0.0, 0.0);
    var updateM = vec3<f32>(0.0, 0.0, 0.0);

    var start = vec3<i32>(max(i - WIN_RAD, 0), max(j - WIN_RAD, 0), max(k - WIN_RAD_Z, 0));
    var end   = vec3<i32>(min(i + WIN_RAD + 1, MP.sz.x),
                          min(j + WIN_RAD + 1, MP.sz.y),
                          min(k + WIN_RAD_Z + 1, MP.sz.z));

    var suma2 = 0.0; var suma = 0.0; var sumac = 0.0;
    var sumc2 = 0.0; var sumc = 0.0;
    var N = 0;
    var valF_center = 0.0;
    var valM_center = 0.0;

    for (var z = start.z; z < end.z; z = z + 1) {
        for (var y = start.y; y < end.y; y = y + 1) {
            for (var x = start.x; x < end.x; x = x + 1) {
                let f = up_img[sidx(x, y, z)];
                let m = down_img[sidx(x, y, z)];
                if (z == k && y == j && x == i) {
                    valF_center = f;
                    valM_center = m;
                }
                suma2 = suma2 + f * f;
                suma  = suma  + f;
                sumc2 = sumc2 + m * m;
                sumc  = sumc  + m;
                sumac = sumac + f * m;
                N = N + 1;
            }
        }
    }

    let Fmean = suma / f32(N);
    let Mmean = sumc / f32(N);

    let valF = valF_center - Fmean;
    let valM = valM_center - Mmean;

    let sFF = suma2 - Fmean * suma;
    let sMM = sumc2 - Mmean * sumc;
    let sFM = sumac - Fmean * sumc;

    var mval = -1.0;
    let sFF_sMM = sFF * sMM;
    if (abs(sFF_sMM) > LIMCC && abs(sMM) > LIMCC) {
        mval = -sFM * sFM / sFF_sMM;

        let first_termF = -2.0 * sFM / sFF_sMM * (valM - sFM / sFF * valF);
        let first_termM = -2.0 * sFM / sFF_sMM * (valF - sFM / sMM * valM);

        let gradI = image_gradient(&up_img, i, j, k);
        let gradJ = image_gradient(&down_img, i, j, k);

        updateF = vec3<f32>(first_termF * gradI.x, first_termF * gradI.y, first_termF * gradI.z);
        updateM = vec3<f32>(first_termM * gradJ.x, first_termM * gradJ.y, first_termM * gradJ.z);
    }

    metric[sidx(i, j, k)] = mval;

    updF[vidx3(i, j, k, 0)] = updateF.x;
    updF[vidx3(i, j, k, 1)] = updateF.y;
    updF[vidx3(i, j, k, 2)] = updateF.z;
    updM[vidx3(i, j, k, 0)] = updateM.x;
    updM[vidx3(i, j, k, 1)] = updateM.y;
    updM[vidx3(i, j, k, 2)] = updateM.z;
}
