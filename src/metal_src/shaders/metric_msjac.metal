// Port of shaders/metric_msjac.wgsl, itself a port of
// compute_metric.cu : ComputeMetric_MSJac_kernel
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
//  * MSL's select(x, y, cond) returns y when cond is true - the same convention as
//    WGSL's, so the two `select` calls below carry over unchanged. (C's ternary
//    would read the other way round; do not "simplify" them into one.)

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];

static inline int phase_axis(constant MetricParams &MP) { return MP.axes.x; }
static inline int phase_xyz(constant MetricParams &MP)  { return MP.axes.y; }

static inline float field_at(device const float *field, constant MetricParams &MP,
                             int i, int j, int k, int c) {
    return field[vidx3(MP, i, j, k, c)];
}

static inline float3 dir_row(constant MetricParams &MP, int r) {
    if(r == 0) { return float3(MP.dir0.x, MP.dir0.y, MP.dir0.z); }
    if(r == 1) { return float3(MP.dir1.x, MP.dir1.y, MP.dir1.z); }
    return float3(MP.dir2.x, MP.dir2.y, MP.dir2.z);
}
static inline float comp(float3 v, int c) {
    if(c == 0) { return v.x; }
    if(c == 1) { return v.y; }
    return v.z;
}
static inline int szc(constant MetricParams &MP, int c) {
    if(c == 0) { return MP.sz.x; }
    if(c == 1) { return MP.sz.y; }
    return MP.sz.z;
}
static inline float spcc(constant MetricParams &MP, int c) {
    if(c == 0) { return MP.spc.x; }
    if(c == 1) { return MP.spc.y; }
    return MP.spc.z;
}

// ComputeSingleJacobianMatrixAtIndex: derivative of the field's phase component
// along the phase axis, rotated into world space, returning the phase_xyz entry.
static inline float jacobian_at(device const float *field, constant MetricParams &MP,
                                int i, int j, int k, int h) {
    const int ph = phase_axis(MP);
    int pos = i;
    if(ph == 1) { pos = j; } else if(ph == 2) { pos = k; }
    if(pos < h || pos > szc(MP, ph) - h - 1) { return 1.0f; }

    float grad = 0.0f;
    if(ph == 0) {
        grad = 0.5f * (field_at(field, MP, i + h, j, k, 0) - field_at(field, MP, i - h, j, k, 0))
               / MP.spc.x / float(h);
    } else if(ph == 1) {
        grad = 0.5f * (field_at(field, MP, i, j + h, k, 1) - field_at(field, MP, i, j - h, k, 1))
               / MP.spc.y / float(h);
    } else {
        grad = 0.5f * (field_at(field, MP, i, j, k + h, 2) - field_at(field, MP, i, j, k - h, 2))
               / MP.spc.z / float(h);
    }

    // temp has `grad` in the phase slot and zero elsewhere, so the rotation
    // reduces to one column of the direction matrix.
    float3 temp = float3(0.0f, 0.0f, 0.0f);
    if(ph == 0) { temp.x = grad; } else if(ph == 1) { temp.y = grad; } else { temp.z = grad; }

    const float3 t2 = float3(dot(dir_row(MP, 0), temp),
                             dot(dir_row(MP, 1), temp),
                             dot(dir_row(MP, 2), temp));
    return comp(t2, phase_xyz(MP));
}

kernel void main_msjac(device const float   *up_img   [[buffer(0)]],
                       device const float   *down_img [[buffer(1)]],
                       device const float   *def_F    [[buffer(2)]],
                       device const float   *def_M    [[buffer(3)]],
                       device       float   *updF     [[buffer(4)]],
                       device       float   *updM     [[buffer(5)]],
                       device       float   *metric   [[buffer(6)]],
                       constant MetricParams &MP      [[buffer(7)]],
                       uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x); const int j = int(gid.y); const int k = int(gid.z);
    if(i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    float updateF[3] = {0.0f, 0.0f, 0.0f};
    float updateM[3] = {0.0f, 0.0f, 0.0f};

    if(i >= 1 && j >= 1 && k >= 1 &&
       i <= MP.sz.x - 2 && j <= MP.sz.y - 2 && k <= MP.sz.z - 2) {

        const float3 gradI2 = image_gradient(up_img, MP, i, j, k);
        const float3 gradJ2 = image_gradient(down_img, MP, i, j, k);

        // The reference derives a_b from the kernel taps and then zeroes it.
        const float a_b = 0.0f;

        const int ph  = phase_axis(MP);
        const int pxz = phase_xyz(MP);
        const float newph = comp(float3(MP.newph.x, MP.newph.y, MP.newph.z), pxz);

        // ---- at x ----
        {
            float detf = jacobian_at(def_F, MP, i, j, k, 1) + 1.0f;
            float detm = jacobian_at(def_M, MP, i, j, k, 1) + 1.0f;
            if(detf <= 0.0f) { detf = 1e-5f; }
            if(detm <= 0.0f) { detm = 1e-5f; }

            const float valf = up_img[sidx(MP, i, j, k)] * detf;
            const float valm = down_img[sidx(MP, i, j, k)] * detm;
            const float K = valf - valm;
            metric[sidx(MP, i, j, k)] = K * K;

            updateF[0] = 2.0f * K * gradI2.x * detf;
            updateF[1] = 2.0f * K * gradI2.y * detf;
            updateF[2] = 2.0f * K * gradI2.z * detf;
            updateM[0] = -2.0f * K * gradJ2.x * detm;
            updateM[1] = -2.0f * K * gradJ2.y * detm;
            updateM[2] = -2.0f * K * gradJ2.z * detm;
        }

        // ---- at x +/- h along the phase axis (only h = 1 is used) ----
        for(int s = 0; s < 2; s = s + 1) {
            const int h = 1;
            const int step = select(-h, h, s == 0);
            int n0 = i; int n1 = j; int n2 = k;
            if(ph == 0) { n0 = n0 + step; } else if(ph == 1) { n1 = n1 + step; } else { n2 = n2 + step; }

            if(n0 >= h && n1 >= h && n2 >= h &&
               n0 <= MP.sz.x - h - 1 && n1 <= MP.sz.y - h - 1 && n2 <= MP.sz.z - h - 1) {

                float detf2 = jacobian_at(def_F, MP, n0, n1, n2, h) + 1.0f;
                float detm2 = jacobian_at(def_M, MP, n0, n1, n2, h) + 1.0f;
                if(detf2 <= 0.0f) { detf2 = 1e-5f; }
                if(detm2 <= 0.0f) { detm2 = 1e-5f; }
                const float detf = detf2;      // mf() is the identity
                const float detm = detm2;

                const float3 gI = image_gradient(up_img, MP, n0, n1, n2);
                const float3 gJ = image_gradient(down_img, MP, n0, n1, n2);

                const float fval = up_img[sidx(MP, n0, n1, n2)];
                const float mval = down_img[sidx(MP, n0, n1, n2)];
                const float K = fval * detf - mval * detm;

                // dmf() is 1; the sign of the 0.5 term flips between +h and -h
                const float sgn = select(0.5f, -0.5f, s == 0);
                updateF[pxz] = updateF[pxz] +
                    2.0f * K * (comp(gI, pxz) * detf * a_b + newph * 1.0f * fval * sgn / spcc(MP, ph) / float(h));
                updateM[pxz] = updateM[pxz] -
                    2.0f * K * (comp(gJ, pxz) * detm * a_b + newph * 1.0f * mval * sgn / spcc(MP, ph) / float(h));
            }
        }

        updF[vidx3(MP, i, j, k, 0)] = updateF[0];
        updF[vidx3(MP, i, j, k, 1)] = updateF[1];
        updF[vidx3(MP, i, j, k, 2)] = updateF[2];
        updM[vidx3(MP, i, j, k, 0)] = updateM[0];
        updM[vidx3(MP, i, j, k, 1)] = updateM[1];
        updM[vidx3(MP, i, j, k, 2)] = updateM[2];
    }
}
