// Port of shaders/metric_ccjacs.wgsl, itself a port of
// compute_metric.cu : computeDetImg + computeFiniteDiffStructs +
// ComputeMetric_CCJacS_kernel.
//
// CCJacS correlates a Jacobian-modulated image against the structural image. The
// host runs this three-kernel chain TWICE - once for (up, def_FINV) producing
// updateFieldF, once for (down, def_MINV) producing updateFieldM - sharing one
// metric image.
//
// Note on the metric image: the "at x" block assigns `metric[n] = -1` before
// accumulating, so the second pass OVERWRITES the first pass's contribution
// rather than adding to it. The reported metric therefore comes from the down
// pass alone. That is the reference's behaviour and is reproduced as-is.
//
// Unlike MSJac, `a_b` here is NOT zeroed - it is the ratio of adjacent smoothing
// kernel taps and multiplies the gradient term at the neighbour offsets.
//
// Buffer indices match the WGSL @binding() numbers one for one. `main_ccjacs`
// binds 9 storage buffers, which exceeded WebGPU's default limit of 8 and is why
// compute_metric.mm calls RequireStorageBuffers(9, ...). Metal allows 31, so the
// call is a no-op here; it is kept so the two backends stay textually parallel.

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];

constant int WIN_RAD_JAC   = 9;
constant int WIN_RAD_JAC_Z = 4;

static inline int phase_axis(constant MetricParams &MP) { return MP.axes.x; }
static inline int phase_xyz(constant MetricParams &MP)  { return MP.axes.y; }

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

static inline float jacobian_at(device const float *field, constant MetricParams &MP,
                                int i, int j, int k, int h) {
    const int ph = phase_axis(MP);
    int pos = i;
    if(ph == 1) { pos = j; } else if(ph == 2) { pos = k; }
    if(pos < h || pos > szc(MP, ph) - h - 1) { return 1.0f; }

    float grad = 0.0f;
    if(ph == 0) {
        grad = 0.5f * (field[vidx3(MP, i + h, j, k, 0)] - field[vidx3(MP, i - h, j, k, 0)])
               / MP.spc.x / float(h);
    } else if(ph == 1) {
        grad = 0.5f * (field[vidx3(MP, i, j + h, k, 1)] - field[vidx3(MP, i, j - h, k, 1)])
               / MP.spc.y / float(h);
    } else {
        grad = 0.5f * (field[vidx3(MP, i, j, k + h, 2)] - field[vidx3(MP, i, j, k - h, 2)])
               / MP.spc.z / float(h);
    }

    float3 temp = float3(0.0f, 0.0f, 0.0f);
    if(ph == 0) { temp.x = grad; } else if(ph == 1) { temp.y = grad; } else { temp.z = grad; }
    const float3 t2 = float3(dot(dir_row(MP, 0), temp),
                             dot(dir_row(MP, 1), temp),
                             dot(dir_row(MP, 2), temp));
    return comp(t2, phase_xyz(MP));
}

// ---- pass 1: det-modulated image ------------------------------------------
kernel void compute_det_img(device const float   *b0_img [[buffer(0)]],
                            device const float   *field  [[buffer(2)]],
                            device       float   *detimg [[buffer(3)]],
                            constant MetricParams &MP     [[buffer(11)]],
                            uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x); const int j = int(gid.y); const int k = int(gid.z);
    if(i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    float det = jacobian_at(field, MP, i, j, k, 1);
    if(det <= -1.0f) { det = -1.0f + 1e-5f; }
    detimg[sidx(MP, i, j, k)] = (1.0f + det) * b0_img[sidx(MP, i, j, k)];
}

// ---- pass 2: windowed correlation structures (WIN_RAD_JAC) -----------------
kernel void compute_structs(device const float   *str_img [[buffer(1)]],
                            device const float   *detimg  [[buffer(3)]],
                            device       float   *sKS     [[buffer(4)]],
                            device       float   *sSS     [[buffer(5)]],
                            device       float   *sKK     [[buffer(6)]],
                            device       float   *valS    [[buffer(7)]],
                            device       float   *valK    [[buffer(8)]],
                            constant MetricParams &MP      [[buffer(11)]],
                            uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x); const int j = int(gid.y); const int k = int(gid.z);
    if(i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    const int3 start = int3(max(i - WIN_RAD_JAC, 0), max(j - WIN_RAD_JAC, 0),
                            max(k - WIN_RAD_JAC_Z, 0));
    const int3 end   = int3(min(i + WIN_RAD_JAC + 1, MP.sz.x),
                            min(j + WIN_RAD_JAC + 1, MP.sz.y),
                            min(k + WIN_RAD_JAC_Z + 1, MP.sz.z));

    float suma2 = 0.0f; float suma = 0.0f; float sumac = 0.0f;
    float sumc2 = 0.0f; float sumc = 0.0f;
    int N = 0;
    float vald_center = 0.0f;
    float valS_center = 0.0f;

    for(int z = start.z; z < end.z; z = z + 1) {
        for(int y = start.y; y < end.y; y = y + 1) {
            for(int x = start.x; x < end.x; x = x + 1) {
                const float Kim = detimg[sidx(MP, x, y, z)];
                const float c   = str_img[sidx(MP, x, y, z)];
                if(z == k && y == j && x == i) {
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

    const float Umean = suma / float(N);
    const float Smean = sumc / float(N);
    const int n = sidx(MP, i, j, k);
    sSS[n]  = sumc2 - Smean * sumc;
    sKS[n]  = sumac - Umean * sumc;
    sKK[n]  = suma2 - Umean * suma;
    valS[n] = valS_center - Smean;
    valK[n] = vald_center - Umean;
}

// ---- pass 3: metric value and update field ---------------------------------
kernel void main_ccjacs(device const float   *b0_img [[buffer(0)]],
                        device const float   *field  [[buffer(2)]],
                        device const float   *sKS    [[buffer(4)]],
                        device const float   *sSS    [[buffer(5)]],
                        device const float   *sKK    [[buffer(6)]],
                        device const float   *valS   [[buffer(7)]],
                        device const float   *valK   [[buffer(8)]],
                        device       float   *upd    [[buffer(9)]],
                        device       float   *metric [[buffer(10)]],
                        constant MetricParams &MP     [[buffer(11)]],
                        uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x); const int j = int(gid.y); const int k = int(gid.z);
    if(i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    float update[3] = {0.0f, 0.0f, 0.0f};

    if(i >= 1 && j >= 1 && k >= 1 &&
       i <= MP.sz.x - 2 && j <= MP.sz.y - 2 && k <= MP.sz.z - 2) {

        const int ph  = phase_axis(MP);
        const int pxz = phase_xyz(MP);
        const float newph = comp(float3(MP.newph.x, MP.newph.y, MP.newph.z), pxz);

        // a_b = ratio of the adjacent smoothing taps (NOT zeroed here)
        const int mid = (MP.sz.w - 1) / 2;
        const float b = MP.taps[mid / 4][mid % 4];
        float a = 0.0f;
        if(mid > 0) { a = MP.taps[(mid - 1) / 4][(mid - 1) % 4]; }
        const float a_b = a / b;

        // ---- at x ----
        {
            const int n = sidx(MP, i, j, k);
            const float sSS_val = sSS[n]; const float sKS_val = sKS[n]; const float sKK_val = sKK[n];
            const float valS_val = valS[n]; const float valK_val = valK[n];
            const float sSS_sKK = sSS_val * sKK_val;

            float m = -1.0f;
            if(abs(sSS_sKK) > LIMCCJAC && abs(sKK_val) > LIMCCJAC) {
                m = m + (-sKS_val * sKS_val / sSS_sKK);
                const float first_term = -2.0f * sKS_val / sSS_sKK;

                float detF2 = jacobian_at(field, MP, i, j, k, 1) + 1.0f;
                if(detF2 <= 0.0f) { detF2 = 1e-5f; }
                const float detF = detF2;                     // mf() identity

                const float3 M1t = image_gradient(b0_img, MP, i, j, k);
                float M1[3] = {M1t.x * detF, M1t.y * detF, M1t.z * detF};
                // M2 is zero in the centre block
                const float second_term = valS_val - sKS_val / sKK_val * valK_val;
                update[0] = first_term * second_term * M1[0];
                update[1] = first_term * second_term * M1[1];
                update[2] = first_term * second_term * M1[2];
            }
            metric[n] = m;
        }

        // ---- at x +/- 1 along the phase axis ----
        for(int s = 0; s < 2; s = s + 1) {
            const int h = 1;
            const int step = select(-h, h, s == 0);
            int n0 = i; int n1 = j; int n2 = k;
            if(ph == 0) { n0 = n0 + step; } else if(ph == 1) { n1 = n1 + step; } else { n2 = n2 + step; }

            if(n0 >= h && n1 >= h && n2 >= h &&
               n0 <= MP.sz.x - h - 1 && n1 <= MP.sz.y - h - 1 && n2 <= MP.sz.z - h - 1) {

                const int n = sidx(MP, n0, n1, n2);
                const float sSS_val = sSS[n]; const float sKS_val = sKS[n]; const float sKK_val = sKK[n];
                const float valS_val = valS[n]; const float valK_val = valK[n];
                const float valb_center = b0_img[n];
                const float sSS_sKK = sSS_val * sKK_val;

                if(abs(sSS_sKK) > LIMCCJAC && abs(sKK_val) > LIMCCJAC) {
                    const float first_term = -2.0f * sKS_val / sSS_sKK;

                    float detF2 = jacobian_at(field, MP, n0, n1, n2, 1) + 1.0f;
                    if(detF2 <= 0.0f) { detF2 = 1e-5f; }
                    const float detF = detF2;

                    const float3 M1t = image_gradient(b0_img, MP, n0, n1, n2);
                    // Reference groups as M1 *= (detF*a_b) (compute_metric.cu:1179-1182),
                    // not ((M1*detF)*a_b); the grouping changes the rounding.
                    const float da = detF * a_b;
                    float M1[3] = {M1t.x * da, M1t.y * da, M1t.z * da};
                    const float sgn = select(0.5f, -0.5f, s == 0);
                    const float M2 = newph * 1.0f * valb_center * sgn / spcc(MP, ph) / float(h);
                    M1[pxz] = M1[pxz] + M2;

                    const float second_term = valS_val - sKS_val / sKK_val * valK_val;
                    update[0] = update[0] + first_term * second_term * M1[0];
                    update[1] = update[1] + first_term * second_term * M1[1];
                    update[2] = update[2] + first_term * second_term * M1[2];
                }
            }
        }

        upd[vidx3(MP, i, j, k, 0)] = update[0];
        upd[vidx3(MP, i, j, k, 1)] = update[1];
        upd[vidx3(MP, i, j, k, 2)] = update[2];
    }
}
