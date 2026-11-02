// Port of shaders/metric_ccsk.wgsl, itself a port of
// compute_metric.cu : Compute_K_image + ComputeMetric_CCSK_kernel
//
// CCSK correlates a "K image" - a t-weighted harmonic-style blend of the up and
// down images - against the structural image. Two entry points, run in sequence
// exactly as the reference does: build K, then correlate.
//
// MP.phase.w carries the blend parameter t.

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];

kernel void compute_k(device const float   *up_img   [[buffer(0)]],
                      device const float   *down_img [[buffer(1)]],
                      device       float   *K_img    [[buffer(2)]],
                      constant MetricParams &MP      [[buffer(7)]],
                      uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x); const int j = int(gid.y); const int k = int(gid.z);
    if(i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    const int n = sidx(MP, i, j, k);
    const float t = MP.phase.w;
    const float a = up_img[n];
    const float b = down_img[n];

    const float a_b = a * t + b * (1.0f - t);
    // K_img was memset to zero; the reference only writes above the threshold.
    if(a_b > LIMCCSK) {
        K_img[n] = a * b / a_b;
    }
}

kernel void main_ccsk(device const float   *up_img   [[buffer(0)]],
                      device const float   *down_img [[buffer(1)]],
                      device const float   *K_img    [[buffer(2)]],
                      device const float   *str_img  [[buffer(3)]],
                      device       float   *updF     [[buffer(4)]],
                      device       float   *updM     [[buffer(5)]],
                      device       float   *metric   [[buffer(6)]],
                      constant MetricParams &MP      [[buffer(7)]],
                      uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x); const int j = int(gid.y); const int k = int(gid.z);
    if(i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    const float t = MP.phase.w;
    float3 updateF = float3(0.0f, 0.0f, 0.0f);
    float3 updateM = float3(0.0f, 0.0f, 0.0f);

    const int3 start = int3(max(i - WIN_RAD, 0), max(j - WIN_RAD, 0), max(k - WIN_RAD_Z, 0));
    const int3 end   = int3(min(i + WIN_RAD + 1, MP.sz.x),
                            min(j + WIN_RAD + 1, MP.sz.y),
                            min(k + WIN_RAD_Z + 1, MP.sz.z));

    float suma2 = 0.0f; float suma = 0.0f; float sumac = 0.0f;
    float sumc2 = 0.0f; float sumc = 0.0f;
    int N = 0;
    float valK_center = 0.0f;
    float valS_center = 0.0f;

    for(int z = start.z; z < end.z; z = z + 1) {
        for(int y = start.y; y < end.y; y = y + 1) {
            for(int x = start.x; x < end.x; x = x + 1) {
                const float Kim = K_img[sidx(MP, x, y, z)];
                const float c   = str_img[sidx(MP, x, y, z)];
                if(z == k && y == j && x == i) {
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

    const float Kmean = suma / float(N);
    const float Smean = sumc / float(N);

    const float valK = valK_center - Kmean;
    const float valS = valS_center - Smean;

    const float sKK = suma2 - Kmean * suma;
    const float sSS = sumc2 - Smean * sumc;
    const float sKS = sumac - Kmean * sumc;

    float mval = -1.0f;
    const float sSS_sKK = sSS * sKK;
    if(abs(sSS_sKK) > LIMCCSK && abs(sKK) > LIMCCSK) {
        mval = -sKS * sKS / sSS_sKK;

        const float first_term = -2.0f * sKS / sSS_sKK * (valS - sKS / sKK * valK);
        const float fval = up_img[sidx(MP, i, j, k)];
        const float mvalue = down_img[sidx(MP, i, j, k)];

        const float sm = fval * t + mvalue * (1.0f - t);
        if(sm * sm > LIMCCSK) {
            {
                const float grad_term = mvalue / sm - fval * mvalue * t / sm / sm;
                const float3 g = image_gradient(up_img, MP, i, j, k);
                updateF = float3(first_term * grad_term * g.x,
                                 first_term * grad_term * g.y,
                                 first_term * grad_term * g.z);
            }
            {
                const float grad_term = fval / sm - fval * mvalue * (1.0f - t) / sm / sm;
                const float3 g = image_gradient(down_img, MP, i, j, k);
                updateM = float3(first_term * grad_term * g.x,
                                 first_term * grad_term * g.y,
                                 first_term * grad_term * g.z);
            }
        }
    }

    metric[sidx(MP, i, j, k)] = mval;

    updF[vidx3(MP, i, j, k, 0)] = updateF.x;
    updF[vidx3(MP, i, j, k, 1)] = updateF.y;
    updF[vidx3(MP, i, j, k, 2)] = updateF.z;
    updM[vidx3(MP, i, j, k, 0)] = updateM.x;
    updM[vidx3(MP, i, j, k, 1)] = updateM.y;
    updM[vidx3(MP, i, j, k, 2)] = updateM.z;
}
