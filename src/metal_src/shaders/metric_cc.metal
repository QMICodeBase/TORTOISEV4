// Port of shaders/metric_cc.wgsl, itself a port of
// compute_metric.cu : ComputeMetric_CC_kernel
//
// Local (windowed) normalised cross-correlation between the up and down images,
// producing a per-voxel metric image plus the two update fields. The host sums
// the metric image and divides by the voxel count.
//
// The window is clipped at the volume edge rather than padded, and N counts only
// the voxels actually visited - so edge windows are smaller, which is the
// reference behaviour and changes the means.

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];

kernel void main_cc(device const float   *up_img   [[buffer(0)]],
                    device const float   *down_img [[buffer(1)]],
                    device       float   *updF     [[buffer(2)]],
                    device       float   *updM     [[buffer(3)]],
                    device       float   *metric   [[buffer(4)]],
                    constant MetricParams &MP      [[buffer(5)]],
                    uint3 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x); const int j = int(gid.y); const int k = int(gid.z);
    if(i >= MP.sz.x || j >= MP.sz.y || k >= MP.sz.z) { return; }

    float3 updateF = float3(0.0f, 0.0f, 0.0f);
    float3 updateM = float3(0.0f, 0.0f, 0.0f);

    const int3 start = int3(max(i - WIN_RAD, 0), max(j - WIN_RAD, 0), max(k - WIN_RAD_Z, 0));
    const int3 end   = int3(min(i + WIN_RAD + 1, MP.sz.x),
                            min(j + WIN_RAD + 1, MP.sz.y),
                            min(k + WIN_RAD_Z + 1, MP.sz.z));

    float suma2 = 0.0f; float suma = 0.0f; float sumac = 0.0f;
    float sumc2 = 0.0f; float sumc = 0.0f;
    int N = 0;
    float valF_center = 0.0f;
    float valM_center = 0.0f;

    for(int z = start.z; z < end.z; z = z + 1) {
        for(int y = start.y; y < end.y; y = y + 1) {
            for(int x = start.x; x < end.x; x = x + 1) {
                const float f = up_img[sidx(MP, x, y, z)];
                const float m = down_img[sidx(MP, x, y, z)];
                if(z == k && y == j && x == i) {
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

    const float Fmean = suma / float(N);
    const float Mmean = sumc / float(N);

    const float valF = valF_center - Fmean;
    const float valM = valM_center - Mmean;

    const float sFF = suma2 - Fmean * suma;
    const float sMM = sumc2 - Mmean * sumc;
    const float sFM = sumac - Fmean * sumc;

    float mval = -1.0f;
    const float sFF_sMM = sFF * sMM;
    if(abs(sFF_sMM) > LIMCC && abs(sMM) > LIMCC) {
        mval = -sFM * sFM / sFF_sMM;

        const float first_termF = -2.0f * sFM / sFF_sMM * (valM - sFM / sFF * valF);
        const float first_termM = -2.0f * sFM / sFF_sMM * (valF - sFM / sMM * valM);

        const float3 gradI = image_gradient(up_img, MP, i, j, k);
        const float3 gradJ = image_gradient(down_img, MP, i, j, k);

        updateF = float3(first_termF * gradI.x, first_termF * gradI.y, first_termF * gradI.z);
        updateM = float3(first_termM * gradJ.x, first_termM * gradJ.y, first_termM * gradJ.z);
    }

    metric[sidx(MP, i, j, k)] = mval;

    updF[vidx3(MP, i, j, k, 0)] = updateF.x;
    updF[vidx3(MP, i, j, k, 1)] = updateF.y;
    updF[vidx3(MP, i, j, k, 2)] = updateF.z;
    updM[vidx3(MP, i, j, k, 0)] = updateM.x;
    updM[vidx3(MP, i, j, k, 1)] = updateM.y;
    updM[vidx3(MP, i, j, k, 2)] = updateM.z;
}
