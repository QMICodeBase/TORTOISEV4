// Port of compute_entropy.cu : the ComputeJointEntropy pipeline.
//
// Design difference, and why it is still exact:
//
// CUDA builds a per-block joint histogram in shared memory (u32 atomics), writes
// one Nbins x Nbins slice per block, then sums the slices into a float histogram.
// With Nbins=100 that slice is 100*100*4 = 40 KB of workgroup storage, which
// exceeds WebGPU's portable limit (16 KB) and Metal's on much hardware. So this
// port accumulates a SINGLE histogram with global atomics instead.
//
// That is not an approximation: the counts are integers, integer addition is
// associative, and the per-block-then-sum total equals the direct total exactly.
// The subsequent u32 -> f32 conversion is exact as long as no bin exceeds 2^24
// (16.7 M); the largest validation volume has 1.8 M voxels, so a single bin
// cannot come close.
//
// Everything downstream - the marginal projections, the p*log(p) transform and
// the sum order - follows the reference exactly.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;
override MARGINAL_AXIS : u32 = 0u;    // 0 = moving (column sums), 1 = fixed (row sums)

const PADDING : i32 = 2;

struct Params {
    sz    : vec4<i32>,      // x, y, z, Nbins
    lims  : vec4<f32>,      // low1, high1, low2, high2
    misc  : vec4<f32>,      // hist_sum
};

@group(0) @binding(0) var<storage, read>       img1  : array<f32>;
@group(0) @binding(1) var<storage, read>       img2  : array<f32>;
@group(0) @binding(2) var<storage, read_write> histu : array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             P     : Params;
@group(0) @binding(4) var<storage, read_write> histf : array<f32>;
@group(0) @binding(5) var<storage, read_write> marg  : array<f32>;

// Parzen bin index, matching ITK's convention as the reference implements it.
fn parzen_index(val : f32, low : f32, high : f32, nbins : i32) -> i32 {
    let bin_size = (high - low) / f32(nbins - 2 * PADDING);
    let norm_min = low / bin_size - f32(PADDING);
    let term = val / bin_size - norm_min;
    var idx = i32(term);
    if (idx < 2) {
        idx = 2;
    } else if (idx > nbins - 3) {
        idx = nbins - 3;
    }
    return idx;
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn joint_histogram(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
    if (i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    let n = (k * P.sz.y + j) * P.sz.x + i;
    let v1 = img1[n];
    let v2 = img2[n];

    let low1 = P.lims.x; let high1 = P.lims.y;
    let low2 = P.lims.z; let high2 = P.lims.w;
    let nbins = P.sz.w;

    if (v1 >= low1 && v2 >= low2 && v1 <= high1 && v2 <= high2) {
        let fixed_idx  = parzen_index(v1, low1, high1, nbins);
        let moving_idx = parzen_index(v2, low2, high2, nbins);
        atomicAdd(&histu[fixed_idx * nbins + moving_idx], 1u);
    }
}

// u32 counts -> f32 histogram (AccumulateJointPartialHistogram_kernel's result).
@compute @workgroup_size(WGX, WGY, WGZ)
fn hist_to_float(@builtin(global_invocation_id) gid : vec3<u32>) {
    let n = i32(gid.x);
    let nbins = P.sz.w;
    if (n >= nbins * nbins) { return; }
    histf[n] = f32(atomicLoad(&histu[n]));
}

// Project the joint histogram onto a marginal:
//   moving (axis 0) sums down columns, fixed (axis 1) sums across rows.
@compute @workgroup_size(WGX, WGY, WGZ)
fn joint_to_marginal(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = i32(gid.x);
    let nbins = P.sz.w;
    if (idx >= nbins) { return; }

    var sm = 0.0;
    for (var t = 0; t < nbins; t = t + 1) {
        if (MARGINAL_AXIS == 0u) {
            sm = sm + histf[t * nbins + idx];      // column sum -> moving
        } else {
            sm = sm + histf[idx * nbins + t];      // row sum    -> fixed
        }
    }
    marg[idx] = sm;
}

// p*log(p), with the reference's 1e-10 cutoff. misc.x carries the histogram sum.
//
// MEASURED RESIDUAL. Replaying the golden vectors at a 1e-9 gate:
//
//   GaussianSmoothImage        bit-exact
//   ComputeJointEntropy        rel 1.06e-7  <- here
//   ResampleImage              rel 1.45e-6  (FMA contraction, see resample_image.wgsl)
//   QuadraticTransformImageC   rel 2.85e-3 against CUDA - NOT bit-exact
//
// CORRECTION: an earlier version of this comment listed QuadraticTransformImageC as
// bit-exact and called this "the ONLY place in the DIFFPREP-reachable set where the
// port is not bit-exact". Both were wrong. `--tol` only overwrote the elementwise
// gate at the time, so a tight value never reached the texture-sampling ops and a
// PASS there printed no residual. That is fixed (`--tol` now sets both gates,
// `--tol-texture` exists, and every PASS reports its residual).
//
// Note also that the 1.06e-7 below is measured on a record whose img2 input IS
// CUDA's own QuadraticTransformImageC output (verified: identical blob digests), so
// the histogram is identical by construction and only the transcendental remains.
// It is a lower bound on this op's contribution, not its contribution in situ.
//
// The structure above matches the reference exactly - same division, same 1e-10
// cutoff, same p*log(p), same summation order (1x256, ScalarFindSum2). What differs
// is the transcendental itself: CUDA device code resolves log(float) to `logf`,
// whose implementation and rounding are NVIDIA's; WGSL `log()` lowers to the
// SPIR-V/Vulkan implementation. They agree to a few ulp, and that error accumulates
// over Nbins*Nbins = 10,000 histogram entries.
//
// This is IRREDUCIBLE without reimplementing logf bit-for-bit, which would be
// hardware-specific and defeat the point of a portable backend. It is also below
// the reference's own noise floor: CUDA compiled -fmad=false disagrees with
// -fmad=true by 2.67e-6 (CLAUDE.md 5.1), ~25x larger than this.
//
// Do not "fix" this by changing the cutoff, the order, or the formula - those would
// be real divergences. See benchmark/milestones/M6.md for the end-to-end consequence.
@compute @workgroup_size(WGX, WGY, WGZ)
fn binwise_entropy_marginal(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x);
    if (i >= P.sz.w) { return; }
    let prob = marg[i] / P.misc.x;
    if (prob > 1e-10) { marg[i] = prob * log(prob); } else { marg[i] = 0.0; }
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn binwise_entropy_joint(@builtin(global_invocation_id) gid : vec3<u32>) {
    let n = i32(gid.x);
    let nbins = P.sz.w;
    if (n >= nbins * nbins) { return; }
    let prob = histf[n] / P.misc.x;
    if (prob > 1e-10) { histf[n] = prob * log(prob); } else { histf[n] = 0.0; }
}
