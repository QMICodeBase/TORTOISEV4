// Port of warp_image.cu : warp_image_kernel
//
// This is the one operation where CUDA's cudaAddressModeBorder is load-bearing:
// the kernel samples tex3D with NO domain guard, so neighbours outside
// [0, N-1] contribute 0 while in-range neighbours still contribute. That is
// different from ResampleImage/QuadraticTransformImage, which zero the whole
// voxel when the sample point leaves the domain (CLAUDE.md 5.2).
//
// WebGPU has no clamp-to-border address mode, so the trilinear filter is done by
// hand.
//
// On weight precision: the CUDA Programming Guide says the texture unit stores
// interpolation weights in 1.8 fixed point (8 fractional bits), which would make
// exact fp32 weights diverge from the reference. TEXFILTER_QUANTISE implements
// that documented model - and MEASUREMENT SAYS IT IS NOT WHAT THE HARDWARE DOES:
//
//   WarpImage.0 vs the CUDA golden vector, exact fp32 weights : rel 1.45e-6
//   WarpImage.0 vs the same vector, floor(w*256)/256          : rel 1.37e-3
//
// Emulating the documented quantisation is ~1000x FURTHER from CUDA. A synthetic
// linear ramp (where trilinear interpolation is exact) confirms the direction:
// CUDA deviates from exact trilinear by 3.32e-5, so its sampler is lossy, but not
// in the way floor(w*256)/256 models. The hardware's actual rounding is not
// established.
//
// So the default is exact fp32 weights because that is empirically the CLOSEST
// AVAILABLE MATCH to the reference - not because it is more accurate. Accuracy is
// not the goal here; agreement is. TEXFILTER_QUANTISE is retained as a diagnostic
// for re-testing this on other hardware (the Metal port will need it).

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;
override TEXFILTER_QUANTISE : bool = false;

struct Params {
    sz   : vec4<i32>,     // x, y, z, unused
    res  : vec4<f32>,
    dir0 : vec4<f32>, dir1 : vec4<f32>, dir2 : vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       img   : array<f32>;   // scalar volume
@group(0) @binding(1) var<storage, read>       field : array<f32>;   // 3-component
@group(0) @binding(2) var<storage, read_write> outp  : array<f32>;
@group(0) @binding(3) var<uniform>             P     : Params;

fn fetch(i : i32, j : i32, k : i32) -> f32 {
    // border: anything outside the array reads as zero
    if (i < 0 || j < 0 || k < 0 || i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) {
        return 0.0;
    }
    return img[(k * P.sz.y + j) * P.sz.x + i];
}

fn weight(a : f32) -> f32 {
    if (TEXFILTER_QUANTISE) {
        // CUDA stores the filter weight in 1.8 fixed point.
        return floor(a * 256.0) / 256.0;
    }
    return a;
}

// Trilinear sample at unnormalised coordinates, zero outside the volume.
// CUDA is handed (iw+0.5) and internally subtracts 0.5, so the base index is
// floor(iw) and the weight is frac(iw) - no half-texel juggling needed here.
fn sample_border(iw : f32, jw : f32, kw : f32) -> f32 {
    let fx = floor(iw); let fy = floor(jw); let fz = floor(kw);
    let ax = weight(iw - fx); let ay = weight(jw - fy); let az = weight(kw - fz);
    let ix = i32(fx); let iy = i32(fy); let iz = i32(fz);

    let c000 = fetch(ix,     iy,     iz);
    let c100 = fetch(ix + 1, iy,     iz);
    let c010 = fetch(ix,     iy + 1, iz);
    let c110 = fetch(ix + 1, iy + 1, iz);
    let c001 = fetch(ix,     iy,     iz + 1);
    let c101 = fetch(ix + 1, iy,     iz + 1);
    let c011 = fetch(ix,     iy + 1, iz + 1);
    let c111 = fetch(ix + 1, iy + 1, iz + 1);

    let x00 = c000 * (1.0 - ax) + c100 * ax;
    let x10 = c010 * (1.0 - ax) + c110 * ax;
    let x01 = c001 * (1.0 - ax) + c101 * ax;
    let x11 = c011 * (1.0 - ax) + c111 * ax;

    let y0 = x00 * (1.0 - ay) + x10 * ay;
    let y1 = x01 * (1.0 - ay) + x11 * ay;

    return y0 * (1.0 - az) + y1 * az;
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);
    if (i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    let vox = (k * P.sz.y + j) * P.sz.x + i;

    let d0 = P.dir0.x; let d1 = P.dir0.y; let d2 = P.dir0.z;
    let d3 = P.dir1.x; let d4 = P.dir1.y; let d5 = P.dir1.z;
    let d6 = P.dir2.x; let d7 = P.dir2.y; let d8 = P.dir2.z;

    let fi = f32(i); let fj = f32(j); let fk = f32(k);

    let x = d0 * P.res.x * fi + d1 * P.res.y * fj + d2 * P.res.z * fk;
    let y = d3 * P.res.x * fi + d4 * P.res.y * fj + d5 * P.res.z * fk;
    let z = d6 * P.res.x * fi + d7 * P.res.y * fj + d8 * P.res.z * fk;

    let xw = x + field[3 * vox + 0];
    let yw = y + field[3 * vox + 1];
    let zw = z + field[3 * vox + 2];

    let iw = (d0 * xw + d3 * yw + d6 * zw) / P.res.x;
    let jw = (d1 * xw + d4 * yw + d7 * zw) / P.res.y;
    let kw = (d2 * xw + d5 * yw + d8 * zw) / P.res.z;

    outp[vox] = sample_border(iw, jw, kw);
}
