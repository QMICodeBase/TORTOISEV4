// Port of cuda_image_utilities.cu : ComposeFields_kernel
//
// Composition of two displacement fields: walk to the point the main field maps
// to, sample the update field there, and return main + update expressed as a
// displacement from the original grid point.
//
// Boundary rule is a third distinct behaviour, different from both WarpImage and
// ResampleImage (CLAUDE.md 5.2): when the sampling point leaves [0, N-1] the output
// falls back to the MAIN field's displacement, it is neither zeroed nor blended.
//
// Interpolation order (lerp z, then y, then x) and the per-term division by
// spacing follow the reference exactly - both affect the last bits.

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;

struct Params {
    sz   : vec4<i32>,
    spc  : vec4<f32>,
    dir0 : vec4<f32>, dir1 : vec4<f32>, dir2 : vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       mainf  : array<f32>;
@group(0) @binding(1) var<storage, read>       updatef: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp   : array<f32>;
@group(0) @binding(3) var<uniform>             P      : Params;

fn uidx(i : i32, j : i32, k : i32, m : i32) -> i32 {
    return ((k * P.sz.y + j) * P.sz.x + i) * 3 + m;
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);
    if (i >= P.sz.x || j >= P.sz.y || k >= P.sz.z) { return; }

    let n = uidx(i, j, k, 0);

    let d0 = P.dir0.x; let d1 = P.dir0.y; let d2 = P.dir0.z;
    let d3 = P.dir1.x; let d4 = P.dir1.y; let d5 = P.dir1.z;
    let d6 = P.dir2.x; let d7 = P.dir2.y; let d8 = P.dir2.z;
    let sx = P.spc.x;  let sy = P.spc.y;  let sz = P.spc.z;

    let fi = f32(i); let fj = f32(j); let fk = f32(k);

    var x : array<f32, 3>;
    x[0] = d0 * sx * fi + d1 * sy * fj + d2 * sz * fk;
    x[1] = d3 * sx * fi + d4 * sy * fj + d5 * sz * fk;
    x[2] = d6 * sx * fi + d7 * sy * fj + d8 * sz * fk;

    var xp : array<f32, 3>;
    xp[0] = x[0] + mainf[n];
    xp[1] = x[1] + mainf[n + 1];
    xp[2] = x[2] + mainf[n + 2];

    let iw = (d0 * xp[0] + d3 * xp[1] + d6 * xp[2]) / sx;
    let jw = (d1 * xp[0] + d4 * xp[1] + d7 * xp[2]) / sy;
    let kw = (d2 * xp[0] + d5 * xp[1] + d8 * xp[2]) / sz;

    if (iw < 0.0 || iw > f32(P.sz.x - 1) ||
        jw < 0.0 || jw > f32(P.sz.y - 1) ||
        kw < 0.0 || kw > f32(P.sz.z - 1)) {
        // outside: keep the main field's displacement unchanged
        outp[n]     = mainf[n];
        outp[n + 1] = mainf[n + 1];
        outp[n + 2] = mainf[n + 2];
        return;
    }

    let fx = i32(floor(iw)); let fy = i32(floor(jw)); let fz = i32(floor(kw));
    let cx = i32(ceil(iw));  let cy = i32(ceil(jw));  let cz = i32(ceil(kw));
    let xd = iw - f32(fx);
    let yd = jw - f32(fy);
    let zd = kw - f32(fz);

    for (var m = 0; m < 3; m = m + 1) {
        let ia1 = updatef[uidx(fx, fy, fz, m)];
        let ja1 = updatef[uidx(cx, fy, fz, m)];
        let ia2 = updatef[uidx(fx, cy, fz, m)];
        let ja2 = updatef[uidx(cx, cy, fz, m)];
        let ib1 = updatef[uidx(fx, fy, cz, m)];
        let jb1 = updatef[uidx(cx, fy, cz, m)];
        let ib2 = updatef[uidx(fx, cy, cz, m)];
        let jb2 = updatef[uidx(cx, cy, cz, m)];

        let i1 = ia1 * (1.0 - zd) + ib1 * zd;
        let i2 = ia2 * (1.0 - zd) + ib2 * zd;
        let j1 = ja1 * (1.0 - zd) + jb1 * zd;
        let j2 = ja2 * (1.0 - zd) + jb2 * zd;

        let w1 = i1 * (1.0 - yd) + i2 * yd;
        let w2 = j1 * (1.0 - yd) + j2 * yd;

        let update = w1 * (1.0 - xd) + w2 * xd;
        outp[n + m] = xp[m] + update - x[m];
    }
}
