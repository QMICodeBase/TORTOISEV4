#!/usr/bin/env python3
"""Build synthetic golden-vector records whose expected output is computed
analytically, not captured from CUDA.

Why: CUDA's hardware texture sampler is itself lossy (CLAUDE.md 5.2 - it deviates from
exact trilinear by ~3e-5 on a linear ramp), so comparing the port against CUDA
can only ever bound the error, never pin correctness. A linear field interpolates
exactly under trilinear interpolation, so these records give an exact reference
that a sampling bug cannot satisfy.

They also cover the edge cases the captured vectors do not exercise:
  * samples that fall outside the source domain (border behaviour)
  * degenerate single-voxel-thick volumes
  * non-identity direction matrices

Usage: make_synthetic_records.py <output_dir> [--from <captured_record>]
"""
import json
import os
import sys

import numpy as np


def fnv1a(path):
    h = 1469598103934665603
    with open(path, 'rb') as f:
        for c in f.read():
            h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def tensor(role, name, dims, ncomp, spacing, origin, direction, fname=None):
    t = {"role": role, "name": name, "dims": list(dims), "ncomp": ncomp,
         "spacing": list(spacing), "origin": list(origin), "direction": list(direction)}
    if fname:
        t["file"] = fname
    return t


def sample_trilinear(vol, isz, iw, jw, kw):
    """Exact trilinear with per-neighbour zero border (WarpImage semantics)."""
    fx, fy, fz = int(np.floor(iw)), int(np.floor(jw)), int(np.floor(kw))
    ax, ay, az = iw - fx, jw - fy, kw - fz

    def g(x, y, z):
        if x < 0 or y < 0 or z < 0 or x >= isz[0] or y >= isz[1] or z >= isz[2]:
            return 0.0
        return float(vol[z, y, x])

    x00 = g(fx, fy, fz) * (1 - ax) + g(fx + 1, fy, fz) * ax
    x10 = g(fx, fy + 1, fz) * (1 - ax) + g(fx + 1, fy + 1, fz) * ax
    x01 = g(fx, fy, fz + 1) * (1 - ax) + g(fx + 1, fy, fz + 1) * ax
    x11 = g(fx, fy + 1, fz + 1) * (1 - ax) + g(fx + 1, fy + 1, fz + 1) * ax
    y0 = x00 * (1 - ay) + x10 * ay
    y1 = x01 * (1 - ay) + x11 * ay
    return y0 * (1 - az) + y1 * az


def write_record(root, name, rec, blobs):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    for fname, arr in blobs.items():
        arr.astype(np.float32).tofile(os.path.join(d, fname))
    for t in rec["tensors"]:
        if t.get("file"):
            t["digest"] = fnv1a(os.path.join(d, t["file"]))
            t["bytes"] = float(os.path.getsize(os.path.join(d, t["file"])))
    with open(os.path.join(d, "record.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  wrote {name}")


IDENT = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
# A deliberately oblique direction (a 3-4-5 rotation in-plane) so the port cannot
# pass by accidentally assuming axis-aligned data.
OBLIQUE = [0.8, -0.6, 0.0, 0.6, 0.8, 0.0, 0.0, 0.0, 1.0]


def make_warp(root, tag, sz, direction, shift):
    """WarpImage with a constant displacement: exercises the per-neighbour zero
    border, because a constant shift pushes an edge band outside the volume."""
    zz, yy, xx = np.meshgrid(np.arange(sz[2]), np.arange(sz[1]), np.arange(sz[0]), indexing='ij')
    vol = (1000.0 + 3.0 * xx + 5.0 * yy + 7.0 * zz).astype(np.float32)
    spc = [2.0, 2.0, 2.0]
    D = np.array(direction).reshape(3, 3)

    field = np.zeros((sz[2], sz[1], sz[0], 3), dtype=np.float32)
    field[..., 0], field[..., 1], field[..., 2] = shift

    out = np.zeros((sz[2], sz[1], sz[0]), dtype=np.float32)
    for k in range(sz[2]):
        for j in range(sz[1]):
            for i in range(sz[0]):
                x = D[0, 0] * spc[0] * i + D[0, 1] * spc[1] * j + D[0, 2] * spc[2] * k
                y = D[1, 0] * spc[0] * i + D[1, 1] * spc[1] * j + D[1, 2] * spc[2] * k
                z = D[2, 0] * spc[0] * i + D[2, 1] * spc[1] * j + D[2, 2] * spc[2] * k
                xw, yw, zw = x + shift[0], y + shift[1], z + shift[2]
                iw = (D[0, 0] * xw + D[1, 0] * yw + D[2, 0] * zw) / spc[0]
                jw = (D[0, 1] * xw + D[1, 1] * yw + D[2, 1] * zw) / spc[1]
                kw = (D[0, 2] * xw + D[1, 2] * yw + D[2, 2] * zw) / spc[2]
                out[k, j, i] = sample_trilinear(vol, sz, iw, jw, kw)

    rec = {"schema_version": 2, "op": "WarpImage", "seq": 0,
           "reference": "exact",
           "params": {}, "scalars": {},
           "tensors": [
               tensor("in", "main_image", sz, 1, spc, [0, 0, 0], direction, "in_main_image.f32"),
               tensor("in", "field_image", sz, 3, spc, [0, 0, 0], direction, "in_field_image.f32"),
               tensor("out", "output", sz, 1, spc, [0, 0, 0], direction, "out_output.f32")]}
    write_record(root, f"WarpImage.synth_{tag}", rec,
                 {"in_main_image.f32": vol, "in_field_image.f32": field, "out_output.f32": out})


def make_resample(root, tag, isz, tsz, ispc, tspc, iorig, torig, direction):
    zz, yy, xx = np.meshgrid(np.arange(isz[2]), np.arange(isz[1]), np.arange(isz[0]), indexing='ij')
    vol = (1000.0 + 3.0 * xx + 5.0 * yy + 7.0 * zz).astype(np.float32)
    D = np.array(direction).reshape(3, 3)
    out = np.zeros((tsz[2], tsz[1], tsz[0]), dtype=np.float32)
    for k in range(tsz[2]):
        for j in range(tsz[1]):
            for i in range(tsz[0]):
                b = np.array([torig[0] - iorig[0], torig[1] - iorig[1], torig[2] - iorig[2]]) + \
                    D @ np.array([tspc[0] * i, tspc[1] * j, tspc[2] * k])
                iw = (D[0, 0] * b[0] + D[1, 0] * b[1] + D[2, 0] * b[2]) / ispc[0]
                jw = (D[0, 1] * b[0] + D[1, 1] * b[1] + D[2, 1] * b[2]) / ispc[1]
                kw = (D[0, 2] * b[0] + D[1, 2] * b[1] + D[2, 2] * b[2]) / ispc[2]
                # ResampleImage zeroes the whole voxel outside [0, N-1] (CLAUDE.md 5.2)
                if not (0 <= iw <= isz[0] - 1 and 0 <= jw <= isz[1] - 1 and 0 <= kw <= isz[2] - 1):
                    continue
                out[k, j, i] = sample_trilinear(vol, isz, iw, jw, kw)

    rec = {"schema_version": 2, "op": "ResampleImage", "seq": 0,
           "reference": "exact",
           "params": {}, "scalars": {},
           "tensors": [
               tensor("in", "main_field", isz, 1, ispc, iorig, direction, "in_main_field.f32"),
               tensor("geom", "virtual_img", tsz, 1, tspc, torig, direction),
               tensor("out", "output", tsz, 1, tspc, torig, direction, "out_output.f32")]}
    write_record(root, f"ResampleImage.synth_{tag}", rec,
                 {"in_main_field.f32": vol, "out_output.f32": out})


def gaussian_taps(variance, max_error):
    """Reproduce itk::GaussianOperator::CreateDirectional() taps."""
    import math
    # ITK picks the radius from the error bound, then samples the Gaussian
    # integral over each unit interval; a direct sampling is close enough for a
    # synthetic reference because the SAME taps are handed to both sides.
    sigma = math.sqrt(variance)
    radius = 1
    while radius < 15:
        tail = math.erfc((radius + 0.5) / (sigma * math.sqrt(2.0)))
        if tail < max_error:
            break
        radius += 1
    xs = list(range(-radius, radius + 1))
    w = [math.exp(-0.5 * (x / sigma) ** 2) for x in xs]
    ssum = sum(w)
    return [v / ssum for v in w]


def make_gaussian(root, tag, sz, ncomp, variance):
    """Separable Gaussian with the reference's TRUNCATED, NON-renormalised
    boundary: out-of-range taps are skipped, so edge voxels darken. For ncomp==3
    the reference also zeroes the outer shell and blends the interior."""
    rng = np.random.RandomState(1234 + len(tag))
    vol = (100.0 + 50.0 * rng.rand(sz[2], sz[1], sz[0], ncomp)).astype(np.float32)
    taps = gaussian_taps(variance, 0.001 if ncomp == 3 else 0.01)
    half = len(taps) // 2

    cur = vol.astype(np.float64)
    for axis, extent in ((0, sz[0]), (1, sz[1]), (2, sz[2])):
        nxt = np.zeros_like(cur)
        for k in range(sz[2]):
            for j in range(sz[1]):
                for i in range(sz[0]):
                    pos = (i, j, k)[axis]
                    for c in range(ncomp):
                        val = 0.0
                        for t, w in enumerate(taps):
                            p = pos + t - half
                            if p < 0 or p >= extent:
                                continue          # skipped, NOT renormalised
                            if axis == 0:   val += cur[k, j, p, c] * w
                            elif axis == 1: val += cur[k, p, i, c] * w
                            else:           val += cur[p, j, i, c] * w
                        nxt[k, j, i, c] = val
        cur = nxt

    if ncomp == 3:
        w1 = 1.0 if variance >= 0.5 else 1.0 - variance / 0.5
        w2 = 1.0 - w1
        for k in range(sz[2]):
            for j in range(sz[1]):
                for i in range(sz[0]):
                    edge = (i in (0, sz[0]-1) or j in (0, sz[1]-1) or k in (0, sz[2]-1))
                    for c in range(ncomp):
                        cur[k, j, i, c] = 0.0 if edge else cur[k, j, i, c] * w1 + vol[k, j, i, c] * w2

    spc = [2.0, 2.0, 2.0]
    rec = {"schema_version": 2, "op": "GaussianSmoothImage", "seq": 0,
           "reference": "exact",
           "params": {"std": variance, "kernel": taps}, "scalars": {},
           "tensors": [
               tensor("in", "main_image", sz, ncomp, spc, [0, 0, 0], IDENT, "in_main_image.f32"),
               tensor("out", "output", sz, ncomp, spc, [0, 0, 0], IDENT, "out_output.f32")]}
    write_record(root, f"GaussianSmoothImage.synth_{tag}", rec,
                 {"in_main_image.f32": vol, "out_output.f32": cur.astype(np.float32)})


def make_elementwise(root):
    """Exact references for the P2 elementwise kernels. All computed in float32
    with the reference's own expression grouping, since regrouping changes the
    rounding."""
    rng = np.random.RandomState(99)
    sz = (11, 9, 7)
    spc = [2.0, 2.0, 2.0]

    def rec_of(op, params, tensors, blobs, tag):
        r = {"schema_version": 2, "op": op, "seq": 0, "reference": "exact",
             "params": params, "scalars": {}, "tensors": tensors}
        write_record(root, f"{op}.synth_{tag}", r, blobs)

    for ncomp, tag in ((1, "scalar"), (3, "field")):
        a = (rng.rand(sz[2], sz[1], sz[0], ncomp).astype(np.float32) * 200 - 100)
        b = (rng.rand(sz[2], sz[1], sz[0], ncomp).astype(np.float32) * 200 - 100)
        T = lambda role, name, f=None, nc=ncomp: tensor(role, name, sz, nc, spc, [0, 0, 0], IDENT, f)

        rec_of("AddImages", {}, [T("in", "im1", "in_im1.f32"), T("in", "im2", "in_im2.f32"),
                                 T("out", "output", "out_output.f32")],
               {"in_im1.f32": a, "in_im2.f32": b, "out_output.f32": a + b}, tag)

        rec_of("MultiplyImages", {}, [T("in", "im1", "in_im1.f32"), T("in", "im2", "in_im2.f32"),
                                      T("out", "output", "out_output.f32")],
               {"in_im1.f32": a, "in_im2.f32": b, "out_output.f32": a * b}, tag)

        f = np.float32(-1.7532)
        rec_of("MultiplyImage", {"factor": float(f)},
               [T("in", "im1", "in_im1.f32"), T("out", "output", "out_output.f32")],
               {"in_im1.f32": a, "out_output.f32": a * f}, tag)

    # PreprocessImage rescales a scalar image; keep the reference's grouping
    img = (rng.rand(sz[2], sz[1], sz[0], 1).astype(np.float32) * 500 + 20)
    lo, up = np.float32(0.0), np.float32(1000.0)
    imin, imax = img.min(), img.max()
    out = (up - lo) / (imax - imin) * img - imin * (up - lo) / (imax - imin) + lo
    rec_of("PreprocessImage", {"low_val": float(lo), "up_val": float(up)},
           [tensor("in", "img", sz, 1, spc, [0, 0, 0], IDENT, "in_img.f32"),
            tensor("out", "output", sz, 1, spc, [0, 0, 0], IDENT, "out_output.f32")],
           {"in_img.f32": img, "out_output.f32": out.astype(np.float32)}, "scalar")

    # RestrictPhase projects each vector onto the phase direction; include exact
    # zero vectors, which the reference leaves untouched rather than dividing by 0
    v = (rng.rand(sz[2], sz[1], sz[0], 3).astype(np.float32) * 4 - 2)
    v[0, 0, 0, :] = 0.0
    phase = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    outv = v.copy()
    nrm = np.sqrt((v.astype(np.float32) ** 2).sum(axis=3)).astype(np.float32)
    for k in range(sz[2]):
        for j in range(sz[1]):
            for i in range(sz[0]):
                n = nrm[k, j, i]
                if n != 0:
                    u = v[k, j, i] / n
                    dot = np.float32(u[0] * phase[0] + u[1] * phase[1] + u[2] * phase[2])
                    outv[k, j, i] = phase * n * dot
    rec_of("RestrictPhase", {"phase": [float(x) for x in phase]},
           [tensor("in", "field", sz, 3, spc, [0, 0, 0], IDENT, "in_field.f32"),
            tensor("out", "field", sz, 3, spc, [0, 0, 0], IDENT, "out_field.f32")],
           {"in_field.f32": v, "out_field.f32": outv}, "field")

    # ContrainDefFields makes the pair exact opposites, in place on both
    u = (rng.rand(sz[2], sz[1], sz[0], 3).astype(np.float32) * 6 - 3)
    dn = (rng.rand(sz[2], sz[1], sz[0], 3).astype(np.float32) * 6 - 3)
    val = ((u - dn) * np.float32(0.5)).astype(np.float32)
    rec_of("ContrainDefFields", {},
           [tensor("in", "ufield", sz, 3, spc, [0, 0, 0], IDENT, "in_ufield.f32"),
            tensor("in", "dfield", sz, 3, spc, [0, 0, 0], IDENT, "in_dfield.f32"),
            tensor("out", "ufield", sz, 3, spc, [0, 0, 0], IDENT, "out_ufield.f32"),
            tensor("out", "dfield", sz, 3, spc, [0, 0, 0], IDENT, "out_dfield.f32")],
           {"in_ufield.f32": u, "in_dfield.f32": dn,
            "out_ufield.f32": val, "out_dfield.f32": -val}, "field")



def okan_matrix(ax, ay, az):
    """Rz * Ry * Rx, exactly as OkanQuadraticTransform::ComputeMatrix builds it
    (itkOkanQuadraticTransform.hxx:423-455). The replay tool cross-checks the
    record's `matrix` against the transform it rebuilds from `params`, so this must
    match or the record is rejected."""
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_quadratic(root, tag, isz, tsz, ispc, tspc, iorig, torig, direction, par):
    """Exact reference for QuadraticTransformImageC.

    WHY THIS EXISTS: CLAUDE.md 5.2 designates the synthetic-exact gate as the PRIMARY
    correctness gate for texture-sampling ops, because CUDA's sampler is the lossy
    side. QuadraticTransformImageC - the transform in DIFFPREP's innermost
    registration loop - had NO such record; its only gate was 5e-3 against CUDA, of
    which it already consumed 57%. The captured records are also degenerate:
    identity matrix, every parameter zero except p[7]=1, plus one pure translation.
    Real converged transforms have rotation and quadratic terms nonzero in 137/138
    volumes, so the coordinate-computation path was entirely untested.

    The source volume is linear, so trilinear interpolation is exact and any
    deviation is a coordinate or boundary error rather than interpolation loss.
    """
    zz, yy, xx = np.meshgrid(np.arange(isz[2]), np.arange(isz[1]), np.arange(isz[0]), indexing='ij')
    vol = (100.0 + 3.0 * xx + 5.0 * yy + 7.0 * zz).astype(np.float32)

    D = np.array(direction).reshape(3, 3)
    # Host construction (quadratic_transform_image.cxx): target index->world, and
    # world->source index with the origin shift folded into column 3.
    smat = np.zeros((3, 4))
    for r in range(3):
        for c in range(3):
            smat[r][c] = D[r][c] * tspc[c]
        smat[r][3] = torig[r]
    sinv = np.zeros((3, 4))
    for r in range(3):
        for c in range(3):
            sinv[r][c] = D[c][r] / ispc[r]      # transposed; divides by ROW spacing
        sinv[r][3] = -(sinv[r][0] * iorig[0] + sinv[r][1] * iorig[1] + sinv[r][2] * iorig[2])

    rot = okan_matrix(par[3], par[4], par[5])
    # phase is DERIVED from p6..p8 by the host, not read from the record
    a6, a7, a8 = abs(par[6]), abs(par[7]), abs(par[8])
    phase = 0
    if a7 > a6 and a7 > a8: phase = 1
    if a8 > a6 and a8 > a7: phase = 2
    do_cubic = any(abs(np.float32(par[q])) > 1e-10 for q in range(14, 21))

    out = np.zeros((tsz[2], tsz[1], tsz[0]), dtype=np.float32)
    for k in range(tsz[2]):
        for j in range(tsz[1]):
            for i in range(tsz[0]):
                v = np.array([i, j, k], dtype=float)
                w = smat[:, :3] @ v + smat[:, 3]
                w = w - np.array([par[21], par[22], par[23]])
                q = rot @ w + np.array([par[0], par[1], par[2]])
                x1, y1, z1 = q
                npc = (par[6] * x1 + par[7] * y1 + par[8] * z1
                       + par[9] * x1 * y1 + par[10] * x1 * z1 + par[11] * y1 * z1
                       + par[12] * (x1 * x1 - y1 * y1)
                       + par[13] * (2 * z1 * z1 - x1 * x1 - y1 * y1))
                cub = 0.0
                if do_cubic:
                    cub = (par[14] * x1 * y1 * z1
                           + par[15] * z1 * (x1 * x1 - y1 * y1)
                           + par[16] * x1 * (4 * z1 * z1 - x1 * x1 - y1 * y1)
                           + par[17] * y1 * (4 * z1 * z1 - x1 * x1 - y1 * y1)
                           + par[18] * x1 * (x1 * x1 - 3 * y1 * y1)
                           + par[19] * y1 * (3 * x1 * x1 - y1 * y1)
                           + par[20] * z1 * (2 * z1 * z1 - 3 * x1 * x1 - 3 * y1 * y1))
                q2 = [x1, y1, z1]
                q2[phase] = npc + cub
                iw, jw, kw = (sinv[:, :3] @ np.array(q2)) + sinv[:, 3]
                # Guarded BEFORE sampling; out-of-domain voxels keep the wrapper's
                # memset zero (CLAUDE.md 5.2). Differs from WarpImage's per-neighbour border.
                if not (0 <= iw <= isz[0] - 1 and 0 <= jw <= isz[1] - 1 and 0 <= kw <= isz[2] - 1):
                    continue
                out[k, j, i] = sample_trilinear(vol, isz, iw, jw, kw)

    rec = {"schema_version": 2, "op": "QuadraticTransformImageC", "seq": 0,
           "reference": "exact",
           "params": {"params": [float(x) for x in par],
                      "phase": int(phase),
                      "matrix": [float(x) for x in rot.reshape(9)]},
           "scalars": {},
           "tensors": [
               tensor("in", "main_image", isz, 1, ispc, iorig, direction, "in_main_image.f32"),
               tensor("geom", "target_img", tsz, 1, tspc, torig, direction),
               tensor("out", "output", tsz, 1, tspc, torig, direction, "out_output.f32")]}
    write_record(root, f"QuadraticTransformImageC.synth_{tag}", rec,
                 {"in_main_image.f32": vol, "out_output.f32": out})



def make_joint_entropy(root, tag, sz, nbins=100):
    """Exact reference for ComputeJointEntropy.

    WHY: the captured entropy records cannot be gated tightly. The statistic is
    QUANTISED - moving one sample between two count-1 histogram bins changes
    sum(p*log p) by exactly 2*ln2/N - and every captured record's quantum exceeds
    the 1e-5 elementwise gate (fast 3.5e-5, medium 1.4e-5). They therefore carry a
    derived absolute floor, which necessarily weakens them. This record recovers the
    tight check by removing the quantisation from the comparison entirely.

    HOW: every voxel value is placed at the CENTRE of its Parzen bin, so the
    truncation `(int)(val/binsize - normalizedMin)` is unambiguous - no value sits
    within rounding distance of a boundary, so the histogram is identical on any
    backend and the only thing left to measure is the p*log(p) arithmetic and its
    summation order. The bin index, the PADDING=2 clamp and the >=low/<=high
    admission test are replicated from compute_entropy.cu:263-290.
    """
    PADDING = 2
    nx, ny, nz = sz
    n = nx * ny * nz

    low1, high1 = 0.1, 8000.0
    low2, high2 = 0.1, 5000.0
    bs1 = (high1 - low1) / (nbins - 2 * PADDING)
    bs2 = (high2 - low2) / (nbins - 2 * PADDING)
    nmin1 = low1 / bs1 - PADDING
    nmin2 = low2 / bs2 - PADDING

    # Spread samples over the usable bin range [2, nbins-3], at bin centres.
    usable = np.arange(2, nbins - 2)
    rng = np.random.default_rng(7)
    b1 = rng.choice(usable, size=n)
    b2 = rng.choice(usable, size=n)
    v1 = ((b1 + 0.5) + nmin1) * bs1
    v2 = ((b2 + 0.5) + nmin2) * bs2

    img1 = v1.astype(np.float32).reshape(nz, ny, nx)
    img2 = v2.astype(np.float32).reshape(nz, ny, nx)

    # Recover the indices the kernel will actually compute, from the float32 values
    # that will be stored - not from b1/b2 - so the reference matches what is read.
    i1 = (img1.reshape(-1).astype(np.float64) / bs1 - nmin1).astype(np.int32)
    i2 = (img2.reshape(-1).astype(np.float64) / bs2 - nmin2).astype(np.int32)
    i1 = np.clip(i1, 2, nbins - 3)
    i2 = np.clip(i2, 2, nbins - 3)

    joint = np.zeros((nbins, nbins), dtype=np.float64)
    np.add.at(joint, (i1, i2), 1.0)
    m1 = joint.sum(axis=1)
    m2 = joint.sum(axis=0)

    def entropy(h):
        s = h.sum()
        p = h / s
        nz_ = p > 1e-10          # the reference's cutoff (compute_entropy.cu)
        return float((p[nz_] * np.log(p[nz_])).sum())

    rec = {"schema_version": 2, "op": "ComputeJointEntropy", "seq": 0,
           "reference": "exact",
           "params": {"Nbins": float(nbins), "lims": [low1, high1, low2, high2]},
           "scalars": {"entropy_joint": entropy(joint),
                       "entropy_img1": entropy(m1),
                       "entropy_img2": entropy(m2)},
           "tensors": [
               tensor("in", "img1", sz, 1, [1, 1, 1], [0, 0, 0], IDENT, "in_img1.f32"),
               tensor("in", "img2", sz, 1, [1, 1, 1], [0, 0, 0], IDENT, "in_img2.f32")]}
    write_record(root, f"ComputeJointEntropy.synth_{tag}", rec,
                 {"in_img1.f32": img1.astype(np.float32),
                  "in_img2.f32": img2.astype(np.float32)})



def make_scale_update_field(root, tag, sz, spc, scale_factor=0.37):
    """Exact reference for ScaleUpdateField - the ONE sanctioned deliberate divergence.

    CLAUDE.md 0.0b records that CUDA's FieldFindMaxLocalNorm reads a pitched
    3-component field with FLAT indexing over pitch/4/3*y*z elements, so component
    triples drift out of alignment with voxels whenever (pitch/4) % 3 != 0, and the
    sweep also covers uninitialised row padding. The port computes the true per-voxel
    maximum, matching TORTOISE's own CPU implementation
    (drbuddi_image_utilities.cxx:229-260).

    WHY THIS RECORD EXISTS. An audit showed the divergence is DORMANT on every
    captured record: CUDA's applied factor agrees with the intended magnitude to six
    digits (padding evidently reads as zero, and the mis-grouped maximum happens to
    coincide with the true one). Four of six records per dataset apply an exact power
    of two. So `132/132` and `118/118` could not be read as evidence that
    ScaleUpdateField is either faithful or correctly divergent - a fault in it was
    invisible.

    This record makes the intended semantics testable: anisotropic spacing (so a
    mis-assigned axis changes the norm), a width whose pitch drifts, and a single
    sharp isolated maximum placed so that any mis-grouping picks a different voxel.
    The expected output is computed from the TRUE per-voxel maximum, i.e. what the
    port and the CPU implementation both intend.
    """
    nx, ny, nz = sz
    rng = np.random.default_rng(11)
    f = (rng.standard_normal((nz, ny, nx, 3)) * 0.01).astype(np.float32)
    # One sharp, isolated maximum, off-centre so a drifted triple cannot reproduce it.
    f[nz // 2, ny // 2, nx // 2] = np.array([3.0, -5.0, 7.0], dtype=np.float32)

    v = f.reshape(-1, 3).astype(np.float64) / np.asarray(spc, dtype=np.float64)
    magnitude = float(np.sqrt((v ** 2).sum(axis=1)).max())
    out = (f.astype(np.float64) * (scale_factor / magnitude)).astype(np.float32)

    rec = {"schema_version": 2, "op": "ScaleUpdateField", "seq": 0,
           "reference": "exact",
           "params": {"scale_factor": scale_factor}, "scalars": {},
           "tensors": [
               tensor("in", "field", sz, 3, spc, [0, 0, 0], IDENT, "in_field.f32"),
               tensor("out", "field", sz, 3, spc, [0, 0, 0], IDENT, "out_field.f32")]}
    write_record(root, f"ScaleUpdateField.synth_{tag}", rec,
                 {"in_field.f32": f, "out_field.f32": out})


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "synthetic_vectors"
    os.makedirs(root, exist_ok=True)
    print(f"synthetic records -> {root}")

    # interior sampling, axis aligned
    make_warp(root, "interior", (16, 14, 12), IDENT, (0.7, 0.3, 0.4))
    # large shift: forces a band of samples outside the volume -> border behaviour
    make_warp(root, "border", (16, 14, 12), IDENT, (5.5, -4.5, 3.5))
    # oblique direction matrix
    make_warp(root, "oblique", (12, 12, 10), OBLIQUE, (1.3, 0.9, -0.7))
    # degenerate: single slice
    make_warp(root, "thin", (9, 9, 1), IDENT, (0.5, 0.5, 0.0))

    make_resample(root, "grid", (20, 18, 16), (13, 11, 9),
                  [2.0, 2.0, 2.0], [3.1, 3.1, 3.1], [0, 0, 0], [1.0, 1.0, 1.0], IDENT)
    # target grid deliberately overhangs the source -> exercises the zeroing rule
    make_resample(root, "overhang", (10, 10, 8), (16, 16, 12),
                  [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0, 0, 0], [-6.0, -6.0, -4.0], IDENT)
    # scalar smoothing, and the 3-component field path with AdjustFieldBoundary -
    # the latter is NOT covered by any captured vector (see M3 notes)
    make_elementwise(root)

    make_gaussian(root, "scalar", (12, 11, 9), 1, 2.25)
    make_gaussian(root, "field", (10, 9, 8), 3, 2.25)
    make_gaussian(root, "field_smallvar", (9, 8, 7), 3, 0.25)   # exercises the w1/w2 blend

    make_resample(root, "oblique", (14, 14, 10), (11, 11, 8),
                  [2.0, 2.0, 2.0], [2.5, 2.5, 2.5], [0, 0, 0], [1.0, -1.0, 0.5], OBLIQUE)

    # QuadraticTransformImageC - the DIFFPREP-critical op that previously had NO
    # exact-reference record. The captured vectors are identity-matrix with a single
    # nonzero parameter, so none of the coordinate path below was exercised.
    def qpar(**kw):
        p = [0.0] * 24
        p[7] = 1.0                      # phase = 1 (j axis), as the pipeline uses
        for k, v in kw.items():
            p[int(k[1:])] = v
        return p

    # non-identity rotation + translation: exercises rot and par[0..2], par[21..23]
    make_quadratic(root, "rotation", (14, 14, 12), (12, 12, 10),
                   [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0, 0, 0], [1.0, 1.0, 1.0], IDENT,
                   qpar(p0=0.8, p1=-0.6, p2=0.4, p3=0.031, p4=-0.024, p5=0.017,
                        p21=1.5, p22=-1.0, p23=0.5))
    # nonzero quadratic terms p9..p13 - the eddy-current model itself
    make_quadratic(root, "quadratic", (14, 14, 12), (12, 12, 10),
                   [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0, 0, 0], [1.0, 1.0, 1.0], IDENT,
                   qpar(p3=0.021, p5=-0.015, p9=0.004, p10=-0.003, p11=0.0025,
                        p12=0.0018, p13=-0.0012))
    # target grid overhangs the source -> exercises the all-or-nothing domain guard
    make_quadratic(root, "overhang", (10, 10, 8), (16, 16, 12),
                   [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0, 0, 0], [-6.0, -6.0, -4.0], IDENT,
                   qpar(p3=0.02, p9=0.003))
    # Oblique direction matrices on both sides, ANISOTROPIC spacing, and a nonzero
    # source origin. All three together are required: smat_inv divides by ROW
    # spacing while smat multiplies by COLUMN spacing, and that index only shows up
    # in the OFF-DIAGONAL entries - so a row/column mix-up is invisible unless the
    # direction matrix is oblique AND the spacing is anisotropic. Every other record
    # in this suite, and both captured CUDA vectors, are isotropic or axis-aligned,
    # which left the convention untested project-wide. The nonzero iorig likewise
    # exercises the origin term folded into smat_inv column 3.
    # Entropy at bin centres: removes the quantisation so the p*log(p) path and the
    # summation order can be checked tightly, which the captured records cannot do.
    make_joint_entropy(root, "bincentres", (20, 25, 17))

    # ScaleUpdateField at a DRIFTING width with anisotropic spacing - the only test
    # of the one sanctioned deliberate divergence (CLAUDE.md 0.0b). nx=39 gives
    # row bytes 468 -> pitch 512 -> pitch/4 = 128, 128 % 3 = 2, so CUDA's flat
    # triple indexing drifts. Anisotropic spacing makes a mis-assigned axis change
    # the computed norm rather than cancel.
    make_scale_update_field(root, "drift_aniso", (39, 17, 9), [1.5, 2.5, 3.5])

    make_quadratic(root, "oblique", (13, 13, 11), (11, 11, 9),
                   [2.0, 2.5, 3.0], [2.4, 2.2, 2.8], [-3.5, 2.0, -1.25], [0.5, -0.5, 0.25],
                   OBLIQUE, qpar(p0=0.5, p3=0.026, p5=0.019, p9=0.0035, p12=0.002))


if __name__ == '__main__':
    main()
