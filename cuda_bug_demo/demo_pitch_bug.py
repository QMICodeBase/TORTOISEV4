#!/usr/bin/env python3
"""Demonstrate the FieldFindMaxLocalNorm pitched-indexing bug in TORTOISE's CUDA backend.

Needs only NumPy. No GPU, no CUDA toolkit, no TORTOISE build.

WHAT IT SHOWS
    TORTOISE reduces the maximum spacing-normalised norm over a 3-component
    displacement field to get `magnitude`, which then normalises the whole DRBUDDI
    update field on every iteration.

    The CUDA kernel walks the buffer FLAT -- gArr[3*i], gArr[3*i+1], gArr[3*i+2] --
    over  pitch/sizeof(float)/3 * sz.y * sz.z  elements. But the buffer comes from
    cudaMalloc3D and is PITCHED: every row is padded out to a 512-byte boundary.
    Triples therefore line up with voxels only when (pitch/4) % 3 == 0.

    This script builds the same pitched buffer NumPy-side, computes the maximum both
    ways, and prints the ratio. Widths where (pitch/4) % 3 != 0 disagree.

USAGE
    python3 demo_pitch_bug.py                # sweep a range of field widths
    python3 demo_pitch_bug.py --padding 200  # model non-zero row padding
    python3 demo_pitch_bug.py --seed 1

REFERENCE
    The correct answer is what TORTOISE's own CPU/ITK implementation computes:
    src/main/drbuddi_image_utilities.cxx, ScaleUpdateField() -- it walks
    sz[2] x sz[1] x sz[0] by ITK index and takes the max per-voxel norm.
"""
import argparse
import numpy as np

PITCH_GRANULARITY = 512          # cudaMalloc3D on current NVIDIA hardware


def pitch_bytes(sz_x, ncomp=3, itemsize=4, gran=PITCH_GRANULARITY):
    """Row stride cudaMalloc3D would return for one row of `sz_x` voxels."""
    row = sz_x * ncomp * itemsize
    return ((row + gran - 1) // gran) * gran


def make_field(sz, spacing, rng):
    """A displacement field with one sharp isolated maximum, like a real update field."""
    nx, ny, nz = sz
    f = rng.normal(0.0, 1.0, size=(nz, ny, nx, 3)).astype(np.float32)
    f[nz // 2, ny // 2, nx // 2] = np.array(spacing, dtype=np.float32) * 50.0
    return f


def reference_max(field, spacing):
    """What the CPU/ITK implementation computes: max per-voxel norm over REAL voxels."""
    v = field.reshape(-1, 3).astype(np.float64)
    n = (v / np.asarray(spacing)) ** 2
    return np.sqrt(n.sum(axis=1)).max()


def cuda_max(field, spacing, padding=0.0):
    """What the CUDA kernel computes: flat 3*i sweep over a PITCHED buffer."""
    nz, ny, nx, nc = field.shape
    pf = pitch_bytes(nx, nc) // 4                       # floats per padded row
    buf = np.full((nz * ny, pf), padding, dtype=np.float32)
    buf[:, : nx * nc] = field.reshape(nz * ny, nx * nc)  # data; rest stays padding
    flat = buf.ravel()

    n_triples = pf // 3 * ny * nz                       # the kernel's arraySize
    g = flat[: n_triples * 3].reshape(-1, 3).astype(np.float64)
    n = (g / np.asarray(spacing)) ** 2
    return np.sqrt(n.sum(axis=1)).max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--padding", type=float, default=0.0,
                    help="value in the uninitialised row padding (default 0). "
                         "CUDAIMAGE::Allocate never clears it, so in reality it is "
                         "whatever the allocator returned. It only changes the result "
                         "when its normalised norm exceeds the field maximum - try 200")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--widths", type=int, nargs="+",
                    default=[25, 42, 50, 70, 100, 128, 140, 175])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    spacing = (1.5, 2.5, 3.5)        # anisotropic: the axis mix-up then shows

    print(f"spacing = {spacing}   row padding modelled as {args.padding}")
    print()
    print(f"{'sz.x':>5} {'pitch':>7} {'pitch/4':>8} {'%3':>3} "
          f"{'reference':>14} {'CUDA':>14} {'CUDA/ref':>10}  ")
    print("-" * 70)

    bad = 0
    for nx in args.widths:
        sz = (nx, 17, 9)
        field = make_field(sz, spacing, rng)
        ref = reference_max(field, spacing)
        cud = cuda_max(field, spacing, args.padding)
        pb = pitch_bytes(nx)
        pf = pb // 4
        ratio = cud / ref
        ok = abs(ratio - 1.0) < 1e-6
        if not ok:
            bad += 1
        print(f"{nx:>5} {pb:>7} {pf:>8} {pf % 3:>3} "
              f"{ref:>14.6f} {cud:>14.6f} {ratio:>10.6f}  "
              f"{'ok' if ok else 'WRONG'}")

    print()
    print(f"{bad} of {len(args.widths)} widths disagree with the reference.")
    print()
    print("Alignment is necessary but NOT sufficient. The sweep also covers row")
    print("padding that CUDAIMAGE::Allocate never initialises (it memsets extent.width")
    print("bytes per row, not pitch). If that memory happens to hold values larger than")
    print("the field's own maximum, even an ALIGNED width returns garbage:")
    print("    python3 demo_pitch_bug.py --padding 200")
    print("(200 is chosen so its spacing-normalised norm exceeds the field maximum;")
    print(" smaller values are simply dominated by real data and change nothing.)")


if __name__ == "__main__":
    main()
