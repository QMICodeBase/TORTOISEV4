# `FieldFindMaxLocalNorm` reads a pitched buffer with flat indexing

**File:** `src/cuda_src/cuda_image_utilities.cu`
**Affects:** DRBUDDI (`ScaleUpdateField_cuda`, `ComputeFieldScale_cuda`) — CUDA builds only.
**Symptom:** the DRBUDDI update field is normalised by a wrong `magnitude`, for most field
widths. Not a crash; results are quietly off.

---

## The bug

`FieldFindMaxLocalNorm` reduces the largest spacing-normalised norm over a 3-component
displacement field. It indexed the buffer **flat**:

```c
for (int i = gthIdx; i < arraySize; i += gridSize)
{
    float sm = (gArr[3*i  ]/spc.x)*(gArr[3*i  ]/spc.x) +
               (gArr[3*i+1]/spc.y)*(gArr[3*i+1]/spc.y) +
               (gArr[3*i+2]/spc.z)*(gArr[3*i+2]/spc.z);
    ...
}
```

launched with

```c
FieldFindMaxLocalNorm<<<gSize, bSize>>>((float *)field.ptr,
        field.pitch/sizeof(float)/3*data_sz.y*data_sz.z, spc, dev_out);
```

The buffer comes from `cudaMalloc3D`, so it is **pitched**: each row is padded out to a
512-byte boundary. Consecutive triples line up with voxels only when

```
(pitch / sizeof(float)) % 3 == 0
```

`cudaMalloc3D` returns 512-byte-granular pitches, so `pitch/4` is a multiple of 128, and
`128 % 3 == 2`. Whether the condition holds therefore depends on the field width:

| `sz.x` | row bytes | pitch | `pitch/4` | `% 3` | |
|---:|---:|---:|---:|---:|---|
| 25 | 300 | 512 | 128 | 2 | drifts |
| 42 | 504 | 512 | 128 | 2 | drifts |
| 50 | 600 | 1024 | 256 | 1 | drifts |
| 70 | 840 | 1024 | 256 | 1 | drifts |
| **100** | 1200 | 1536 | 384 | **0** | aligned |
| 128 | 1536 | 1536 | 384 | **0** | aligned, and no padding at all |
| **140** | 1680 | 2048 | 512 | 2 | drifts |
| 175 | 2100 | 2560 | 640 | 1 | drifts |

Two things go wrong when it drifts:

1. **Components are mis-grouped.** Each "voxel" the reduction sees is assembled from
   components of *different* voxels straddling a row boundary — and their axes are
   mis-assigned, so a *z* component gets divided by `spc.x`, and so on.
2. **It reads uninitialised memory.** `arraySize` is derived from `pitch`, so the sweep
   covers the row padding. `CUDAIMAGE::Allocate` (`src/cuda_src/cuda_image.cxx`) clears
   only the data region — it passes `extent.width` bytes per row, not `pitch` — so the
   padding is whatever the allocator returned.

## Why it matters

`magnitude` is the step-size normaliser. `ScaleUpdateField` divides the entire update field
by it, and DRBUDDI calls it on **every iteration** (`src/main/run_drbuddi_stage.cxx`). A
wrong magnitude is a global scale error on the search direction, which sends the optimiser
down a different convergence path — not a rounding difference.

DRBUDDI also runs a multi-resolution pyramid, so most levels have small `sz.x` — exactly
where drift occurs. The alignment at `sz.x = 100` is a coincidence, not the normal case.

## Which implementation is right

TORTOISE's own CPU path already does it correctly:
`src/main/drbuddi_image_utilities.cxx`, `ScaleUpdateField()` walks `sz[2] × sz[1] × sz[0]`
by ITK index and takes the maximum per-voxel norm. The CUDA kernel disagreed with the CPU
implementation of the same function; the CPU one reflects the intent.

---

## Reproducing it without a GPU

```bash
python3 demo_pitch_bug.py
```

NumPy only. It builds the pitched buffer exactly as `cudaMalloc3D` would, fills the data
region with a known field and the padding with a chosen value, then computes the maximum
both ways and prints the ratio:

```
 sz.x   pitch  pitch/4  %3      reference           CUDA   CUDA/ref
   25     512      128   2      86.602540     108.834303   1.256710  WRONG
   42     512      128   2      86.602540     108.832258   1.256687  WRONG
   50    1024      256   1      86.602540     116.667010   1.347155  WRONG
   70    1024      256   1      86.602540     116.669518   1.347184  WRONG
  100    1536      384   0      86.602540      86.602540   1.000000  ok
  128    1536      384   0      86.602540      86.602540   1.000000  ok
  140    2048      512   2      86.602540     108.832190   1.256686  WRONG
  175    2560      640   1      86.602540     116.666727   1.347151  WRONG
```

Alignment is necessary but not sufficient — with non-zero row padding even an aligned width
reads uninitialised memory:

```bash
python3 demo_pitch_bug.py --padding 200
```

`sz.x = 128` survives that too, because `128 × 3 × 4 = 1536` fills the pitch exactly and
there is no padding to read.

### Confirmed on hardware

The model above is not just arithmetic. Running the real CUDA kernel on a 39×17×9 field with
anisotropic spacing and `pitch/4 = 128` (so it drifts), against an analytically computed
reference:

| | relative error |
|---|---:|
| before the fix | **0.204** (20 % wrong) |
| after the fix | **7.97e-08** (float rounding) |

Verified isolated: across a suite of 25 analytic reference records, 24 replay byte-identically
before and after; only the record constructed to expose this indexing bug changes.

---

## The fix

Pass the volume dimensions and the row pitch, and reduce over the real voxels only:

```c
__global__ void
FieldFindMaxLocalNorm(const float *gArr, const int3 sz, const size_t pitch_f,
                      const float3 spc, float *gOut)
{
    ...
    const int nvox = sz.x*sz.y*sz.z;
    for (int i = gthIdx; i < nvox; i += gridSize)
    {
        const int x =  i % sz.x;
        const int y = (i / sz.x) % sz.y;
        const int z =  i / (sz.x*sz.y);
        const float *v = gArr + ((size_t)z*sz.y + y)*pitch_f + 3*x;

        float sm = (v[0]/spc.x)*(v[0]/spc.x) +
                   (v[1]/spc.y)*(v[1]/spc.y) +
                   (v[2]/spc.z)*(v[2]/spc.z);
        ...
    }
```

and at both call sites:

```c
FieldFindMaxLocalNorm<<<gSize, bSize>>>((float *)field.ptr, data_sz,
        field.pitch/sizeof(float), spc, dev_out);
```

That is the whole change — one kernel and two call sites, in
`src/cuda_src/cuda_image_utilities.cu`.

## Note for maintainers

Applying this **changes DRBUDDI output** on any dataset whose field widths drift, which is
most of them. That is the fix working: the corrected magnitude matches what the CPU/ITK path
has always computed. Any stored regression baselines captured from the CUDA path will need
regenerating.
