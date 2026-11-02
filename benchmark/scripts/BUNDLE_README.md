# TORTOISE WebGPU — golden-vector reference suite

A self-contained correctness suite for porting the TORTOISE WebGPU/WGSL backend to a new
platform (macOS/Metal in particular). Everything needed to grade a backend is in this
directory. **No CUDA, no NVIDIA hardware, and no TORTOISE datasets are required.**

---

## 1. Why this works without CUDA

Each record is a directory containing `record.json` plus raw fp32 blobs, holding
`{op, params, inputs, outputs}` captured at a CUDA host wrapper during a real pipeline run.
The **`out` tensors are CUDA's own output.** So replaying a record on a new backend compares
that backend against the CUDA reference directly, from data.

`webgpu_replay` links no CUDA libraries. Verified with `ldd`: zero.

What this bundle **cannot** do is create new records. That requires `TORTOISEProcess_cuda` on
an NVIDIA machine. Capture what you need before leaving such a machine.

---

## 2. Quick start

```bash
# 1. integrity — did the records survive the transfer?
python3 verify_bundle.py

# 2. build webgpu_replay for this platform (from the TORTOISE source tree)
#    see section 6 for the macOS adapter/backend environment

# 3. grade the backend against every suite
./verify.sh /path/to/bin/webgpu_replay
```

`verify.sh` exits non-zero if integrity fails or any suite fails.

---

## 3. What is in here

| tree | records | ops | what it is for |
|---|---:|---:|---|
| `records/fast/` | 132 | 20 | **the coverage set** — carries a structural image, so it runs all 4 DRBUDDI metrics and is a superset of the others |
| `records/medium/` | 118 | 18 | non-overfitting + memory gate. No structural, so `ComputeMetric_CCJacS`/`CCSK` never run |
| `records/slow/` | 106 | 16 | **large-matrix** coverage (140×140×92). One DRBUDDI metric only, so it also lacks `ComputeMetric_CC` and `AddToUpdateField`. It adds *scale*, not new ops |
| `records/synthetic/` | 25 | 12 | **exact analytic references** (`reference: "exact"`) — see §5 |

All 20 reachable ops appear in `fast`. The full list:

```
AddImages          AddToUpdateField    ComposeFields       ComputeJointEntropy
ComputeMetric_CC   ComputeMetric_CCJacS ComputeMetric_CCSK ComputeMetric_MSJac
ContrainDefFields  GaussianSmoothImage InvertField         MultiplyImage
MultiplyImages     PreprocessImage     QuadraticTransformImageC  ResampleImage
RestrictPhase      ScaleUpdateField    SumImage            WarpImage
```

`ContrainDefFields` is spelled as upstream spells it. Reachability was determined by **runtime
capture**, not by grepping: static counting suggested 23 wrappers, an instrumented run captured 20.
The three absentees (`ComputeEntropy`, `NegateField`, `ComputeImageGradientImg`) are dead — their
apparent callers are the CPU/ITK namesakes.

---

## 4. Tolerances — and why you must not relax them

Every number here was derived from a measurement. They are floors, not preferences.

| class | vs CUDA | per-element | notes |
|---|---:|---:|---|
| elementwise / reduction | `1e-5` | `1e-4` | |
| texture-sampling (`WarpImage`, `QuadraticTransformImageC`) | `5e-3` | `2e-2` | deliberately loose — see below |
| guarded (`ComposeFields`, `ResampleImage`, `QuadraticTransformImageC`, `InvertField`) | as above, plus ≤ 0.1 % of elements may exceed | `1e-3` | |
| synthetic `reference: "exact"` | `1e-5` | `1e-4` | the real gate for sampling ops |

**Why 1e-5 and not 1e-6.** The same CUDA source compiled with `-fmad=false` disagrees with itself
by **2.67e-6**. nvcc contracts `a*b+c` into FMA by default; WGSL has no contraction control. A 1e-6
gate would sit *below the reference implementation's own noise floor*. 1e-5 is ≈4× the measured
compiler-induced spread; a real porting bug (wrong index, wrong boundary rule, wrong interpolation
order) is orders of magnitude larger.

**Why texture ops get 5e-3 against CUDA.** CUDA's hardware sampler is *itself lossy* — it misses
exact trilinear by 3.3e-5 on a linear ramp. That gate bounds CUDA's own error and catches gross
faults only. **The real correctness gate for sampling is the synthetic exact records at 1e-5.**

**Two gates run, not one.** The max-normalised `max|a-b|/max|b|` is a single global allowance set by
the largest voxel, so a per-element gate `|a-b| ≤ tol·(|b| + rms(ref))` runs alongside it. Be
honest about how much that buys: `rms` is outlier-dominated, and the measured effect is a median
band shrink of only 1.08×. It tightens the small elementwise ops and barely touches the metric
kernels.

**A failure is information.** If a suite fails, find the cause. Do not move a threshold to clear it.

---

## 5. The synthetic records exist because CUDA is the less accurate implementation

For texture-sampled ops, validating only against CUDA means measuring against a known-lossy
reference. The synthetic records carry outputs computed **analytically** — a linear field
interpolates exactly under trilinear interpolation, so a sampling bug cannot satisfy them.

They also reach edge cases the captured vectors never do: border overhang, oblique direction
matrices, a degenerate single-slice volume, and the 3-component field path.

---

## 6. `ScaleUpdateField` — a CUDA bug that was found and fixed

Earlier revisions of this bundle shipped three `records/slow/` entries that were **expected to fail**
against CUDA, because CUDA's `FieldFindMaxLocalNorm` read a pitched 3-component field with flat
`3*i` indexing and produced a wrong maximum for most field widths.

**That bug is fixed** (kernel plus two call sites in `src/cuda_src/cuda_image_utilities.cu`), and
every record in this bundle was recaptured against the fixed build. There are no expected
divergences left: the WebGPU backend and CUDA agree on all six `ScaleUpdateField` records, four of
them **bit-exactly**.

You should therefore see a clean pass with **no `XDIVERGE` lines**. If you do see one, you are
running against a stale bundle — check `MANIFEST.json`'s `capture_binaries` entry.

The bug and a GPU-free NumPy reproducer are written up in the source tree under `cuda_bug_demo/`;
they are worth reading only if you are curious how a pitched-buffer indexing error survives for
years (short answer: the one field width that happens to be aligned, 100, is the one people used).

## 7. Running on macOS / Metal

The backend and adapter policy default to NVIDIA-discrete-Vulkan, which rejects every Apple GPU.
Override them:

```bash
export TORTOISE_WEBGPU_BACKEND=metal
export TORTOISE_WEBGPU_ADAPTER_TYPE=integrated   # Apple Silicon reports IntegratedGPU
export TORTOISE_WEBGPU_VENDOR_ID=0x106B          # Apple; or TORTOISE_WEBGPU_ALLOW_ANY_ADAPTER=1
```

| variable | default | accepts |
|---|---|---|
| `TORTOISE_WEBGPU_BACKEND` | `vulkan` | `vulkan` \| `metal` \| `d3d12` \| `opengl` \| `auto` |
| `TORTOISE_WEBGPU_ADAPTER_TYPE` | `discrete` | `discrete` \| `integrated` \| `any` |
| `TORTOISE_WEBGPU_VENDOR_ID` | `0x10DE` | any vendor ID |

An unrecognised value is a hard error, not a silent fallback. **This recipe has never been executed
on Metal** — it was verified only in the directions testable on Linux.

### Expect these to need attention first

- **Storage-buffer limit.** `metric_ccjacs.wgsl` binds **nine** storage buffers; WebGPU's default
  limit is **eight**. The Vulkan build raises it at device creation; confirm Metal allows the same.
- **No sampler is used anywhere.** `GPUAddressMode` has no `clamp-to-border`, while the CUDA
  textures use `cudaAddressModeBorder`, so all sampling is manual trilinear over storage buffers and
  `CreateTexture()` is a no-op. This is the portable choice — do not "optimise" it into a sampler.
- **Four sampling kernels, three different out-of-domain rules.** `ResampleImage` zeroes the whole
  output voxel; `QuadraticTransformImageC` guards *before* sampling; `WarpImage` is *unguarded*, so
  a one-voxel halo gets nonzero partial values; `ComposeFields` falls back to the main field's
  displacement. Assuming one shared convention silently breaks three of the four.

---

## 8. Interpreting results

- `N passed, 0 failed, 0 unusable, 0 not yet ported` — the suite passed.
- `unusable` — a record failed to load (truncated blob, digest mismatch, schema mismatch). Run
  `verify_bundle.py`; this usually means transfer corruption, not a backend fault.
- `not yet ported` — the op has no implementation registered. Counts as failure.
- A **wrong-shaped output throws** rather than comparing a common prefix. A wrong-dimensioned
  result reporting PASS is the one thing this harness must never do.
- **Matching NaNs count as agreement** — if the reference has NaN at a voxel and so does the
  backend, the port reproduced the reference. A non-finite value in *one* backend only is a failure.

`MANIFEST.json` records per-record SHA-256, the capturing CUDA binary, and the host it came from.
