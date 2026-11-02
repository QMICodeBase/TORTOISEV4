# CLAUDE.md — TORTOISE V4 GPU backend ports (CUDA reference, WebGPU, Metal)

Gotchas, exhausted paths and non-obvious findings. Not a history. Anything here is
here because it cost someone time or would silently break correctness.

Companion documents, **all gitignored** — `benchmark/` is excluded except `scripts/`, so
every `benchmark/...` path below resolves to nothing in a fresh clone: `benchmark/PERF_NOTES.md`
(performance evidence), `benchmark/README.md` (datasets/methodology; ~114 GB, copied between
machines), and the `plan_metal.md` / `plan_faster.md` working plans. **Never make committed
source or docs depend on them.**

---

## 0. The governing principle: MIMIC THE REFERENCE, DO NOT IMPROVE IT

These are *faithful ports*. A port must reproduce the CUDA backend's behaviour **including
its quirks and its absence of defensive checks**. Adding a guard the reference lacks is a
divergence, and a divergence silently invalidates the golden-vector comparison.

**The CUDA kernels contain ZERO NaN/Inf checks** (`grep -n "isnan\|isinf" src/cuda_src/*.cu`
returns nothing). The ports must not add any. Their **only** guards are degeneracy
thresholds, and this is the complete list — each must exist in WGSL and MSL, exactly:

| Guard | Where |
|---|---|
| `LIMCC 1E-10`, `LIMCCSK 1E-5`, `LIMCCJAC 1E-5` | `compute_metric.cu:10-12`, mirrored in `compute_metrics_*.h` |
| `det = -1 + 1E-5` when the Jacobian goes non-positive | `compute_metric.cu:230,501,594,703` |
| `if(detf <= 0) detf = 1E-5;` (and `detF2`) | `compute_metric.cu:1495-1498,1086,1174,1264` |
| `if(nrm != 0)` | `cuda_image_utilities.cu:467` |
| `if(magnitude > 1E-20)` — **easy to miss**; reproduced including CUDA's NaN behaviour | `cuda_image_utilities.cu:413` (`ScaleUpdateField`) |
| `if(magnitude != 0)` | `cuda_image_utilities.cu:316` |

**Aborting on a device failure is faithful, not an addition.** `cuda_utils.h`'s `gpuErrchk`
prints and `exit()`s on any error. So `wgpuctx::Dispatch` aborting on failed validation, and
`mtlctx::Sync()` aborting on a failed command buffer, match the reference's error discipline.
**A failed dispatch that silently leaves buffers untouched is indistinguishable from a
legitimate result — never allow it.**

### `FillBuffer` reproduces a reference bug on purpose — do not "fix" it

`CUDAIMAGE::FillBuffer` (`cuda_image.h:51-55`) calls `cudaMemset3D(ptr, val, extent)`, whose
second parameter is an **`int` byte value**. The float is truncated to int and its low byte
written to every *byte*:

| call | reference result per float |
|---|---|
| `FillBuffer(0)` | `0.0f` (agrees by luck) |
| `FillBuffer(1)` | `0x01010101` = **2.3694e-38**, not `1.0` |
| `FillBuffer(-1)` | `0xFFFFFFFF` = **NaN**, not `-1.0` |

The ports reproduce the byte fill. Reachability was **checked, not assumed**: this overload
has no call site in the GPU pipeline — every apparent one binds to ITK's `FillBuffer`. The
divergence is latent and no golden vector covers it, which is exactly why it is written down.

Two over-defensive checks were added during porting and **deliberately reverted**; do not
re-introduce them: a `components_per_voxel` check in `CudaImageToITKImage` (the reference reads
`NumVoxels()` floats regardless), and zero-filling a destination on failed readback (handing
back zeros makes a GPU failure look like a result).

Conversely `CudaImageToITKField`'s `components_per_voxel != 3` check **is** required — the
reference has it (`cuda_image.cxx:328-329`), and removing it would make the port *less*
defensive than the reference, which §0 forbids in both directions.

### Standing rules

| Rule | Why |
|---|---|
| **Never add a check the reference does not have.** | Above. |
| **Never modify `README.md`.** | It is the upstream project README. |
| **Never relax a tolerance to make a test pass.** | Every tolerance in §3 was derived from a measurement. Find the cause first. |
| **`gpu_replay` (CUDA self-replay) must stay bit-exact.** | It anchors everything else. Unrunnable on macOS (no CUDA) — there the anchor is `metal_replay` against the same records (§2). |
| **Do not "fix" upstream oddities encountered while porting.** | Except the swapped `kernel<<<blockSize, gridSize>>>` launches, which really were swapped — fixed for 46 elementwise launches (`PERF_NOTES.md` §14). Reductions keep their geometry: block size sets summation order. |

Non-goals: algorithmic changes, DRTAMAS, TVVF, any CUDA path not reachable from the default
DIFFPREP+DRBUDDI pipeline. The only fp64 kernels are DRTAMAS-only, so **no fp64 emulation is
needed** — neither WebGPU nor Metal has it.

---

## 1. Layout and build

| Path | Contents |
|---|---|
| `src/cuda_src/` | CUDA backend (reference) + `gpu_capture.{h,cxx}` (golden-vector hook) |
| `src/webgpu_src/` | WebGPU/Dawn backend, namespace `wgpuctx`, `shaders/*.wgsl` |
| `src/metal_src/` | Metal backend, Objective-C++ (`.mm`, **no ARC**), namespace `mtlctx`, `shaders/*.metal` |
| `src/gpu_src/` | one shim header per interface: `USEMETAL` / `USEWEBGPU` / CUDA. Plus `gpu_backend.h` (replay-harness only, no CUDA arm) |
| `src/tools/GPUReplay/` | `gpu_replay_main.cxx` (CUDA self-replay) and `webgpu_replay_main.cxx`, which is compiled **twice** — into `webgpu_replay` and `metal_replay` — so both backends are graded against the same records and tolerances by construction |

`DRBUDDI_Diffeo.h`'s `using CurrentImageType = CUDAIMAGE` is the insertion point; each backend
supplies `using CUDAIMAGE = GPUIMAGE;` so `main/` needs no rewrite.

**Per-backend wrappers stay duplicated.** Several audits proposed unifying them (8 of 11
`metal_src`/`webgpu_src` pairs are byte-identical over ~1064 lines). Rejected: `cuda_src/`
already carries its own copy, so a third backend behaving differently buys consistency in one
place at the cost of consistency everywhere. The shared seam is the *replay harness*, not the
wrappers. **Do not re-raise.**

### Build options

`USECUDA`, `USEWEBGPU`, `USEMETAL` — mutually exclusive, all default 0, each also defines
`USEGPU` **on the command line** (several files use `#ifdef USEGPU` without including
`defines.h`; an invisible macro would silently compile the GPU branch out and still "work").

Defs and include paths must be applied **per target**, never `add_definitions` — directory-wide
`-DUSEGPU` once leaked into the plain CPU targets. Likewise the CPU targets are excluded when
`USEWEBGPU` or `USEMETAL` is set, or they clobber the shared `bin/`.

**Shader embedding staleness trap:** `metric_common.wgsl` / `.metal` is a prelude *textually
prepended* to the metric shaders. It is excluded from the glob and **must be in every embed
rule's `DEPENDS`** — without that, editing it leaves every generated header stale and the build
reports success while running old shader code.

### macOS (Apple silicon) — build gotchas

| Item | Gotcha |
|---|---|
| **ITK 6.0b02** | **Must be built from source with `-DModule_ITKVtkGlue=OFF`.** Homebrew ITK is unusable: `ITKConfig.cmake:88` calls `itk_module_config(ITK ${ITK_MODULES_ENABLED})` **before** reading `ITK_FIND_COMPONENTS`, so `ITKVtkGlue`'s hardcoded `find_package(VTK NO_MODULE REQUIRED)` drags in VTK then Qt6 with **no consumer-side escape**. Do not re-litigate with `COMPONENTS`. |
| **Eigen** | 3.4 (Eigen 5 fails the `3.4` same-major check), and **ITK must be built against the same Eigen** or its bundled copy collides. |
| **Boost** | request only `iostreams filesystem`. Boost.System is header-only since 1.69 and ships no stub. |
| **C++ standard** | C++17 minimum (ITK 5.4+ hard-errors below it on Apple LLVM). Metal targets C++17, WebGPU C++20 (Dawn). |
| **`std::bind2nd`** | removed in C++17 → `-D_LIBCPP_ENABLE_CXX17_REMOVED_BINDERS -D_LIBCPP_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS`. Upstream call sites left untouched. |
| **`-march`** | `TORTOISE_ARCH` defaults to `none` on Apple (arm64 has no `x86-64-v3`). Host FP therefore differs from the Linux reference before any GPU kernel runs — though measured contribution is nil (§4). |
| **`libbetokan`** | Linux-x86-64 only, no source in tree. Nothing is linked on macOS; brain extraction shells out to FSL `bet2` (§5). |
| **GNU-only link flags** | `-static-libgcc`, `-Wl,--exclude-libs`, `-Wl,--start-group` and `-ldl` are all unusable on ld64 — hence the conditional `BETOKAN_LIB` / `STATIC_CXX_LIBS` / `EXCLUDE_ZLIB_FLAG` / `GRP_START` variables. |
| **Binary size** | `-ffunction-sections -fdata-sections -Wl,-dead_strip` on Apple: 25.7 → **17.4 MB**. ITK's IO factories are **not** stripped (Mach-O keeps `__mod_init_func` as roots) — verified by running `FlipImage3D` on a 46 MB NIfTI and `nm -a \| grep -c NiftiImageIOFactory` = 14. |

Build: `benchmark/scripts/mac_configure.sh <webgpu|metal>` then `cmake --build build_metal -j`.

---

## 2. Validation

The pipeline **cannot reproduce itself** end-to-end (§4), so correctness is established at the
*kernel* level with deterministic golden vectors. This is the load-bearing design decision.

| Layer | Proves | Determinism |
|---|---|---|
| capture (`TORTOISE_GPU_CAPTURE`) | records `{op, params, in, out}` at the 20 reachable host wrappers | — |
| `gpu_replay` | CUDA reproduces its own capture **bit-for-bit** | exact |
| `webgpu_replay` / `metal_replay` | WGSL / MSL vs the **CUDA outputs carried in each record**, and vs exact analytic references | tolerance-gated |
| `compare_outputs.py` | full-pipeline **smoke test only** | non-deterministic |

**Each record carries CUDA's recorded output**, so `metal_replay` is a *direct* Metal-vs-CUDA
comparison that needs no CUDA and no Linux machine. This is why macOS validation is
self-contained, and why cross-machine end-to-end comparisons add little.

Reachability was determined by **runtime capture, not grep** — static counting suggested 23
wrappers; an instrumented run captured **20**. The three absentees (`ComputeEntropy`,
`NegateField`, `ComputeImageGradientImg`) are dead; their apparent callers are the CPU/ITK
namesakes. Do not re-open this by grepping.

**Record counts, pinned:** 132 `fast` / 118 `medium` / 106 `slow` / 25 synthetic = **381**.
`medium` covers 18 ops (no structural → no CCJacS/CCSK), `slow` 16.

```bash
# capture (CUDA build) - N=6 is REQUIRED; the built-in default of 2 gives 44 records
TORTOISE_GPU_CAPTURE=<dir> TORTOISE_GPU_CAPTURE_N=6 bin/TORTOISEProcess_cuda ...
bin/gpu_replay    --all benchmark/fast/CAPTURE/golden_vectors     # MUST be bit-exact
bin/metal_replay  --all benchmark/<ds>/CAPTURE/golden_vectors     # or webgpu_replay
benchmark/scripts/revalidate.sh                                   # one-command gate suite
```

**Capture quotas are per SHAPE CLASS**, not per op. "First N calls per op" biased capture to the
first pipeline phase, leaving `GaussianSmoothImage`'s 3-component field path with zero coverage.
Keys are `GaussianSmoothImage.v1.*` / `.v3.*` (likewise `ResampleImage`). Do not revert.

**Synthetic records exist because CUDA is the LESS accurate implementation for sampled ops**
(§3). A linear ramp interpolates exactly under trilinear, so it gives a reference a sampling bug
cannot satisfy. They also cover border overhang, oblique directions, a single-slice volume and
the 3-component field path — none of which the captured vectors reach.

**NaN policy in the harness — divergence only.** Matching NaNs between backends count as
agreement; a non-finite value in **one** backend is a failure. This exists because
`std::max(x, NaN)` returns `x`, so NaN differences were invisible and PASSED. It is a fix to the
*comparison*, not a judgement that NaN is undesirable. `Compare()` also **throws** on an output
size mismatch rather than comparing the common prefix — a wrong-shaped result once reported PASS.

**On macOS `revalidate.sh` skips 5 CUDA-only gates** (record manifest, CUDA self-replay slow,
DIFFPREP pa registration, isolated DRBUDDI Step2, end-to-end comparison) and says
`PASSED, but N gate(s) SKIPPED - Not a full validation`. A macOS pass is not a full validation.

---

## 3. Measured numerical findings — DO NOT re-derive or undo

### Tolerances

**Elementwise 1e-5, and it is a floor, not a fudge.** `ResampleImage` (no hardware filtering, so
it isolates the buffer/index/arithmetic layer) reproduced CUDA to 1.45e-6. Rather than relax the
original 1e-6, CUDA was rebuilt with `-fmad=false` and replayed against its own FMA-built
vectors: **the same source, one contraction flag apart, disagrees with itself by 2.67e-6** —
more than the port does. nvcc contracts `a*b+c` by default and WGSL exposes no contraction
control, so cross-backend bit-exactness is unattainable in principle. 1e-5 is ≈4× the measured
compiler-induced spread; a real porting bug is orders of magnitude larger.

A **second, per-element gate** also applies — `|a-b| <= elem_tol * (|b| + rms(ref))`, with
`elem_tol` 1e-4 elementwise/exact, 1e-3 guarded, 2e-2 texture. The max-normalised gate alone
would let a port return zero for every small element of a large-max field and still pass. Be
accurate about its worth: median band shrink is **1.08×**, and 50 of 150 tensors are *looser*
per-element. The metric kernels' real evidence is expression-by-expression source review.

**Guarded ops use an outlier-fraction rule, not max-error.** `ComposeFields` exceeded tolerance
on 52 of 95823 elements; CUDA-vs-CUDA under the FMA toggle also exceeds it. A domain-guard branch
flips when a coordinate lands within 1 ulp of the boundary and the two branches return genuinely
different things — the error is *discontinuous*, so max-error is the wrong statistic. **Rule:
guarded ops pass if exceedances are ≤ 0.1 % of elements.** Guarded: `ComposeFields`,
`ResampleImage`, `QuadraticTransformImageC`, `InvertField` (which calls `ComposeFields` ~40×).

### CUDA's texture sampler is lossy — the ports are the accurate ones

| test | result |
|---|---|
| synthetic linear ramp, **CUDA** vs exact trilinear | **3.32e-5** — CUDA's sampler is lossy |
| synthetic linear ramp, port vs exact trilinear | passes at 1e-5 |

Emulating the documented `floor(w*256)/256` was **~1000× worse** than exact weights, so the
hardware's rounding model is not established and does not need to be. Texture-sampling ops get a
two-sided gate: **vs exact trilinear at 1e-5** (primary) and **vs CUDA at 5e-3** (bounds CUDA's
own loss, catches gross errors only).

`TEXFILTER_QUANTISE` is a diagnostic only. **CLOSED for Metal by measurement: not needed** — the
ports never touch a texture unit, so there is no hardware filter whose rounding could differ.
The conclusion is structural; **do not re-open it as an unknown for a fourth backend.**

**WebGPU/Metal cannot reproduce the CUDA sampler with a sampler at all** — neither has
`clamp-to-border`, which the CUDA textures use. Manual trilinear over a storage buffer is
therefore *mandatory*, `CreateTexture()` is a no-op, and no sampler is used anywhere.

**Four sampling kernels, three different out-of-domain rules.** Assuming one shared convention
would produce quietly wrong edges in three of four:

| kernel | rule |
|---|---|
| `ResampleImage` | whole output voxel zeroed if **any** axis leaves `[0,N-1]` |
| `QuadraticTransformImageC` | guarded **before** sampling; out-of-domain voxels left at the allocation zero |
| `WarpImage` | **unguarded**; per-neighbour border zero, so a one-voxel halo gets nonzero partials |
| `ComposeFields` | falls back to the **main field's** displacement — neither zeroed nor blended |

All sample at `tex3D(iw+0.5, ...)` with unnormalised coords so CUDA's internal `u = x-0.5`
recovers `iw`; **WGSL and MSL use `iw` directly — no half-texel juggling.**

### Other traps

**Row padding is not uniformly harmless.** CUDA's reductions size themselves as
`pitch/sizeof(float)*sy*sz`, which includes row padding that `Allocate()` never initialises. For
*summations* the contribution is negligible; for **`ScalarFindMax`/`ScalarFindMin`
(`PreprocessImage`, `InvertField`) it is not — max/min do not average**, so one garbage float
sets the result outright. Dormant on this allocator (all records replay bit-exactly), but note
`ScalarFindMax` seeds at `-1`, so CUDA's max is `max(true_max, 0, -1)`; on an all-negative image
CUDA would return 0 where the ports return the true maximum.

**Metric images are memset to BYTE 1, not 0** (`compute_metric.cu:836,1326,...`), the same
`int`-byte-value parameter as `FillBuffer` — every float starts at `2.3694e-38`. Contributes
~1e-32 against metric values of order 1. Recorded so nobody mistakes it for a defect.

Gaussian smoothing reads taps straight from the buffer rather than staging lines in workgroup
memory (arithmetically identical, and avoids tying workgroup size to an image dimension), and
boundary taps are **skipped, not renormalised** — edge voxels are convolved with a truncated
kernel and get systematically darker. That is the reference behaviour.

---

## 4. The GPU/CPU split is dynamic — this explains most "divergence"

**`GPU_CPU_ratio` no longer exists.** `DIFFPREP.cxx:597-615` discovers the split at runtime from
measured per-volume times: a CPU thread takes another volume only while
`remaining * t_gpu > t_cpu`. Volumes come from a shared counter; the estimates are `static` and
survive across calls.

**Consequence, and it is not obvious: a SLOWER GPU backend pushes MORE volumes onto the
non-reproducible ITK CPU path, so the end-to-end statistic degrades with backend SPEED, not only
with backend accuracy.** Measured on `fast` (138 up volumes):

| backend | `t_gpu` s/vol | split GPU/CPU |
|---|---:|---|
| Linux CUDA | 0.51 | 108 / 30 |
| Linux WebGPU | 1.26 | 78 / 60 |
| macOS Dawn | 5.20 | 41 / 97 |
| macOS Metal | 1.96 | 76 / 62 |

**Pearson r falls monotonically with split difference** — same machine, same backend, only the
routing differing:

| GPU-volume delta | r |
|---:|---:|
| 3 | 0.999675 |
| 0 | 0.999424 |
| 34 | 0.998892 |
| 47 | 0.998837 |
| 81 | **0.998644** |

So an end-to-end r of ~0.998 is **the split, not the backend**. Corroborated: Dawn-on-Metal —
different backend, same WGSL that scores ~0.9991 against CUDA on Linux — scores **0.998067**
against Linux CUDA from macOS, within 4e-5 of native Metal's 0.998037.

**The end-to-end comparison is therefore a SMOKE TEST, not a gate** (r ≥ 0.9986, spread ≤ 20 —
gross breakage only). The nominal `r ≥ 0.99999` in the original plan **cannot be met by CUDA
against itself** and must never be applied.

**Making it deterministic:** `-DDETERMINISTIC_GPU=1` (a *build* switch,
`DIFFPREP.cxx:617-625`) sets `n_omp_threads = NGPUs`, routing every volume to the GPU and
disabling ITK CPU registration. It **does change computed values** — its `OmpThreadBase.h` half
fixes ITK's work-unit count — so det-vs-det is valid, det-vs-nondet is not. Binaries get a
`_det` suffix so they cannot overwrite the reference ones.

**`DETERMINISTIC_GPU` alone is not sufficient**, and neither is `--DRBUDDI_step 2` alone:
- `DETERMINISTIC_GPU` fixes DIFFPREP routing, but DRBUDDI **Step1** (rigid/structural
  registration) runs on ITK CPU regardless.
- `--DRBUDDI_step 2` skips Step0/Step1, but the DIFFPREP that still runs overwrites Step2's
  inputs *inside the frozen fixture* — so the fixture is frozen on disk but not in effect.

**Together they work.** Verified bit-identical on both platforms: Linux CUDA 867 == 867
iterations with matching sha256; macOS Metal 911 == 911, all five Step2 artefacts byte-equal.
Iteration counts differ *across* platforms because the DIFFPREP that re-runs is backend-specific,
so a cross-backend Step2 comparison still needs the **same fixture on both machines**.

Also measured: `ap_proc.nii` (import + denoise + Gibbs — pure CPU, no GPU) is **r =
1.000000000000** across macOS and Linux. Host arithmetic contributes essentially nothing; the
divergence is entirely registration.

---

## 5. Upstream bugs — not caused by this work

- **`DRBUDDI.cxx` standalone** did `this->stream = &((*stream))` on an uninitialised member;
  fixed to `&std::cout`. Standalone DRBUDDI still needs a `<basename>.bmtxt` next to each NIfTI
  which the datasets lack — `CreateCorrectionImage` hands an unread matrix to `vnl_svd` and
  segfaults. Generate one from `.bval`/`.bvec` (6 columns: `bxx, 2bxy, 2bxz, byy, 2byz, bzz`).
- **`create_mask.cxx`'s `#ifdef __APPLE__` branch had never been compiled** — it referenced
  `list`, `vol_id`, `b0_mask_img` from no enclosing scope, had no `return`, and shelled out to a
  path not in the tree. Rewritten: resolves `$TORTOISE_BET2` then `bet2` on `$PATH`, keeps
  `FSLOUTPUTTYPE=NIFTI`, shell-quotes all three paths, and **checks the mask is non-empty** —
  bet2 can exit 0 and write a valid **all-zero** NIfTI, which would propagate silently through
  the whole pipeline. Deliberately *not* checked: arm64-ness (an x86-64 `bet2` runs fine under
  Rosetta) and wrapper-script resolution.
- **VXL's `operator vnl_vector<T>()` / `vnl_matrix<T>()` are `explicit`** in this VXL, so implicit
  conversions from the `_fixed` types do not compile. Fixed with `.as_vector()` /
  `.as_matrix()` in `DRBUDDIBase.cxx` and `itkANTSAffine3DTransform.hxx` — both return **by value
  via `std::copy`** (deep copies, not views), all sites read-only, so semantically identical.
  Note `DRBUDDIBase.cxx` is also compiled by the Linux CUDA build.
- **POSIX `finite()` removed from modern libc** — `estimate_mapmri_pa.cxx` and `mpfit.h`.
- `DRTAMAS_cuda` needs `gpu_capture.cxx` in its sources because it compiles shared TUs that now
  reference `gpucap::Rec`. The clean fix is a compile-time switch; deliberately not done.

---

## 6. Metal-specific gotchas

**Metal defaults to fast math, and `mathMode` alone is NOT enough.** `MTLCompileOptions` splits
the deprecated `fastMathEnabled` into two properties:

| property | governs | default |
|---|---|---|
| `mathMode` | reassociation, denormal flush, IEEE arithmetic | `Relaxed` |
| `mathFloatingPointFunctions` | which math **library** `sqrt`/`log` resolve to | **`Fast`** |

Setting only `mathMode = Safe` left every `sqrt`/`log` on the low-precision library on macOS 15+,
while the macOS-14 fallback (`fastMathEnabled = NO`) set both — **the same source produced
different numbers depending on host OS version**. Measured: 1154 of 4096 `sqrt` samples
misrounded; setting `Precise` took `fast` from 91 to **102 of 132 records bit-exact with CUDA**,
including `ScaleUpdateField`, whose magnitude normalises the update field on *every* DRBUDDI
iteration. **Set both.**

Dawn has the same trap from the other side: its Metal backend emits
`#pragma METAL fp math_mode(relaxed)` unless `strictMath` is requested via
`ShaderModuleCompilationOptions`. Measured cost of leaving it: **124/132**, with `ComposeFields`
and `InvertField` over the 1e-5 gate on 0.26–6.25 % of elements.

**Other Metal rules:**
- **Threadgroup size == the WGSL `@workgroup_size`.** `Pipeline()` **aborts** if it exceeds
  `maxTotalThreadsPerThreadgroup` rather than shrinking — a shrink would change reduction
  summation order. Only `reductions/reduce` requests the full 1024 (no headroom); everything else
  is 128 or 64. Device limits are identical across Apple7 (M1) → Apple9, and threadgroup memory
  use is 4 KiB against 32 KiB. `TORTOISE_METAL_DEBUG=1` prints per-pipeline headroom.
- **MSL `select(x, y, cond)` returns `y` when true — same as WGSL.** C's ternary reads the other
  way. Do not "simplify" a `select()`.
- WGSL `override`s become **`uint` function constants**; every one they replace is integral, so
  no name→type table is needed. None carries a default, so a misspelled name fails loudly.
- **MSL contracts `a*b+c` into FMA and WGSL does not**, so Metal lands *closer* to CUDA:
  97 of 132 `fast` records bit-exact vs WebGPU's 84.

**Host/GPU ordering on `MTLResourceStorageModeShared`.** A host `memcpy` is **not** ordered
against queued GPU work. `Allocate()` queues a zero-fill and `ToDevice()` memcpys into the same
buffer — measured **26011 of 95823 elements zeroed, 72/72 records failing at `rel = 1`** before
`mtlctx::Sync()` was introduced. **Do not remove that `Sync()` while optimising; narrow it.**
`ZeroFresh`/`CreateStorageFrom`/`CreateUniform` skip it legitimately because a just-created buffer
cannot be referenced by queued work (command buffers use *retained references*, so the allocator
cannot recycle pages an in-flight buffer is reading).

**The backend is compiled WITHOUT `-fobjc-arc`** — manual retain/release, deliberate, because it
keeps `metal_context.h` pure C++ via `shared_ptr<void>` over ObjC objects. It cost four bugs:

| bug | effect |
|---|---|
| `Wrap()` used `__bridge_retained` on an already-`+1` `new…` object | **every `MTLBuffer` leaked** |
| no `@autoreleasepool` in `Zero`/`CopyBuffer`/`Sync`/`DispatchAt`/`Init` — and **a command buffer retains every resource it references** | `phys_footprint` climbed **20.4 → 27.6 GiB** while RSS showed **zero** growth and `leaks(1)` reported **nothing** (pooled ≠ leaked) |
| `MTLFunction`, `MTLFunctionConstantValues`, `MTLCompileOptions`, `MTLCopyAllDevices` unreleased | bounded |
| `[cb commit]` outside the mutex publishing `g_last_cb` | `Sync()` could wait on an *earlier* buffer and memcpy shared storage another kernel was still writing |

Two further ordering rules now in force: **failure observation must be synchronous** — waiting on
the last command buffer proves earlier ones *completed* but not that their completion handlers
have *run*, so `Sync()` waits for a completion counter to reach the commit counter before reading
the error flag; and abort paths use `_Exit`, because `exit()` under a held mutex can deadlock on a
static destructor re-entering the backend.

The in-flight command-buffer cap (default 16) was deleted once as "measures as a no-op on memory"
— that measurement was taken while the buffer leak dominated, so it could not have shown an
effect. **Restored; do not delete again.** Autorelease pools release *our* reference, not the
queue's, so they never bounded retention.

---

## 7. Measuring on macOS — two things that void a measurement

**1. Idle sleep silently inflates wall clock.** macOS entered Maintenance Sleep mid-run while
`etime` kept counting: **one run recorded 44 min elapsed for ~20 min of compute, which read as a
2.4× regression that did not exist.** Every timed run is wrapped in `caffeinate -i`. **Any macOS
timing taken without it is void** — check `pmset -g log | grep "Entering Sleep"` before trusting
a number. (BSD `time` also has no `-v`/`-o`, so the harness synthesises the GNU-format line.)

**2. RSS under-counts GPU memory; `phys_footprint` is the honest metric.** On unified memory
there is no `nvidia-smi` equivalent, and RSS excludes IOKit/IOAccelerator allocations — it can
under-count a backend holding buffers device-private and over-count one holding them host-visible,
so **two backends can differ purely from accounting**. The 7 GiB climb above is the proof: RSS
reported zero. `ProfileScope` emits, fields **appended never reordered**:

```
[PROFILE] <name> <seconds> peak_MiB <p> grew_MiB <g> foot_MiB <f>
```

**`grew_MiB` is NOT a per-stage allocation measurement.** `ru_maxrss` is a process high-water: a
positive value identifies the scope that crossed a **new global peak**; **zero does not mean the
scope allocated nothing**. Do not use zeros in late stages as evidence that those stages are
cheap. `benchmark/scripts/mac_stage_memory.py <runA-dir> <runB-dir>` diffs two runs stage by stage
(`--tol-mib`, `--tol-frac`, non-zero exit on divergence).

**Never accept a regression from n=1.** `PERF_NOTES.md` §12: DRBUDDI per-iteration time swings
**34 %** between identical runs, and three regressions were confirmed then retracted — one
claiming a 40 s cost for an operation measured at 0.44 s. `DIFFPREP.Register` (±1 s over 21 runs)
is the stable signal; DRBUDDI wall time is not.

---

## 8. Exhausted paths — do not repeat these

| Attempt | Outcome |
|---|---|
| Register blocking on `computeFiniteDiffStructs` | **59 % slower** — the kernel wants occupancy, not ILP |
| Emulating CUDA's `floor(w*256)/256` filter weights | ~1000× *further* from CUDA than exact weights |
| `TEXFILTER_QUANTISE` for the Metal port | not needed; structural, not hardware-specific |
| Rebuilding ITK with a newer x86 ISA | no meaningful CPU win |
| `find_package(ITK COMPONENTS …)` to dodge `ITKVtkGlue` | impossible — `ITKConfig.cmake` loads every module first |
| Deleting the Metal in-flight cap | the evidence for deleting it was invalid (leak-dominated) |
| Unifying the per-backend op wrappers | rejected — `cuda_src/` sets the pattern (§1) |
| `ncu` profiling on the Linux host | blocked: `ERR_NVGPUCTRPERM` needs root, no sudo available |

**Running sums for the correlation-window kernels** remain the largest theoretical win —
`computeFiniteDiffStructs` evaluates a 19×19×9 = 3249-voxel window per output voxel and adjacent
outputs share ~95 % of it, so a summed-area formulation is O(1) per voxel instead of O(w³). **It
is not bit-exact**: incremental sums accumulate in a different order, so `gpu_replay` would stop
being byte-identical — the anchor everything rests on. Requires explicit sign-off, a full
recapture of all 381 records, and a fresh derivation of every tolerance in §3. **Do not start it
as ordinary performance work.**

**Known remaining inefficiency, deliberately deferred:** each `CCJacS` call allocates the metric,
two output fields and twelve full scalar scratch images across its two passes, and every dispatch
allocates a parameter buffer. Profile before changing it, and use a bounded shape-keyed scratch
lease released only after the dependent submission completes — not generic global pooling.

**WebGPU still performs five scalar readbacks per MI evaluation where Metal drains once.** This is
a deliberate non-change: WebGPU's role is a *correctness* control, not a speed yardstick, and the
drain count does not affect the replay comparison. **Consequence: Metal-vs-WebGPU timing is not
comparable** and no such ratio should be quoted. Judge Metal against its own history and against
hardware-normalised CUDA (Dawn runs the same WGSL on both machines: macOS 21:59 vs Linux 10:37 =
**2.07×**, which is the silicon, not the port).
