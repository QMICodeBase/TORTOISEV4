#!/usr/bin/env bash
# Minimal demonstration: TORTOISE's DIFFPREP registration is not reproducible
# run-to-run, and the cause is the ITK CPU registration path, not the GPU.
#
# This is a property of the UPSTREAM CUDA/CPU code. It is not caused by, and does
# not depend on, the WebGPU port. Reproduce with stock TORTOISEProcess_cuda.
#
#   benchmark/scripts/demo_diffprep_nondeterminism.sh            # explain + check existing runs
#   benchmark/scripts/demo_diffprep_nondeterminism.sh --run      # do two fresh CUDA runs first
#
# ---------------------------------------------------------------------------
# THE MECHANISM (src/main/DIFFPREP.cxx:578-663)
#
# Each DWI volume is registered to the b=0 reference. Volumes are split between
# the CUDA path and the ITK CPU path by PURE INDEX ARITHMETIC:
#
#     Nt            = omp_get_max_threads()
#     GPU_CPU_ratio = 15                       // assumed GPU:CPU per-volume speedup
#     max_t_per_pass = NGPUs*GPU_CPU_ratio + Nt - NGPUs
#     npass          = ceil(Nvols / max_t_per_pass)
#     per pass: the first GPU_CPU_ratio*NGPUs volumes -> GPU, the remainder -> CPU
#
# That assignment is DETERMINISTIC - it depends only on Nvols, NGPUs and Nt, and
# the loop uses `schedule(static,1)`, so a given volume always takes the same path
# on the same machine. The nondeterminism is therefore NOT "which volume went
# where"; it is inside the ITK CPU registration itself.
#
# The prediction that follows is sharp and testable: a dataset small enough that
# EVERY volume lands on the GPU must be bit-reproducible, while a larger one that
# spills onto the CPU path must not be. Both appear in a single `fast` run:
#
#   pa (down) 10 volumes  -> 10 <= GPU_CPU_ratio*NGPUs = 15  -> ALL GPU  -> reproducible
#   ap (up)  138 volumes  -> 45 GPU, 93 ITK CPU              -> MIXED    -> NOT reproducible
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD

if [ "${1:-}" = "--run" ]; then
    echo "Two fresh CUDA runs (~12 min each). Sequential - concurrent runs share the GPU."
    . benchmark/scripts/gpu_env.sh
    for tag in DEMO_A DEMO_B; do
        echo "=== $tag ==="
        STAGE_CACHE=0 benchmark/scripts/run_cuda_reference.sh fast "$tag" >"/tmp/$tag.log" 2>&1 \
            || { echo "run failed, see /tmp/$tag.log"; exit 1; }
    done
    A=benchmark/fast/DEMO_A; B=benchmark/fast/DEMO_B
else
    # Any two existing CUDA runs of the same dataset serve equally well.
    A=""; B=""
    for d in benchmark/fast/CUDA benchmark/fast/CUDA_C benchmark/fast/CUDA_D benchmark/fast/CUDA_E; do
        [ -d "$d" ] || continue
        [ -z "$A" ] && { A=$d; continue; }
        [ -z "$B" ] && { B=$d; break; }
    done
    [ -n "$B" ] || { echo "Need two CUDA runs. Re-run with --run."; exit 1; }
fi

echo
echo "Comparing two runs of the SAME CUDA binary on the SAME input:"
echo "  A = $A"
echo "  B = $B"
echo

# The two transformation files are the direct output of DIFFPREP registration:
# 24 quadratic parameters per volume, written as text. Comparing them isolates
# registration from every later stage.
status=0
for tag in pa ap; do
    a=$(find "$A" -name "${tag}_proc_moteddy_transformations.txt" 2>/dev/null | head -1)
    b=$(find "$B" -name "${tag}_proc_moteddy_transformations.txt" 2>/dev/null | head -1)
    [ -n "$a" ] && [ -n "$b" ] || { echo "  $tag: transformations not found - skipping"; continue; }

    if cmp -s "$a" "$b"; then
        verdict="IDENTICAL"
    else
        verdict="DIFFERS"
        status=1
    fi
    n=$(python3 -c "
import re,sys
f=lambda p:[float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?',open(p).read())]
A,B=f('$a'),f('$b'); d=[abs(x-y) for x,y in zip(A,B)]; nz=[v for v in d if v>0]
print(f'{len(nz)}/{len(d)} params differ, median {sorted(nz)[len(nz)//2]:.2e}, max {max(d):.4f}' if nz else f'0/{len(d)} params differ')
")
    printf "  %-3s (%s)  %-10s  %s\n" "$tag" \
        "$([ "$tag" = pa ] && echo 'all-GPU ' || echo 'GPU+ITK')" "$verdict" "$n"
done

cat <<'EOF'

INTERPRETATION
  pa uses ONLY the CUDA path and is bit-reproducible: the CUDA kernels are
  deterministic. (Independently confirmed - every captured kernel golden vector
  replays bit-exactly under `gpu_replay --all`.)

  ap spills 93 of 138 volumes onto the ITK CPU path and is NOT reproducible.

  Same binary, same input, same machine. The only difference between the two is
  which registration implementation ran.

CONSEQUENCE
  Any end-to-end comparison of TORTOISE against itself inherits this. Two runs of
  the same CUDA build differ in ~35% of final DWI voxels, with Pearson r ~0.9995
  (see benchmark/README.md). An acceptance gate for a port therefore cannot be
  "bit-identical to CUDA", and cannot even be "inside the CUDA-vs-CUDA spread"
  without first establishing that that spread bounds arithmetic differences - it
  does not; it bounds thread-scheduling nondeterminism in one CPU code path.

MAKING IT DETERMINISTIC
  Forcing every volume onto the GPU path removes the nondeterministic component.
  With no code change at all:

      OMP_NUM_THREADS=1 bin/TORTOISEProcess_cuda ...

  because Nt=1 makes max_t_per_pass = GPU_CPU_ratio*NGPUs, so no pass ever exceeds
  the GPU quota. The cost is severe and not limited to registration: it
  single-threads the WHOLE pipeline, including denoising and tensor fitting. On
  this host a `fast` CUDA run uses ~17x parallelism (11,967 s CPU / 698 s wall).

  The targeted alternative is a one-constant change at DIFFPREP.cxx:589 - raise
  GPU_CPU_ratio above Nvols (or gate it behind an env var) so the CPU branch is
  never taken, leaving every other stage multi-threaded. Registration then runs
  ~3x longer for this dataset (138 volumes serially on one GPU, versus 3 passes of
  15-GPU-plus-31-CPU in parallel), while the rest of the pipeline is unaffected.

  Neither is proposed as a change to this project: DIFFPREP.cxx is upstream code
  and outside the WebGPU port's remit. They are listed so the cost of determinism
  is on the record.
EOF
exit $status
