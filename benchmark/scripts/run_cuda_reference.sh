#!/bin/bash
# Generate CUDA reference outputs + timing/memory profile for one validation dataset.
#
# Usage: run_cuda_reference.sh <fast|medium|slow> [backend]
#   backend: CUDA (default) or WebGPU -> selects the executable and the output folder.
#
# Outputs land in benchmark/<dataset>/<backend>/ ; benchmark/<dataset>/In/ is never modified.
set -euo pipefail

DS=${1:?usage: run_cuda_reference.sh <fast|medium|slow> [CUDA|WebGPU]}
BACKEND=${2:-CUDA}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BENCH=$ROOT/benchmark/$DS
IN=$BENCH/In
OUT=$BENCH/$BACKEND

case $BACKEND in
    # CUDA_* variants run the same CUDA binary into a separate output folder, so
    # repeat runs can be compared against each other to measure the run-to-run
    # floor (CLAUDE.md 5.4). The floor was originally estimated from n=2, which is not
    # enough to judge a near-miss against.
    # -fmad=false control build (CLAUDE.md 5.1). Used to measure how far a
    # PURE compiler-contraction change moves the end-to-end result, which bounds what
    # any non-CUDA backend could achieve.
    # Deterministic validation build: every volume routed to the GPU path.
    DET|DET_*)     EXE=$ROOT/bin/TORTOISEProcess_cuda_det ;;
    NOFMA)         EXE=$ROOT/bin/TORTOISEProcess_cuda_nofma ;;
    # CAPTURE* runs the CUDA binary with the golden-vector hooks enabled.
    CUDA|CUDA_*|CAPTURE*)   EXE=$ROOT/bin/TORTOISEProcess_cuda ;;
    # WebGPU_* variants (e.g. WebGPU_2STR) run the same binary into a separate
    # folder, mirroring the CUDA_* convention.
    # *_DET arms MUST precede their generic counterpart: WebGPU_* would otherwise
    # swallow WebGPU_DET and run the NON-deterministic binary under a _DET tag.
    CUDA_DET*)   EXE=$ROOT/bin/TORTOISEProcess_cuda_det ;;
    WebGPU_DET*) EXE=$ROOT/bin/TORTOISEProcess_webgpu_det ;;
    WebGPU|WebGPU_*) EXE=$ROOT/bin/TORTOISEProcess_webgpu ;;
    # Metal_* variants mirror the CUDA_*/WebGPU_* convention.
    # *_DET variants use the DETERMINISTIC_GPU build, which the build system gives a
    # _det suffix so it can never overwrite the ordinary binary.
    Metal_DET*) EXE=$ROOT/bin/TORTOISEProcess_metal_det ;;
    Metal|Metal_*) EXE=$ROOT/bin/TORTOISEProcess_metal ;;
    CPU)    EXE=$ROOT/bin/TORTOISEProcess ;;
    *) echo "unknown backend $BACKEND" >&2; exit 1 ;;
esac

# Pin to the discrete NVIDIA GPU; never the AMD integrated display adapter.
# There is no such card on macOS and gpu_env.sh's Vulkan ICD pins would only
# mislead, so it is skipped there - the Metal backend selects its device in-process.
if [ "$(uname -s)" != "Darwin" ]; then
    source $ROOT/benchmark/scripts/gpu_env.sh
fi

mkdir -p "$OUT"

# A run overwrites its output directory, so without this only the MOST RECENT run's
# provenance survives and a sequence cannot be reconstructed later. That gap made a
# claim about four consecutive runs unverifiable after the fact. Keep a small,
# append-only trail: provenance + time.txt, stamped with when they were superseded.
if [ -f "$OUT/provenance.json" ] || [ -f "$OUT/time.txt" ]; then
    ARCH="$OUT/history/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$ARCH"
    for f in provenance.json time.txt run.log gpu_samples.csv; do
        [ -f "$OUT/$f" ] && cp -a "$OUT/$f" "$ARCH/" 2>/dev/null || true
    done
    echo "archived previous run metadata -> $ARCH"
fi

# time.txt/time.raw/provenance.json MUST go too: they were archived above, and
# leaving them means a run that dies before writing its own can be finalised into a
# provenance.json carrying the PREVIOUS run's wall clock and RSS with the CURRENT
# binary's hash. *.nii.gz as well - `slow` inputs are gzipped and cp -n would
# silently reuse a truncated copy.
rm -rf "$OUT"/*_temp_proc "$OUT"/*.nii "$OUT"/*.nii.gz "$OUT"/*.log \
       "$OUT"/time.txt "$OUT"/time.raw "$OUT"/provenance.json 2>/dev/null || true

# TORTOISE writes its *_temp_proc folder AND its final output next to the input
# files, and it resolves symlinks when choosing the final output path -- a symlink
# here writes the results back into In/. Copy instead, so In/ is never touched.
for f in "$IN"/*; do
    b=$(basename "$f")
    [[ $b == ._* ]] && continue
    cp -n "$f" "$OUT/$b"
done
chmod -R u+w "$OUT"

# --- dataset input preparation ------------------------------------------
# Applied to the copies in $OUT only; In/ stays exactly as delivered.
# `slow` shipped without JSON sidecars, and its gradient files are named for
# the session rather than the NIfTI, so TORTOISE cannot pair them.
# Phase encoding per dataset owner: dir-AP -> "j", dir-PA -> "j-".
# (Axis is what matters; DRBUDDI solves for the midpoint, so the sign
#  convention between the two is symmetric.)
if [ "$DS" = "slow" ]; then
    for pe in AP:j PA:j-; do
        d=${pe%%:*}; dir=${pe##*:}
        mag=$(ls "$OUT"/*dir-${d}*part-mag_dwi.nii.gz 2>/dev/null | head -1 || true)
        [ -n "$mag" ] || { echo "slow prep: no part-mag NIfTI for dir-$d" >&2; exit 1; }
        base=${mag%.nii.gz}
        for ext in bval bvec; do
            src=$(ls "$OUT"/*dir-${d}_dwi.$ext 2>/dev/null | head -1 || true)
            [ -n "$src" ] || { echo "slow prep: no .$ext for dir-$d" >&2; exit 1; }
            [ "$src" = "$base.$ext" ] || cp "$src" "$base.$ext"
        done
        printf '{\n  "PhaseEncodingDirection": "%s"\n}\n' "$dir" > "$base.json"
    done
    # part-phase volumes are not TORTOISE inputs
    rm -f "$OUT"/*part-phase*.nii.gz
    echo "slow prep: gradient files paired, JSON sidecars written (AP=j, PA=j-), phase volumes dropped"
fi

# Dataset-specific input names.
cd "$OUT"
shopt -s nullglob
pick() {   # pick <glob>... -> first match that is not an AppleDouble file
    local f
    for f in "$@"; do
        [[ $(basename "$f") == ._* ]] && continue
        echo "$f"; return 0
    done
    return 0
}
UP=$(pick *dir-AP*.nii *dir-AP*.nii.gz *ap.nii *ap.nii.gz)
DOWN=$(pick *dir-PA*.nii *dir-PA*.nii.gz *pa.nii *pa.nii.gz)
STRUCT=$(pick *T2w*.nii *T2w*.nii.gz *T1w*.nii *T1w*.nii.gz)
[ -z "${UP:-}" ] && { echo "no up-phase DWI found in $IN" >&2; exit 1; }

ARGS="--up_data $UP"
[ -n "${DOWN:-}" ] && ARGS="$ARGS --down_data $DOWN"
# `--structural` is repeatable, and DRBUDDI_Diffeo::SetDefaultStages does
# `for(int s=0;s<Nstr;s++)` for CCSK (and likewise CCJacS) - so each additional
# structural adds a metric, raising both coverage of those two metrics and peak GPU
# (~1.9 GiB per metric per megavoxel, see benchmark/README.md).
#
# OPT-IN ONLY. Passing every structural by default would silently change what a
# dataset means and invalidate every existing baseline and golden vector captured
# with one structural. Set STRUCTURALS=all to opt in.
if [ "${STRUCTURALS:-one}" = "all" ]; then
    nstr=0
    for sf in "$OUT"/*T2w*.nii "$OUT"/*T2w*.nii.gz "$OUT"/*T1w*.nii "$OUT"/*T1w*.nii.gz; do
        [ -f "$sf" ] || continue
        ARGS="$ARGS --structural $sf"; nstr=$((nstr+1))
    done
    echo "STRUCTURALS=all: passing $nstr structural image(s)"
    [ "$nstr" -eq 0 ] && { echo "  none found - aborting rather than silently running with none" >&2; exit 1; }
else
    [ -n "${STRUCT:-}" ] && ARGS="$ARGS --structural $STRUCT"
fi
# Extra flags for experiments, e.g. EXTRA_ARGS="--disable_itk_threads 1" to make
# ITK's threaded metric reductions deterministic.
[ -n "${EXTRA_ARGS:-}" ] && ARGS="$ARGS $EXTRA_ARGS"
# Pin the final output inside $OUT as well.
ARGS="$ARGS --output $OUT/${UP%%.nii*}_TORTOISE_final.nii"

echo "=== $DS / $BACKEND ==="
echo "cmd: $EXE $ARGS"

# Sample GPU memory (this process only) and host RSS once a second.
GPUSAMPLE=$OUT/gpu_samples.csv
# Truncate: the cleanup above does not remove this file, and the sampler appends,
# so re-running into an existing folder would blend two runs' samples together.
: > "$GPUSAMPLE"
if [ "$(uname -s)" = "Darwin" ]; then
    # No nvidia-smi, and no separate VRAM to query: Apple silicon is unified memory,
    # so the process footprint IS the GPU-inclusive figure. Same CSV shape
    # (epoch,pid,MiB) so finalize_run.sh's pid filtering works unchanged.
    # Match the ACTUAL executable, not a pattern: `pgrep -f` also matched the
    # /usr/bin/time and caffeinate wrappers (4 distinct pids appeared in one run) and
    # finalize_run.sh takes max() over the column, so a wrapper could become the
    # reported peak. -f "^$EXE$" pins it to the binary this run launched, and it is
    # derived from $EXE so a WebGPU or CPU run on macOS samples correctly too - the
    # previous hardcoded Metal pattern recorded NOTHING for a WebGPU run and silently
    # reported peak_gpu_mib 0.
    ( while true; do
        for pid in $(pgrep -x "$(basename "$EXE")" 2>/dev/null || true); do
            # || true: without it a pid vanishing between pgrep and ps trips
            # `set -euo pipefail` and kills the sampler for the rest of the run.
            rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)
            [ -n "$rss" ] && echo "$(date +%s),$pid,$((rss / 1024))" >> "$GPUSAMPLE"
        done
        sleep 1
      done ) &
else
( while true; do
    # Records every compute process on the card, one row per process. Consumers
    # MUST filter by pid - taking a bare max over the memory column picks up any
    # concurrent job. finalize_run.sh does this filtering.
    { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | \
      awk -F', ' -v t="$(date +%s)" '{print t","$1","$2}' >> "$GPUSAMPLE"; } || true
    sleep 1
  done ) &
fi
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null || true" EXIT

# Golden-vector capture. The backend case above documents CAPTURE* as "runs the CUDA
# binary with the golden-vector hooks enabled", but nothing ever set the environment
# variable that enables them - so `run_cuda_reference.sh <ds> CAPTURE` produced a
# normal run in a CAPTURE/ folder with ZERO records, and looked like it had worked.
# N defaults to 6: that is what the committed fast/medium suites were captured with
# (22 shape classes x 6 = 132 on fast). The built-in default is 2, which would give a
# smaller suite that no longer matches the pinned record counts.
if [[ $BACKEND == CAPTURE* ]]; then
    export TORTOISE_GPU_CAPTURE="$OUT/golden_vectors"
    export TORTOISE_GPU_CAPTURE_N="${TORTOISE_GPU_CAPTURE_N:-6}"
    mkdir -p "$TORTOISE_GPU_CAPTURE"
    echo "capture ENABLED -> $TORTOISE_GPU_CAPTURE (N=$TORTOISE_GPU_CAPTURE_N)"
else
    # Never inherit capture from the caller's shell: it would silently rewrite an
    # existing golden-vector tree during an ordinary reference run.
    unset TORTOISE_GPU_CAPTURE TORTOISE_GPU_CAPTURE_N
fi

if [ "$(uname -s)" = "Darwin" ]; then
    # caffeinate -i blocks IDLE sleep for the child. Without it macOS entered
    # Maintenance Sleep mid-run and etime/TOTAL kept counting: one run recorded 44 min
    # elapsed for ~20 min of compute, which read as a 2.4x regression that did not
    # exist. Any timing measured without this is void.
    # BSD time has no -v/-o. Emit a "Maximum resident set size (kbytes): N" line so
    # finalize_run.sh parses macOS and Linux runs with the same code.
    { /usr/bin/time -l caffeinate -i $EXE $ARGS 2>"$OUT/time.raw"; } | tee "$OUT/run.log"
    awk '/maximum resident set size/ {printf "\tMaximum resident set size (kbytes): %d\n", $1/1024}
         /real/ {print "\tElapsed (wall clock) time: " $1 " s"}' \
        "$OUT/time.raw" > "$OUT/time.txt"
    cat "$OUT/time.raw" >> "$OUT/time.txt"
else
/usr/bin/time -v -o "$OUT/time.txt" $EXE $ARGS 2>&1 | tee "$OUT/run.log"
fi

kill $SAMPLER 2>/dev/null || true

# M0 evidence: provenance record + immutable DIFFPREP stage cache.
"$ROOT/benchmark/scripts/finalize_run.sh" "$DS" "$BACKEND"
