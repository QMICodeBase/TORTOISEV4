#!/usr/bin/env bash
# Isolated DRBUDDI diffeomorphic-stage comparison.
#
# WHY: the full pipeline is not reproducible, so an end-to-end backend comparison
# is statistical rather than exact. Two sources were measured:
#   1. DIFFPREP registration - the ITK CPU path (see demo_diffprep_nondeterminism.sh)
#   2. DRBUDDI Step1 rigid/structural registration - also ITK CPU
# Both are CPU code that the WebGPU port does not touch, yet both make the
# reference disagree with itself.
#
# HOW: `--DRBUDDI_step 2` skips Step0 (b0/FA creation) and Step1 (rigid
# registration) entirely and enters at Step2_DiffeoRegistration - which is the
# GPU-bearing stage the port actually reproduces. Its inputs are read from
# <out>/*_temp_proc/ on disk, so a frozen snapshot makes them byte-identical
# across runs and across backends.
#
# The result: any difference in Step2's output is attributable to the GPU backend,
# not to CPU thread scheduling. This is the exact comparison the golden vectors do
# per-kernel, but at whole-stage level.
#
#   drbuddi_isolated.sh snapshot <src-run>     # freeze a run's temp_proc as the fixture
#   drbuddi_isolated.sh run <tag> <backend>    # run Step2 from the fixture
#   drbuddi_isolated.sh compare <tagA> <tagB>  # diff two Step2 outputs
#
# Note: TORTOISEProcess has no flag to skip DIFFPREP, so each run still pays that
# cost. It does NOT affect the comparison - Step2 reads none of DIFFPREP's output,
# only Step0/Step1's, which come from the frozen fixture.
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
DS=fast
FIXTURE=$ROOT/benchmark/$DS/DRB_FIXTURE

case "${1:-}" in
snapshot)
    SRC=${2:?usage: drbuddi_isolated.sh snapshot <run-dir-name, e.g. CUDA>}
    S=$ROOT/benchmark/$DS/$SRC
    T=$(find "$S" -maxdepth 1 -name "*_temp_proc" -type d | head -1)
    [ -n "$T" ] || { echo "no *_temp_proc under $S"; exit 1; }
    # Step2 needs these to exist; refuse to freeze an incomplete run.
    for f in blip_up_b0.nii blip_down_b0.nii structural_used.nii; do
        [ -f "$T/$f" ] || { echo "fixture source is missing $f - was the run complete?"; exit 1; }
    done
    rm -rf "$FIXTURE"; mkdir -p "$FIXTURE"
    # A previous snapshot chmod -R a-w the fixture, so a re-snapshot hits
    # permission-denied on every file. cp's status was never checked, so the script
    # printed "fixture frozen" and carried on with a STALE fixture - 109 silent
    # failures observed 2026-08-21. Make it writable first, and fail loudly.
    [ -d "$FIXTURE" ] && chmod -R u+w "$FIXTURE"
    cp -a "$T" "$FIXTURE/" || { echo "snapshot: copy FAILED - fixture not refreshed" >&2; exit 1; }
    for f in "$S"/*.nii "$S"/*.bval "$S"/*.bvec "$S"/*.json; do
        [ -f "$f" ] && { cp -a "$f" "$FIXTURE/" || { echo "snapshot: copy FAILED on $f" >&2; exit 1; }; }
    done
    chmod -R a-w "$FIXTURE"          # immutable, like the M0 stage cache
    echo "fixture frozen at $FIXTURE ($(du -sh "$FIXTURE" | cut -f1), read-only)"
    ;;

run)
    TAG=${2:?usage: drbuddi_isolated.sh run <tag> <CUDA|WebGPU|Metal|*_DET|NOFMA>}
    BE=${3:?usage: drbuddi_isolated.sh run <tag> <CUDA|WebGPU|Metal|*_DET|NOFMA>}
    [ -d "$FIXTURE" ] || { echo "no fixture - run 'snapshot' first"; exit 1; }
    case $BE in
        CUDA)   EXE=$ROOT/bin/TORTOISEProcess_cuda ;;
        WebGPU) EXE=$ROOT/bin/TORTOISEProcess_webgpu ;;
        Metal)  EXE=$ROOT/bin/TORTOISEProcess_metal ;;
        # DETERMINISTIC_GPU builds. --DRBUDDI_step 2 alone leaves the DIFFPREP that still
        # runs routing volumes by measured speed, which rewrites fixture inputs; this
        # closes that second hole. Bare DET is CUDA, matching run_cuda_reference.sh.
        DET)        EXE=$ROOT/bin/TORTOISEProcess_cuda_det ;;
        WebGPU_DET) EXE=$ROOT/bin/TORTOISEProcess_webgpu_det ;;
        Metal_DET)  EXE=$ROOT/bin/TORTOISEProcess_metal_det ;;
        # Same CUDA source, -fmad=false only. Bounds what a legitimate arithmetic
        # variation of the reference does to this stage, measured the same way.
        NOFMA)  EXE=$ROOT/bin/TORTOISEProcess_cuda_nofma ;;
        *) echo "backend must be CUDA, WebGPU, Metal, DET, WebGPU_DET, Metal_DET or NOFMA"; exit 1 ;;
    esac
    # gpu_env.sh pins an NVIDIA card that does not exist on macOS.
    if [ "$(uname -s)" = "Darwin" ]; then
        export TORTOISE_WEBGPU_BACKEND=${TORTOISE_WEBGPU_BACKEND:-metal}
        export TORTOISE_WEBGPU_ADAPTER_TYPE=${TORTOISE_WEBGPU_ADAPTER_TYPE:-integrated}
        export TORTOISE_WEBGPU_VENDOR_ID=${TORTOISE_WEBGPU_VENDOR_ID:-0x106B}
        export TORTOISE_BET2=${TORTOISE_BET2:-$(command -v bet2 || true)}
    else
        . "$ROOT/benchmark/scripts/gpu_env.sh"
    fi
    OUT=$ROOT/benchmark/$DS/$TAG
    rm -rf "$OUT"; mkdir -p "$OUT"
    cp -a "$FIXTURE"/. "$OUT"/
    chmod -R u+w "$OUT"

    # CRITICAL: the fixture includes Step2/Step3 outputs, which are exactly the
    # files the gate compares. Left in place, a run that dies before writing them
    # leaves CUDA-derived artefacts behind and the gate scores them as a PASS.
    # Delete them so the gate can only ever see files THIS run produced.
    OT=$(find "$OUT" -maxdepth 1 -name "*_temp_proc" -type d | head -1)
    rm -f "$OT"/b0_corrected_final.nii "$OT"/blip_up_b0_corrected*.nii \
          "$OT"/blip_down_b0_corrected*.nii "$OT"/deformation_FINV.nii* \
          "$OT"/deformation_MINV.nii*

    UP=$(ls "$OUT"/*.nii | grep -viE "temp_proc|_final|T1w|T2w|structural" | head -1)
    DOWN=$(ls "$OUT"/*.nii | grep -viE "temp_proc|_final|T1w|T2w|structural" | sed -n 2p)
    STR=$(ls "$OUT"/T2w*.nii 2>/dev/null | head -1)
    A="--up_data $UP"
    [ -n "${DOWN:-}" ] && A="$A --down_data $DOWN"
    [ -n "${STR:-}"  ] && A="$A --structural $STR"
    A="$A --DRBUDDI_step 2 --output $OUT/$(basename "${UP%.nii}")_TORTOISE_final.nii"

    echo "running $BE with --DRBUDDI_step 2 (Step0/Step1 read from the frozen fixture)"
    if [ "$(uname -s)" = "Darwin" ]; then
        # BSD time has no -v/-o, and caffeinate stops idle sleep inflating the timing.
        { /usr/bin/time -l caffeinate -i $EXE $A 2>"$OUT/time.raw"; } | tee "$OUT/run.log" | tail -3
        awk '/maximum resident set size/ {printf "\tMaximum resident set size (kbytes): %d\n", $1/1024}
             /real/ {print "\tElapsed (wall clock) time: " $1 " s"}' "$OUT/time.raw" > "$OUT/time.txt"
        cat "$OUT/time.raw" >> "$OUT/time.txt"
    else
    /usr/bin/time -v -o "$OUT/time.txt" $EXE $A 2>&1 | tee "$OUT/run.log" | tail -3
    fi
    # PIPESTATUS, not $? - the latter is tee's, so a crashed executable would look
    # like success and leave the (now deleted) artefacts missing rather than wrong.
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "  $BE FAILED with exit $rc - see $OUT/run.log" >&2
        exit "$rc"
    fi
    for f in b0_corrected_final.nii blip_up_b0_corrected.nii blip_down_b0_corrected.nii; do
        [ -f "$OT/$f" ] || { echo "  $BE produced no $f - Step2 did not complete" >&2; exit 1; }
    done
    # Real per-tag provenance. The fixture's copy says backend=CUDA and names the
    # CUDA binary, so inheriting it mislabels every WebGPU run on disk.
    python3 - "$OUT" "$BE" "$EXE" <<'PY'
import hashlib, json, os, sys
out, be, exe = sys.argv[1:4]
h = hashlib.sha256(open(exe, 'rb').read()).hexdigest()
json.dump({"dataset": "fast", "backend": be, "executable": exe, "exe_sha256": h,
           "mode": "DRBUDDI_step 2 from frozen fixture",
           "fixture": "benchmark/fast/DRB_FIXTURE"},
          open(os.path.join(out, "provenance.json"), "w"), indent=2)
PY
    echo "  wrote $OUT/provenance.json ($BE)"
    ;;

compare)
    A=$ROOT/benchmark/$DS/${2:?}; B=$ROOT/benchmark/$DS/${3:?}
    ta=$(find "$A" -maxdepth 1 -name "*_temp_proc" | head -1)
    tb=$(find "$B" -maxdepth 1 -name "*_temp_proc" | head -1)
    echo "Step2 (diffeomorphic) outputs - these are the GPU-bearing artefacts:"
    st=0
    n=0
    # NOTE the .nii.gz: DRBUDDI writes the deformation fields gzipped
    # (DRBUDDI.cxx:1233). Looking for plain .nii silently skipped the two direct
    # outputs of the stage this script exists to compare.
    for f in deformation_FINV.nii.gz deformation_MINV.nii.gz blip_up_b0_corrected.nii \
             blip_down_b0_corrected.nii b0_corrected_final.nii; do
        if [ ! -f "$ta/$f" ] || [ ! -f "$tb/$f" ]; then
            printf "  %-34s MISSING in one or both runs\n" "$f"; st=1; continue
        fi
        n=$((n+1))
        if cmp -s "$ta/$f" "$tb/$f"; then v="IDENTICAL"; else v="DIFFERS"; st=1; fi
        printf "  %-34s %s\n" "$f" "$v"
    done
    # "nothing compared" must not read as success.
    [ "$n" -eq 0 ] && { echo "  no artefacts compared - refusing to report success"; exit 1; }
    exit $st
    ;;
*)
    sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
