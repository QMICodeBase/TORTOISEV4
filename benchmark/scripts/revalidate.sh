#!/usr/bin/env bash
# Full re-validation after a code change. Everything here needs the GPU, so it is
# meant to run once an end-to-end job has released it.
#
# Rebuilds all three configurations and runs every gate, continuing past failures
# so one broken suite does not hide the state of the others. Exits non-zero if any
# gate failed; the summary at the end is the verdict.
#
#   benchmark/scripts/revalidate.sh            # build + all replay suites
#   benchmark/scripts/revalidate.sh --compare  # also diff the WebGPU end-to-end
#                                              # output against the CUDA reference
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
LIB=${TORTOISE_LIB:-$(cd "$(dirname "$0")/../../../tortoise_libraries" 2>/dev/null && pwd)}
export CPATH=$LIB/local/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$LIB/local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
CMAKE=$LIB/cmake/bin/cmake
# macOS has neither the pinned cmake nor an NVIDIA card for gpu_env.sh to pin.
case "$(uname -s)" in
Darwin)
    IS_MAC=1
    CMAKE=$(command -v cmake)
    export TORTOISE_WEBGPU_BACKEND=${TORTOISE_WEBGPU_BACKEND:-metal}
    export TORTOISE_WEBGPU_ADAPTER_TYPE=${TORTOISE_WEBGPU_ADAPTER_TYPE:-integrated}
    export TORTOISE_WEBGPU_VENDOR_ID=${TORTOISE_WEBGPU_VENDOR_ID:-0x106B}
    ;;
*)
    IS_MAC=0
    . benchmark/scripts/gpu_env.sh
    ;;
esac

# CRITICAL: gpu_replay links gpu_capture.cxx and calls the HOOKED host wrappers. If
# TORTOISE_GPU_CAPTURE is exported in the caller's shell, `gpu_replay --all` REWRITES
# the golden vectors in place with the binary-under-test's own output - and the
# capture quota exactly matches the record count, so the
# overwrite is complete and silent. The gate would then still report "bit-exact"
# against records it had just written. Clear it, and the other behaviour-changing
# switches, so a gate run cannot inherit them.
unset TORTOISE_GPU_CAPTURE TORTOISE_GPU_CAPTURE_N
unset TORTOISE_WEBGPU_ALLOW_ANY_ADAPTER TORTOISE_WEBGPU_CUDA_TEXFILTER
unset TORTOISE_METAL_CUDA_TEXFILTER TORTOISE_METAL_INFLIGHT TORTOISE_METAL_DEBUG \
      TORTOISE_METAL_NEGATIVE TORTOISE_METAL_PROBE_ONLY TORTOISE_METAL_DEVICE_NAME
unset TORTOISE_WEBGPU_SYNC_EACH_DISPATCH TORTOISE_WEBGPU_PROBE_ONLY

DO_COMPARE=0
[ "${1:-}" = "--compare" ] && DO_COMPARE=1

# Detect GPU jobs by resolving /proc/<pid>/exe, NOT by string-matching command
# lines. pgrep -f matches any process whose argv contains the pattern - including
# this script when the pattern appears in its own text, and any editor or shell
# that happens to mention a binary name. Comparing resolved executable paths is
# exact: a process is a GPU job iff its exe IS one of our binaries.
#
# Any GPU binary counts, not just WebGPU ones: a rebuild relinks the CUDA
# binaries too, and the replay suites contend for the card. Corrupting an
# in-flight floor measurement is exactly what this prevents.
busy=""
if [ "$IS_MAC" = 1 ]; then
    # No /proc. pgrep -x matches the full (untruncated) executable name on Darwin.
    for nm in TORTOISEProcess_metal TORTOISEProcess_metal_det TORTOISEProcess_webgpu \
              DRBUDDI_metal DRBUDDI_webgpu metal_replay webgpu_replay metal_probe webgpu_probe; do
        for pid in $(pgrep -x "$nm" 2>/dev/null || true); do
            [ "$pid" = "$$" ] && continue
            busy="$busy  $pid $nm"$'\n'
        done
    done
fi
for exe_link in /proc/[0-9]*/exe; do
    pid=${exe_link#/proc/}; pid=${pid%/exe}
    [ "$pid" = "$$" ] && continue
    target=$(readlink "$exe_link" 2>/dev/null) || continue    # not ours / gone
    case ${target##*/} in
        TORTOISEProcess*|DRBUDDI*|gpu_replay*|webgpu_replay|webgpu_probe|DRTAMAS*)
            busy="$busy  $pid $target"$'\n' ;;
    esac
done
if [ -n "$busy" ]; then
    echo "A GPU job is still running - binaries cannot be relinked and the GPU is contended:"
    printf '%s' "$busy"
    exit 1
fi

RESULTS=()
SKIPPED=0
record() { RESULTS+=("$1|$2|$3"); }   # name|PASS/FAIL/SKIP|detail
# SKIP exists so absent evidence is VISIBLE. Silently omitting a gate is how a
# missing build directory once produced "ALL GATES PASSED" for uncompiled code.
skip()   { RESULTS+=("$1|SKIP|$2"); SKIPPED=$((SKIPPED+1)); }

run_gate() {   # name, expected-count (empty = just require rc=0), command...
    # Set XDIVERGE_EXPECTED=<n> for suites containing records where the port is
    # KNOWN to differ from CUDA because CUDA is wrong (CLAUDE.md 0.0b). Those report
    # XDIVERGE, not PASS, and the count is pinned like every other: if a record stops
    # diverging, the backend has reproduced CUDA's bug and this must fail.
    local name=$1 want=$2; shift 2
    local want_xd=${XDIVERGE_EXPECTED:-0}
    echo; echo "=== $name ==="
    local out; out=$("$@" 2>&1); local rc=$?
    echo "$out" | tail -20
    local pass; pass=$(echo "$out" | grep -c '^PASS')
    local fail; fail=$(echo "$out" | grep -c '^FAIL')
    local xd;   xd=$(echo "$out"   | grep -c '^XDIVERGE')
    if [ "$rc" -eq 0 ] && [ "$fail" -eq 0 ] && { [ -z "$want" ] || [ "$pass" -eq "$want" ]; } && [ "$xd" -eq "$want_xd" ]; then
        record "$name" PASS "$pass/$want$([ "$want_xd" -gt 0 ] && echo " +${xd} expected divergence(s)")"
    else
        record "$name" FAIL "$pass pass, $fail fail, $xd xdiverge (expected $want pass, $want_xd xdiverge), rc=$rc"
    fi
}

# ---- builds -----------------------------------------------------------------
# A stale binary after a failed build has burned us before: a broken build must be
# a hard stop, not a warning that scrolls past.
# NOTE: build_cuda_nofma and build_cuda_det are NOT built here. They are variant
# validation builds, and rebuilding them is a manual step - but the NOFMA binary is
# part of the isolated-DRBUDDI gate's reference class, so if a shared CUDA source
# changes, that gate will (correctly) refuse to run until NOFMA is rebuilt and
# re-baselined by hand.
if [ "$IS_MAC" = 1 ]; then
    # build_metal_det shares bin/ and run_cuda_reference.sh routes Metal_DET* to it,
    # so a stale copy is a live trap - build it here rather than trusting it.
    BUILD_CFGS="build_metal build_metal_det build_webgpu"
else
    BUILD_CFGS="build_cpu build_cuda build_webgpu"
fi
for cfg in $BUILD_CFGS; do
    if [ ! -d "$cfg" ]; then
        # NOT a skip: bin/ is shared and never cleaned, so continuing here would
        # grade stale binaries and still print ALL GATES PASSED.
        record "build $cfg" FAIL "not configured - cannot verify bin/ is current"
        echo "$cfg is not configured; bin/ may hold stale binaries. Stopping."
        printf '%s\n' "${RESULTS[@]}" | column -t -s'|'
        exit 1
    fi
    echo; echo "=== build $cfg ==="
    if timeout 3600 $CMAKE --build "$cfg" -j 12 >"/tmp/${cfg}_build.log" 2>&1; then
        record "build $cfg" PASS "clean"
    else
        record "build $cfg" FAIL "see /tmp/${cfg}_build.log"
        grep -m5 -E 'error:' "/tmp/${cfg}_build.log"
        echo "BUILD FAILED - stopping before any gate runs on a stale binary."
        printf '%s\n' "${RESULTS[@]}" | column -t -s'|'
        exit 1
    fi
done

# ---- record integrity -------------------------------------------------------
# Before grading anything, confirm the records themselves have not drifted. They are
# gitignored, so git cannot tell you this; and nothing else binds the captured
# vectors to their capturing binary or the synthetic ones to their generator.
if [ "$IS_MAC" = 1 ]; then
    skip "record manifest" "manifest is maintained on the Linux host"
else
echo; echo "=== record manifest ==="
if out=$(python3 benchmark/scripts/manifest.py check 2>&1); then
    echo "$out" | tail -5
    record "record manifest" PASS "record trees match MANIFEST.json"
else
    echo "$out" | tail -8
    record "record manifest" FAIL "records drifted - see above; regenerate or investigate"
fi
fi

# ---- replay gates -----------------------------------------------------------
# CUDA self-replay must be bit-exact; it is the control that proves the harness
# and the capture files themselves have not drifted.
if [ "$IS_MAC" = 1 ]; then
    # No CUDA on macOS, so gpu_replay's bit-exact self-check cannot run. The anchor
    # here is metal_replay against the SAME Linux-captured records, which carry CUDA's
    # recorded outputs (CLAUDE.md 3.1) - graded at the unchanged section 5 tolerances.
    run_gate "Metal replay fast"          132 \
        ./bin/metal_replay  --all benchmark/fast/CAPTURE/golden_vectors
    run_gate "Metal replay medium"        118 \
        ./bin/metal_replay  --all benchmark/medium/CAPTURE/golden_vectors
    run_gate "Metal replay slow"          106 \
        ./bin/metal_replay  --all benchmark/slow/CAPTURE/golden_vectors
    run_gate "Metal replay synthetic"     25 \
        ./bin/metal_replay  --all benchmark/synthetic_vectors
else
    run_gate "CUDA self-replay (bitwise)" 132 \
        ./bin/gpu_replay    --all benchmark/fast/CAPTURE/golden_vectors
fi
run_gate "WebGPU replay fast"         132 \
    ./bin/webgpu_replay --all benchmark/fast/CAPTURE/golden_vectors
run_gate "WebGPU replay medium"       118 \
    ./bin/webgpu_replay --all benchmark/medium/CAPTURE/golden_vectors
run_gate "WebGPU replay synthetic"    25 \
    ./bin/webgpu_replay --all benchmark/synthetic_vectors

# `slow` was captured 2026-08-21 and is the only large-matrix (140x140x92) coverage.
# Three of its ScaleUpdateField records are EXPECTED to diverge - see CLAUDE.md 0.0b.
# Guarded on existence because the record trees are gitignored, so a fresh clone has
# none of them; skipping is honest, silently omitting the suite is not.
if [ -d benchmark/slow/CAPTURE/golden_vectors ]; then
    if [ "$IS_MAC" = 1 ]; then
        skip "CUDA self-replay slow" "no CUDA on macOS; metal_replay slow is the anchor"
    else
    run_gate "CUDA self-replay slow"  106 \
        ./bin/gpu_replay    --all benchmark/slow/CAPTURE/golden_vectors
    # Was 103 pass + 3 XDIVERGE, when CUDA's FieldFindMaxLocalNorm bug made the port
    # correctly disagree on three ScaleUpdateField records. That bug is fixed
    # (0a0e44c) and everything was recaptured, so it is now a clean 106 with NO
    # expected divergences - four of those records are bit-exact.
    fi
    run_gate "WebGPU replay slow" 106 \
        ./bin/webgpu_replay --all benchmark/slow/CAPTURE/golden_vectors
else
    skip "CUDA self-replay slow" "benchmark/slow/CAPTURE/golden_vectors absent"
    skip "WebGPU replay slow"    "benchmark/slow/CAPTURE/golden_vectors absent"
fi

# ---- M2 adapter gate --------------------------------------------------------
if [ "$IS_MAC" = 1 ]; then
    run_gate "metal_probe" "" ./bin/metal_probe
fi
echo; echo "=== webgpu_probe (adapter selection + allocation) ==="
if out=$(./bin/webgpu_probe 2>&1); then
    echo "$out" | tail -6
    record "webgpu_probe" PASS "adapter selected, allocation round-trips"
else
    echo "$out" | tail -10
    record "webgpu_probe" FAIL "probe failed - see output above"
fi

# ---- M1 negative tests ------------------------------------------------------
# These assert the harness still REJECTS corrupted records. A suite that only ever
# confirms passes cannot tell "correct" from "not actually checking".
echo; echo "=== negative tests (harness rejects bad records) ==="
if [ "$IS_MAC" = 1 ]; then
    # negative_tests.sh drives gpu_replay (CUDA-only). The same assurance here comes
    # from metal_replay: a truncated blob must be REJECTED, not silently passed.
    tmpd=$(mktemp -d)
    cp -a benchmark/fast/CAPTURE/golden_vectors/WarpImage.0 "$tmpd/" 2>/dev/null
    blob=$(ls "$tmpd"/WarpImage.0/*.f32 2>/dev/null | head -1)
    if [ -n "$blob" ] && : > "$blob" && \
       ! ./bin/metal_replay "$tmpd/WarpImage.0" >/dev/null 2>&1; then
        record "negative tests" PASS "metal_replay rejects a truncated record"
    else
        record "negative tests" FAIL "metal_replay did NOT reject a truncated record"
    fi
    rm -rf "$tmpd"
elif out=$(src/tools/GPUReplay/negative_tests.sh \
             benchmark/fast/CAPTURE/golden_vectors/WarpImage.0 2>&1); then
    echo "$out" | tail -8
    record "negative tests" PASS "harness rejects corrupted records"
else
    echo "$out" | tail -15
    record "negative tests" FAIL "a corrupted record was NOT rejected"
fi

# ---- DIFFPREP GPU registration (the `pa` transformations) -------------------
if [ "$IS_MAC" = 1 ]; then
    skip "DIFFPREP pa registration" "baseline recorded against a CUDA binary"
else
# The other exact whole-stage gate, and it costs nothing: every run already writes
# this file. `pa` goes entirely to the GPU path (10 volumes <= GPU_CPU_ratio), so the
# reference is bit-reproducible across independent CUDA runs and any difference is
# attributable to the backend. This covers DIFFPREP's registration, which the
# isolated DRBUDDI gate does not reach.
echo; echo "=== DIFFPREP pa registration (vs NOFMA baseline) ==="
PA_BASE=benchmark/scripts/diffprep_pa_baseline.json
if [ ! -f "$PA_BASE" ]; then
    echo "  no baseline - run: diffprep_pa_gate.py baseline NOFMA"
    skip "DIFFPREP pa registration" "no NOFMA baseline recorded"
else
    # The premise must hold before the comparison means anything.
    if ! out=$(python3 benchmark/scripts/diffprep_pa_gate.py verify CUDA_C 2>&1); then
        echo "$out" | tail -4
        record "DIFFPREP pa registration" FAIL "reference is no longer bit-reproducible"
    elif out=$(python3 benchmark/scripts/diffprep_pa_gate.py check WebGPU 2>&1); then
        echo "$out" | tail -8
        record "DIFFPREP pa registration" PASS "within NOFMA-relative thresholds"
    else
        echo "$out" | tail -10
        # The gate refuses to grade a stale output and exits non-zero for that too.
        # Reporting that as a threshold breach sends the reader hunting a numerical
        # regression that does not exist - distinguish the two causes.
        if echo "$out" | grep -q "STALE"; then
            record "DIFFPREP pa registration" FAIL "stale - re-run the pa registration for this binary"
        else
            record "DIFFPREP pa registration" FAIL "outside NOFMA-relative thresholds"
        fi
    fi
fi

fi
# ---- isolated DRBUDDI Step2 -------------------------------------------------
if [ "$IS_MAC" = 1 ]; then
    skip "isolated DRBUDDI Step2" "reference class is CUDA/NOFMA-only"
else
# The only EXACT backend comparison available: the reference side is bit-reproducible
# (two CUDA runs from a frozen fixture agree byte-for-byte), so a difference here is
# attributable to the backend rather than to thread scheduling. Gated against the
# NOFMA baseline - a CUDA-only control - so it can regress loudly instead of silently.
echo; echo "=== isolated DRBUDDI Step2 (vs NOFMA baseline) ==="
DRB_BASE=benchmark/scripts/drbuddi_step2_baseline.json
if [ ! -f "$DRB_BASE" ]; then
    echo "  no baseline - run: drbuddi_isolated.sh snapshot CUDA && ... && drbuddi_step2_gate.py baseline DRB_N1"
    skip "isolated DRBUDDI Step2" "no NOFMA baseline recorded"
elif [ ! -d benchmark/fast/DRB_W1 ]; then
    echo "  no DRB_W1 run - run: drbuddi_isolated.sh run DRB_W1 WebGPU"
    skip "isolated DRBUDDI Step2" "baseline exists but no WebGPU run to gate"
elif [ ! -f benchmark/fast/DRB_W1/provenance.json ]; then
    echo "  DRB_W1 has no provenance.json - cannot attribute it to a binary."
    skip "isolated DRBUDDI Step2" "no provenance for DRB_W1"
else
    # Same content-hash staleness rule as the end-to-end gate. Without it, changing
    # a shader and rebuilding would still score the PREVIOUS binary's Step2 output
    # and report PASS.
    DRB_REC=$(python3 -c "import json;print(json.load(open('benchmark/fast/DRB_W1/provenance.json')).get('exe_sha256',''))" 2>/dev/null || true)
    DRB_CUR=$(sha256sum bin/TORTOISEProcess_webgpu 2>/dev/null | cut -d' ' -f1)
    if [ -z "$DRB_REC" ] || [ "$DRB_REC" != "$DRB_CUR" ]; then
        echo "  DRB_W1 was produced by a different binary (${DRB_REC:0:16} vs ${DRB_CUR:0:16})."
        record "isolated DRBUDDI Step2" FAIL "stale - re-run drbuddi_isolated.sh run DRB_W1 WebGPU"
    elif out=$(python3 benchmark/scripts/drbuddi_step2_gate.py check DRB_W1 2>&1); then
        echo "$out" | tail -9
        record "isolated DRBUDDI Step2" PASS "within NOFMA-relative thresholds"
    else
        echo "$out" | tail -12
        if echo "$out" | grep -q "STALE"; then
            record "isolated DRBUDDI Step2" FAIL "stale - re-run drbuddi_isolated.sh run DRB_W1 WebGPU"
        else
            record "isolated DRBUDDI Step2" FAIL "outside NOFMA-relative thresholds"
        fi
    fi
fi

fi
# ---- end-to-end comparison --------------------------------------------------
if [ "$IS_MAC" = 1 ]; then
    skip "end-to-end comparison" "reference output is the Linux CUDA run; see CLAUDE.md 7.5"
else
if [ "$DO_COMPARE" -eq 1 ]; then
    echo; echo "=== end-to-end fast: WebGPU vs CUDA ==="
    # Staleness by CONTENT HASH, not mtime. provenance.json records the sha256 of
    # the binary that produced the run, so this compares what actually matters: an
    # mtime-only check false-positives on a no-op relink (as it did) and would
    # false-negative on a binary restored with an older timestamp.
    W_STALE=0
    # Both halves of a comparison must be current. The CUDA reference was silently
    # exempt, and is in fact stale right now.
    CPROV=benchmark/fast/CUDA/provenance.json
    if [ -f "$CPROV" ]; then
        CREC=$(python3 -c "import json;print(json.load(open('$CPROV')).get('exe_sha256',''))" 2>/dev/null || true)
        CCUR=$(sha256sum bin/TORTOISEProcess_cuda 2>/dev/null | cut -d' ' -f1)
        if [ -n "$CREC" ] && [ "$CREC" != "$CCUR" ]; then
            echo "  WARNING: the CUDA REFERENCE benchmark/fast/CUDA was produced by"
            echo "           ${CREC:0:16}, current bin/TORTOISEProcess_cuda is ${CCUR:0:16}."
            record "end-to-end fast" FAIL "stale CUDA reference - re-run benchmark/fast/CUDA"
            W_STALE=1
        fi
    fi
    PROV=benchmark/fast/WebGPU/provenance.json
    if [ "$W_STALE" = "1" ]; then
        :
    elif [ ! -f "$PROV" ]; then
        echo "  WARNING: no provenance.json for the WebGPU run - cannot verify which"
        echo "           binary produced it. Treating as stale."
        record "end-to-end fast" FAIL "no provenance - cannot attribute the output to a binary"
        W_STALE=1
    else
        REC=$(python3 -c "import json;print(json.load(open('$PROV'))['exe_sha256'])" 2>/dev/null || true)
        CUR=$(sha256sum bin/TORTOISEProcess_webgpu 2>/dev/null | cut -d' ' -f1)
        if [ -z "$REC" ] || [ "$REC" != "$CUR" ]; then
            echo "  WARNING: bin/TORTOISEProcess_webgpu does not match the binary that"
            echo "           produced this output (${REC:0:16} vs ${CUR:0:16})."
            record "end-to-end fast" FAIL "stale output - produced by a different binary"
            W_STALE=1
        fi
    fi

    if [ "${W_STALE:-0}" = "1" ]; then
        :
    elif python3 benchmark/scripts/compare_outputs.py fast CUDA WebGPU --smoke; then
        # SMOKE TEST, not a correctness gate - see benchmark/README.md, "Why the
        # end-to-end comparison is not a gate". Four runs of identical code span
        # r 0.999067-0.999314 while a Pearson gate derived from the CUDA-vs-CUDA
        # floor sits at 0.999224, INSIDE that spread, so it passed or failed on the
        # draw. The NOFMA control (CUDA + one compiler flag) sits at 0.999208,
        # indistinguishable from the port. The bounds here only catch gross breakage.
        record "end-to-end fast (smoke)" PASS "sanity bounds - not a correctness gate"
    else
        record "end-to-end fast (smoke)" FAIL "GROSS breakage - r < 0.9986 or spread > 20"
    fi
fi

fi

# ---- verdict ----------------------------------------------------------------
echo; echo "================ SUMMARY ================"
printf '%s\n' "${RESULTS[@]}" | column -t -s'|'
if printf '%s\n' "${RESULTS[@]}" | grep -q '|FAIL|'; then
    echo "RESULT: FAILED"; exit 1
fi
if [ "$SKIPPED" -gt 0 ]; then
    # Deliberately not "ALL GATES PASSED": some evidence was not available.
    echo "RESULT: PASSED, but $SKIPPED gate(s) SKIPPED - see above. Not a full validation."
    exit 0
fi
# Leave an artefact. Until now there was no evidence anywhere that revalidate had
# ever passed - the claim lived only in prose.
python3 - "$PWD" <<'PYJSON'
import hashlib, json, os, subprocess, sys
root = sys.argv[1]
def sha(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest() if os.path.exists(p) else None
res = {
  "utc": subprocess.run(["date","-u","+%Y-%m-%dT%H:%M:%SZ"],capture_output=True,text=True).stdout.strip(),
  "git_sha": subprocess.run(["git","-C",root,"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
  "git_dirty": bool(subprocess.run(["git","-C",root,"status","--porcelain"],capture_output=True,text=True).stdout.strip()),
  "binaries": {n: sha(os.path.join(root,"bin",n)) for n in
               ("TORTOISEProcess_cuda","TORTOISEProcess_webgpu","gpu_replay","webgpu_replay",
                "TORTOISEProcess_cuda_nofma")},
  "result": "ALL GATES PASSED",
}
json.dump(res, open(os.path.join(root,"benchmark","LAST_REVALIDATE.json"),"w"), indent=2)
print("  wrote benchmark/LAST_REVALIDATE.json")
PYJSON
echo "RESULT: ALL GATES PASSED"
