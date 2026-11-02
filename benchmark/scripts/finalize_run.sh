#!/bin/bash
# Post-run M0 evidence for one benchmark run: provenance.json + immutable stage cache.
#
# Usage: finalize_run.sh <fast|medium|slow> [CUDA|WebGPU|Metal|CPU]
#
# Idempotent and fully post-hoc: it reconstructs everything from the artefacts the
# run left behind (time.txt, gpu_samples.csv, run.log), so it can be re-run later to
# backfill a run that finished without it. Set STAGE_CACHE=0 to skip the cache copy.
set -euo pipefail

DS=${1:?usage: finalize_run.sh <fast|medium|slow> [backend]}
BACKEND=${2:-CUDA}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/benchmark/$DS/$BACKEND

case $BACKEND in
    # CUDA_* are repeat runs of the same CUDA binary into separate folders, used to
    # measure the run-to-run floor. They must match here too, or `set -u` aborts on
    # the unset EXE after a completed multi-minute run.
    DET|DET_*)   EXE=$ROOT/bin/TORTOISEProcess_cuda_det;   BUILDDIR=$ROOT/build_cuda_det ;;
    NOFMA)       EXE=$ROOT/bin/TORTOISEProcess_cuda_nofma; BUILDDIR=$ROOT/build_cuda_nofma ;;
    CUDA|CUDA_*|CAPTURE*) EXE=$ROOT/bin/TORTOISEProcess_cuda;   BUILDDIR=$ROOT/build_cuda ;;
    # *_DET arms MUST precede their generic counterpart, or WebGPU_* swallows
    # WebGPU_DET and records the non-deterministic binary under a _DET tag.
    CUDA_DET*)   EXE=$ROOT/bin/TORTOISEProcess_cuda_det;   BUILDDIR=$ROOT/build_cuda_det ;;
    WebGPU_DET*) EXE=$ROOT/bin/TORTOISEProcess_webgpu_det; BUILDDIR=$ROOT/build_webgpu_det ;;
    WebGPU|WebGPU_*) EXE=$ROOT/bin/TORTOISEProcess_webgpu; BUILDDIR=$ROOT/build_webgpu ;;
    Metal_DET*) EXE=$ROOT/bin/TORTOISEProcess_metal_det; BUILDDIR=$ROOT/build_metal_det ;;
    Metal|Metal_*) EXE=$ROOT/bin/TORTOISEProcess_metal; BUILDDIR=$ROOT/build_metal ;;
    CPU)    EXE=$ROOT/bin/TORTOISEProcess;        BUILDDIR=$ROOT/build_cpu ;;
    # Fail with the backend name rather than letting `set -u` report an unset EXE
    # two lines later, which says nothing about the actual cause.
    *) echo "finalize_run.sh: unknown backend '$BACKEND'" >&2; exit 1 ;;
esac

[ -f "$OUT/time.txt" ] || { echo "no time.txt in $OUT - run not finished?" >&2; exit 1; }

python3 - "$OUT" "$EXE" "$BUILDDIR" "$DS" "$BACKEND" "$ROOT" <<'PY'
import json, os, platform, subprocess, sys
out, exe, builddir, ds, backend, root = sys.argv[1:7]

def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return ""

def cmakevar(k):
    try:
        for l in open(os.path.join(builddir, "CMakeCache.txt")):
            if l.startswith(k + ":"): return l.split("=", 1)[1].strip()
    except Exception: pass
    return ""

wall = rss = ""
for l in open(os.path.join(out, "time.txt")):
    if "Elapsed (wall" in l: wall = l.split(": ", 1)[1].strip()
    if "Maximum resident" in l: rss = int(l.split(": ")[1])

# Peak GPU memory, attributed per-pid. The sampler emits one row per compute
# process on the card, so a bare max() over the memory column would report a
# concurrent job's footprint as this run's. Take the largest per-pid peak: with a
# single run that is exactly its peak, and with a stray process present it reports
# the biggest single consumer rather than a meaningless blend.
gpu = 0
per_pid = {}
gs = os.path.join(out, "gpu_samples.csv")
if os.path.exists(gs):
    for l in open(gs):
        p = l.split(",")
        if len(p) == 3 and p[2].strip().isdigit() and p[1].strip().isdigit():
            pid = p[1].strip()
            per_pid[pid] = max(per_pid.get(pid, 0), int(p[2]))
    if per_pid:
        gpu = max(per_pid.values())
        if len(per_pid) > 1:
            print(f"WARNING: {len(per_pid)} compute processes seen in {gs};"
                  f" peak_gpu_mib is the largest single one ({gpu} MiB),"
                  f" not necessarily this run's", file=sys.stderr)

# TORTOISE echoes its own full command line as the second line of its output.
invocation = ""
rl = os.path.join(out, "run.log")
if os.path.exists(rl):
    for l in list(open(rl, errors="replace"))[:5]:
        if "TORTOISEProcess" in l and "--up_data" in l:
            invocation = l.strip(); break

# macOS has no nvidia-smi; read the adapter identity from system_profiler instead.
def _is_mac(): return platform.system() == "Darwin"
def gpu_name():
    if _is_mac():
        return sh("system_profiler SPDisplaysDataType | awk -F': ' '/Chipset Model/{print $2; exit}'")
    return sh("nvidia-smi --query-gpu=name --format=csv,noheader").split("\n")[0]
def gpu_uuid():
    if _is_mac(): return sh("sysctl -n hw.model")
    return sh("nvidia-smi --query-gpu=uuid --format=csv,noheader").split("\n")[0]
def gpu_driver_version():
    if _is_mac(): return "Metal on macOS " + platform.mac_ver()[0]
    return sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").split("\n")[0]


d = {
    "dataset": ds, "backend": backend,
    "utc": sh("date -u +%Y-%m-%dT%H:%M:%SZ"),
    "git_sha": sh(f"git -C {root} rev-parse HEAD"),
    "git_dirty": bool(sh(f"git -C {root} status --porcelain")),
    "executable": exe,
    "exe_sha256": ((sh(f"sha256sum {exe}") or "").split() or [""])[0],
    "build_dir": builddir,
    "cmake": {k: cmakevar(k) for k in
              ["CMAKE_BUILD_TYPE", "USECUDA", "USEWEBGPU", "USEMETAL", "CMAKE_CUDA_ARCHITECTURES", "ITK_DIR"]},
    "adapter": gpu_name(),
    "adapter_uuid": gpu_uuid(),
    "driver": gpu_driver_version(),
    "nvcc": sh("/usr/local/cuda/bin/nvcc --version | grep -oP 'release \\K[0-9.]+'"),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "invocation": invocation,
    "results": {"wall_clock": wall, "peak_host_rss_kb": rss, "peak_gpu_mib": gpu,
                "peak_gpu_mib_is_host_rss": _is_mac()},
}
json.dump(d, open(os.path.join(out, "provenance.json"), "w"), indent=2)
print(json.dumps({"dataset": ds, "adapter": d["adapter"], "driver": d["driver"],
                  **d["results"]}, indent=2))
PY

# Immutable DIFFPREP stage cache, so DRBUDDI can be replayed in isolation (CLAUDE.md 4.4).
if [ "${STAGE_CACHE:-1}" = "1" ] && [ "$BACKEND" = "CUDA" ]; then
    CACHE=$OUT/stage_cache
    if [ ! -d "$CACHE" ]; then
        mkdir -p "$CACHE"
        for d in "$OUT"/*_temp_proc; do
            [ -d "$d" ] || continue
            cp -a "$d" "$CACHE/"
        done
        chmod -R a-w "$CACHE"
    fi
    echo "stage cache: $(du -sh "$CACHE" | cut -f1) (read-only)"
fi
