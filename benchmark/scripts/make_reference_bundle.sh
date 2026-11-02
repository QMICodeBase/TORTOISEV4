#!/bin/bash
# Assemble a SELF-CONTAINED golden-vector reference suite for porting the WebGPU
# backend to another platform (macOS/Metal in particular).
#
#   make_reference_bundle.sh <output-dir> [--tar]
#
# WHY THIS EXISTS. `benchmark/` is ~114 GB, almost all of it raw datasets and
# pipeline outputs that a port does not need. The part that actually constitutes the
# test suite - the golden-vector records - is a few GB, and it is fully portable:
#
#   * bin/webgpu_replay links NO CUDA libraries, and
#   * every record carries its own `out` tensors, which ARE the CUDA reference.
#
# So a machine with no NVIDIA hardware and no CUDA toolkit can grade a new backend
# against exactly the reference, and exactly the tolerances, that the Vulkan backend
# passes here. What such a machine CANNOT do is capture new records - that needs
# TORTOISEProcess_cuda. Hence the bundle.
set -euo pipefail

OUT=${1:?usage: make_reference_bundle.sh <output-dir> [--tar]}
DO_TAR=${2:-}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

[ -e "$OUT" ] && { echo "refusing to overwrite existing $OUT" >&2; exit 1; }
mkdir -p "$OUT/records"

echo "assembling reference bundle -> $OUT"

# ---- record trees -----------------------------------------------------------
# slow is included only if it was actually captured; a silently-absent dataset is
# how a suite ends up looking more comprehensive than it is.
copied=""
for spec in "fast:$ROOT/benchmark/fast/CAPTURE/golden_vectors" \
            "medium:$ROOT/benchmark/medium/CAPTURE/golden_vectors" \
            "slow:$ROOT/benchmark/slow/CAPTURE/golden_vectors" \
            "synthetic:$ROOT/benchmark/synthetic_vectors"; do
    name=${spec%%:*}; src=${spec#*:}
    n=$(find "$src" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "  $name: ABSENT - not included"
        continue
    fi
    echo "  $name: $n records"
    cp -a "$src" "$OUT/records/$name"
    copied="$copied $name:$n"
done

# ---- provenance -------------------------------------------------------------
python3 - "$OUT" "$ROOT" "$STAMP" "$copied" <<'PY'
import hashlib, json, os, subprocess, sys
out, root, stamp, copied = sys.argv[1:5]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def rec_digest(d):
    h = hashlib.sha256()
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        if os.path.isfile(p):
            h.update(n.encode()); h.update(sha(p).encode())
    return h.hexdigest()

def sh(c):
    try:
        return subprocess.check_output(c, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

trees = {}
for name in sorted(os.listdir(os.path.join(out, "records"))):
    base = os.path.join(out, "records", name)
    if not os.path.isdir(base):
        continue
    recs = {e: rec_digest(os.path.join(base, e))
            for e in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, e))}
    ops = sorted({e.rsplit(".", 1)[0].split(".v")[0] for e in recs})
    trees[name] = {"count": len(recs), "ops": ops, "records": recs}

# Bind the captured trees to the CUDA binary that produced them, where known.
cap = {}
for ds in ("fast", "medium", "slow"):
    p = os.path.join(root, "benchmark", ds, "CAPTURE", "provenance.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            cap[ds] = {"exe_sha256": d.get("exe_sha256"), "invocation": d.get("invocation")}
        except Exception:
            pass

gen = os.path.join(root, "benchmark", "scripts", "make_synthetic_records.py")
json.dump({
    "bundle": "TORTOISE WebGPU golden-vector reference suite",
    "created_utc": stamp,
    "created_on": {
        "host_gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader") or "unknown",
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader") or "unknown",
        "git_commit": sh(f"git -C {root} rev-parse HEAD") or "unknown",
        "git_dirty": bool(sh(f"git -C {root} status --porcelain")),
    },
    "capture_binaries": cap,
    "synthetic_generator_sha256": sha(gen) if os.path.exists(gen) else None,
    "trees": trees,
}, open(os.path.join(out, "MANIFEST.json"), "w"), indent=2, sort_keys=True)

total = sum(t["count"] for t in trees.values())
print(f"  MANIFEST.json written: {total} records across {len(trees)} trees")
PY

# ---- the bundle carries its own integrity checker ---------------------------
# Deliberately NOT a copy of benchmark/scripts/manifest.py: that one also verifies
# repo state (capture binary still present, generator unchanged) which is
# meaningless off this machine. This one answers one question - did the records
# arrive intact.
cat > "$OUT/verify_bundle.py" <<'PY'
#!/usr/bin/env python3
"""Verify this bundle's records against its MANIFEST.json.

Answers exactly one question: did the records survive the transfer intact?
It does NOT run any backend - see verify.sh for that.
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def rec_digest(d):
    h = hashlib.sha256()
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        if os.path.isfile(p):
            h.update(n.encode()); h.update(sha(p).encode())
    return h.hexdigest()

man = json.load(open(os.path.join(HERE, "MANIFEST.json")))
ok = True
for name, tree in sorted(man["trees"].items()):
    base = os.path.join(HERE, "records", name)
    if not os.path.isdir(base):
        print(f"  {name:<10} MISSING"); ok = False; continue
    have = {e for e in os.listdir(base) if os.path.isdir(os.path.join(base, e))}
    want = set(tree["records"])
    missing, extra = sorted(want - have), sorted(have - want)
    bad = [r for r in sorted(want & have)
           if rec_digest(os.path.join(base, r)) != tree["records"][r]]
    if missing or extra or bad:
        ok = False
        print(f"  {name:<10} {len(missing)} missing, {len(extra)} extra, {len(bad)} corrupted")
        for r in (missing + extra + bad)[:5]:
            print(f"      {r}")
    else:
        print(f"  {name:<10} {tree['count']:>4} records intact")
print("\nRESULT:", "PASS" if ok else "CORRUPTION DETECTED")
sys.exit(0 if ok else 1)
PY
chmod +x "$OUT/verify_bundle.py"

# ---- one-command validation -------------------------------------------------
cat > "$OUT/verify.sh" <<'SH'
#!/bin/bash
# Validate a backend against this bundle.
#   ./verify.sh /path/to/bin/webgpu_replay
#
# Exits non-zero if integrity or any replay suite fails.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPLAY=${1:?usage: verify.sh /path/to/webgpu_replay}
[ -x "$REPLAY" ] || { echo "not executable: $REPLAY" >&2; exit 1; }

echo "=== 1. bundle integrity ==="
python3 "$HERE/verify_bundle.py" || exit 1

echo
echo "=== 2. replay every suite ==="
rc=0
for d in "$HERE"/records/*/; do
    name=$(basename "$d")
    printf "  %-10s " "$name"
    out=$("$REPLAY" --all "$d" 2>&1 | tail -1)
    echo "$out"
    # A suite passes when nothing failed, nothing was unusable and nothing was
    # unported. The line may carry a trailing ", N expected divergence(s)" - those
    # are records where the port is KNOWN to differ from CUDA because CUDA is wrong
    # (see README section 6), and the replay tool has already verified the
    # divergence is of the predicted size. Matching on a fixed suffix here treated
    # a passing slow suite as a failure.
    case "$out" in
        *" 0 failed, 0 unusable, 0 not yet ported"*) ;;
        *) rc=1 ;;
    esac
done

echo
if [ $rc -eq 0 ]; then echo "RESULT: ALL SUITES PASSED"
else echo "RESULT: FAILURES - see above. Do NOT relax a tolerance to clear one."; fi
exit $rc
SH
chmod +x "$OUT/verify.sh"

cp "$ROOT/benchmark/scripts/BUNDLE_README.md" "$OUT/README.md" 2>/dev/null || true

echo
du -sh "$OUT" | sed 's/^/  bundle size: /'

if [ "$DO_TAR" = "--tar" ]; then
    echo "  creating tarball (this takes a while at this size)..."
    tar -C "$(dirname "$OUT")" -czf "$OUT.tar.gz" "$(basename "$OUT")"
    du -sh "$OUT.tar.gz" | sed 's/^/  tarball: /'
    sha256sum "$OUT.tar.gz" | cut -c1-16 | sed 's/^/  tarball sha256 (first 16): /'
fi
echo "done."
