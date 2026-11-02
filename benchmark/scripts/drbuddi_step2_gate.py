#!/usr/bin/env python3
"""Machine-readable acceptance gate for the isolated DRBUDDI Step2 comparison.

`drbuddi_isolated.sh compare` only does a bytewise `cmp`, which necessarily reports
DIFFERS for any legitimate backend difference — so it cannot enforce an acceptance
threshold, and the whole-stage evidence could regress while the release command
still reported success. This closes that.

THE REFERENCE CLASS. The gate is defined against NOFMA — the same CUDA source built
with `-fmad=false` only — run through the identical frozen fixture. That is a
CUDA-only control: it measures what a legitimate arithmetic variation of the
reference does to this stage. Judging the port against the CUDA-vs-CUDA
*run-to-run* spread instead would be a category error, because that spread measures
thread scheduling, not arithmetic sensitivity (CLAUDE.md 5.4).

WHAT THIS IS AND IS NOT. It is a REGRESSION gate: it detects the port drifting away
from the NOFMA-class behaviour it currently exhibits. It is not proof of
correctness — per-kernel golden vectors and the exact analytic references are that.
The multipliers below carry headroom above the observed values so ordinary noise
does not trip them, which also means a small real regression could pass. Tightening
them requires re-measuring the baseline, not editing the numbers.

  drbuddi_step2_gate.py baseline <nofma-tag>          # write the NOFMA baseline
  drbuddi_step2_gate.py check <test-tag> [--json]     # gate a backend against it
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_outputs  # noqa: E402

DS = "fast"
REF_TAG = "DRB_C1"                       # the CUDA run every comparison is against
BASELINE = os.path.join(HERE, "drbuddi_step2_baseline.json")
ARTEFACTS = ("blip_up_b0_corrected.nii",
             "blip_down_b0_corrected.nii",
             "b0_corrected_final.nii")

# Pre-specified, and deliberately loose enough not to trip on noise:
TOL_MULT = 1.5        # max|d|/p99 may be up to 1.5x NOFMA's (observed: 1.08-1.17x)
TOL_DIFF_PP = 2.0     # differing-voxel fraction within 2 percentage points of NOFMA
TOL_R_ABS = 5e-4      # Pearson r no more than 5e-4 below NOFMA's


def provenance(tag):
    """The binary that actually produced a run's artefacts."""
    p = os.path.join(ROOT, "benchmark", DS, tag, "provenance.json")
    if not os.path.exists(p):
        raise SystemExit(f"{tag} has no provenance.json - cannot attribute its output to a binary. "
                         f"Re-run: drbuddi_isolated.sh run {tag} <backend>")
    d = json.load(open(p))
    if "exe_sha256" not in d or "executable" not in d:
        raise SystemExit(f"{tag}/provenance.json predates per-run provenance (it is the fixture's "
                         f"inherited copy). Re-run: drbuddi_isolated.sh run {tag} <backend>")
    return d


def current_hash(path):
    import hashlib
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_fresh(tags):
    """Every run in the comparison must come from the binary that exists NOW.

    Checking only the run under test is not enough: a shared CUDA-source change
    rebuilds CUDA and WebGPU, the WebGPU run is correctly flagged stale and re-run,
    and the gate then scores fresh WebGPU output against an OBSOLETE CUDA reference
    and an obsolete NOFMA baseline - and passes. The reference class has to be as
    fresh as the thing being judged.
    """
    stale = []
    for t in tags:
        d = provenance(t)
        cur = current_hash(d["executable"])
        if cur is None:
            stale.append(f"{t}: {d['executable']} no longer exists")
        elif cur != d["exe_sha256"]:
            stale.append(f"{t}: produced by {d['exe_sha256'][:16]}, current is {cur[:16]} "
                         f"({d.get('backend', '?')})")
    if stale:
        raise SystemExit("STALE reference class - refusing to gate:\n  " + "\n  ".join(stale) +
                         "\n\nRe-run all three (CUDA, NOFMA, WebGPU) and re-baseline. Note "
                         "revalidate.sh does not build build_cuda_nofma, so a NOFMA rebuild is "
                         "a manual step.")


def temp_proc(tag):
    d = os.path.join(ROOT, "benchmark", DS, tag)
    for e in os.listdir(d):
        if e.endswith("_temp_proc"):
            return os.path.join(d, e)
    raise SystemExit(f"no *_temp_proc under {d} - was the run completed?")


def stats(tag):
    """max|d|/p99, Pearson r and differing fraction for each Step2 artefact,
    always against the same CUDA reference run."""
    out = {}
    for f in ARTEFACTS:
        ra = os.path.join(temp_proc(REF_TAG), f)
        rb = os.path.join(temp_proc(tag), f)
        if not (os.path.exists(ra) and os.path.exists(rb)):
            raise SystemExit(f"missing artefact {f} for {REF_TAG} or {tag}")
        a, _ = compare_outputs.load(ra)
        b, _ = compare_outputs.load(rb)
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        if a.shape != b.shape:
            raise SystemExit(f"{f}: shape mismatch {a.shape} vs {b.shape}")
        p99 = np.percentile(np.abs(a), 99)
        d = np.abs(a - b)
        out[f] = {"rel": float(d.max() / p99) if p99 > 0 else float(d.max()),
                  "r": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                  "diff_pct": float(100.0 * np.count_nonzero(d) / d.size)}
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    mode, tag = sys.argv[1], sys.argv[2]

    if mode == "baseline":
        check_fresh([REF_TAG, tag])
        s = stats(tag)
        json.dump({"reference_run": REF_TAG, "baseline_run": tag,
                   "note": "NOFMA = same CUDA source, -fmad=false only. CUDA-only control.",
                   # Pin the binaries this baseline describes, so `check` can refuse
                   # to score against an obsolete reference class.
                   "reference_exe_sha256": provenance(REF_TAG)["exe_sha256"],
                   "baseline_exe_sha256": provenance(tag)["exe_sha256"],
                   "stats": s}, open(BASELINE, "w"), indent=2)
        print(f"baseline written from {tag} -> {BASELINE}")
        for f, v in s.items():
            print(f"  {f:<30} rel {v['rel']:.4f}  r {v['r']:.8f}  differ {v['diff_pct']:.2f}%")
        return 0

    if mode != "check":
        raise SystemExit(__doc__)

    if not os.path.exists(BASELINE):
        raise SystemExit(f"no baseline at {BASELINE} - run 'baseline <nofma-tag>' first")
    bj = json.load(open(BASELINE))
    # All three legs must be current: the run under test, the CUDA reference it is
    # measured against, and the NOFMA run the thresholds came from.
    check_fresh([REF_TAG, bj.get("baseline_run", "DRB_N1"), tag])
    for key, t in (("reference_exe_sha256", REF_TAG),
                   ("baseline_exe_sha256", bj.get("baseline_run", "DRB_N1"))):
        want = bj.get(key)
        if want and want != provenance(t)["exe_sha256"]:
            raise SystemExit(f"baseline was recorded against a different {t} binary "
                             f"({want[:16]} vs {provenance(t)['exe_sha256'][:16]}) - re-baseline")
    base = bj["stats"]
    cur = stats(tag)

    ok = True
    rows = []
    for f in ARTEFACTS:
        b, c = base[f], cur[f]
        lim_rel = b["rel"] * TOL_MULT
        lim_r = b["r"] - TOL_R_ABS
        d_pp = abs(c["diff_pct"] - b["diff_pct"])
        f_ok = (c["rel"] <= lim_rel) and (c["r"] >= lim_r) and (d_pp <= TOL_DIFF_PP)
        ok &= f_ok
        rows.append((f, c, b, lim_rel, lim_r, d_pp, f_ok))

    if "--json" in sys.argv:
        print(json.dumps({"tag": tag, "pass": bool(ok), "baseline": base, "current": cur}, indent=2))
    else:
        print(f"Isolated DRBUDDI Step2: {tag} vs {REF_TAG}, gated against the NOFMA baseline\n")
        print(f"  {'artefact':<30} {'rel':>9} {'limit':>9} {'r':>13} {'d%%Δ':>7}  verdict")
        for f, c, b, lim_rel, lim_r, d_pp, f_ok in rows:
            print(f"  {f:<30} {c['rel']:>9.4f} {lim_rel:>9.4f} {c['r']:>13.8f} "
                  f"{d_pp:>7.2f}  {'PASS' if f_ok else 'FAIL'}")
        print(f"\n  Thresholds (pre-specified): rel <= {TOL_MULT}x NOFMA, "
              f"r >= NOFMA-{TOL_R_ABS}, differing fraction within {TOL_DIFF_PP} pp.")
        print("  Regression gate against a CUDA-only control - not a proof of correctness.")
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
