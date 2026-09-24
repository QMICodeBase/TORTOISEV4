#!/usr/bin/env python3
"""Whole-stage regression gate for DIFFPREP's GPU registration (`pa` transformations).

WHY `pa`. DIFFPREP splits volumes between the CUDA path and the ITK CPU path by index
arithmetic (DIFFPREP.cxx:600-634). With Nt=32, NGPUs=1, GPU_CPU_ratio=15, `ap` (138
volumes) puts 93 on the non-reproducible ITK path, but `pa` (10 volumes) goes entirely
to the GPU. Measured: `pa` is bit-identical across independent CUDA runs (0 of 127
live parameters differ) while `ap` is not. So the reference is noise-free for exactly
the code being ported, and it costs nothing - every run already writes the file.

WHAT THIS IS. A REGRESSION gate: it detects the port drifting from its own measured
behaviour. It is NOT a proof of equivalence, and in particular it is NOT the claim
"the port is within 1.5x of a compiler flag" - see the honesty note below.

HONESTY NOTE - read before quoting this gate.
An earlier version gated a single statistic (max |delta|) against 1.5x the NOFMA
control, and passed. That was selecting the favourable statistic. Measured ratios of
port to NOFMA, same data:

    median 1.72x   p75 2.06x   p90 2.49x   mean 1.38x   max 1.26x

The port's error distribution is genuinely SHIFTED relative to a compiler flag's -
heavier in the bulk, comparable in the tail. That is unsurprising for a different
shading language versus an FMA toggle, but it means no single "within Nx of NOFMA"
threshold is well founded, and picking the one that passes is exactly the failure
mode this project's own documents criticise elsewhere. So: all five statistics are
reported, the gate is against the PORT's recorded baseline (drift detection, which
needs no cross-backend threshold), and the NOFMA ratios travel with the output as
context rather than as a pass criterion.

Also note 113 of the 240 numbers in the file are structurally zero in every run, so
only 127 are live. An earlier criterion on the differing-COUNT was inert: it is 0 for
any pure-CUDA run and 126 for any perturbation whatsoever, so it could neither be
missed nor hit meaningfully - the same saturated-statistic error M6.md retracts.

  diffprep_pa_gate.py verify <tag>...          # assert the reference is reproducible
  diffprep_pa_gate.py baseline <port-tag> <nofma-tag>
  diffprep_pa_gate.py check <port-tag> [--json]
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DS = "fast"
REF_TAG = "CUDA"
BASELINE = os.path.join(HERE, "diffprep_pa_baseline.json")
FNAME = "pa_proc_moteddy_transformations.txt"

# Drift allowance against this port's own recorded values. Wide enough not to trip on
# a rebuild that changes nothing arithmetically, tight enough that a real change in
# the registration path shows up.
DRIFT = 1.25
STATS = ("median", "p75", "p90", "mean", "max")


def _file(tag):
    g = glob.glob(os.path.join(ROOT, "benchmark", DS, tag, "*_temp_proc", FNAME))
    if not g:
        raise SystemExit(f"{tag}: no {FNAME} - was the run completed?")
    return g[0]


def params(tag):
    return np.array([float(x) for x in
                     re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', open(_file(tag)).read())])


def provenance(tag):
    p = os.path.join(ROOT, "benchmark", DS, tag, "provenance.json")
    if not os.path.exists(p):
        raise SystemExit(f"{tag}: no provenance.json - cannot attribute it to a binary")
    d = json.load(open(p))
    if "exe_sha256" not in d or "executable" not in d:
        raise SystemExit(f"{tag}: provenance.json has no exe_sha256")
    return d


def check_fresh(tags):
    """Every leg must come from a binary that exists now. The Step2 gate learned this
    the hard way: checking only the run under test lets a fresh port output be scored
    against an obsolete reference."""
    stale = []
    for t in tags:
        d = provenance(t)
        exe = d["executable"]
        cur = (hashlib.sha256(open(exe, "rb").read()).hexdigest()
               if os.path.exists(exe) else None)
        if cur is None:
            stale.append(f"{t}: {exe} no longer exists")
        elif cur != d["exe_sha256"]:
            stale.append(f"{t}: produced by {d['exe_sha256'][:16]}, current {cur[:16]}")
    if stale:
        raise SystemExit("STALE - refusing to gate:\n  " + "\n  ".join(stale))


def stats(tag):
    ref = params(REF_TAG)
    cur = params(tag)
    if len(ref) != len(cur):
        raise SystemExit(f"{tag}: {len(cur)} parameters, reference has {len(ref)}")
    d = np.abs(cur - ref)
    nz = d[d > 0]
    live = int(np.sum(ref != 0))
    if nz.size == 0:
        return {"live": live, "ndiff": 0, "median": 0.0, "p75": 0.0,
                "p90": 0.0, "mean": 0.0, "max": 0.0}
    return {"live": live, "ndiff": int(nz.size),
            "median": float(np.median(nz)), "p75": float(np.percentile(nz, 75)),
            "p90": float(np.percentile(nz, 90)), "mean": float(np.mean(nz)),
            "max": float(np.max(nz))}


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    mode = sys.argv[1]

    if mode == "verify":
        ok = True
        for t in sys.argv[2:]:
            s = stats(t)
            good = s["ndiff"] == 0
            print(f"  {REF_TAG} vs {t:<10} {s['ndiff']}/{s['live']} live params differ  "
                  f"{'BIT-IDENTICAL' if good else 'DIFFERS - premise broken'}")
            ok &= good
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if mode == "baseline":
        if len(sys.argv) < 4:
            raise SystemExit("baseline needs <port-tag> <nofma-tag>")
        port, nofma = sys.argv[2], sys.argv[3]
        check_fresh([REF_TAG, port, nofma])
        sp, sn = stats(port), stats(nofma)
        # Overwriting an existing baseline is how a regression becomes the accepted
        # normal: re-baseline after a change and the gate that would have caught it
        # now certifies it. Refuse unless the caller says so explicitly, and show
        # what would move so the decision is informed rather than reflexive.
        if os.path.exists(BASELINE) and "--force" not in sys.argv:
            old_b = json.load(open(BASELINE))
            moved = [(k, old_b["port"][k], sp[k]) for k in STATS
                     if old_b.get("port", {}).get(k) and
                     abs(sp[k] - old_b["port"][k]) > 1e-12 * max(1.0, abs(old_b["port"][k]))]
            if moved:
                print(f"refusing to overwrite {BASELINE} - the port's own numbers moved:")
                for k, o, n in moved:
                    print(f"  {k:<8} {o:.6e} -> {n:.6e}   ({n/o:.4f}x)" if o else f"  {k}")
                print("\nIf this change is intended and understood, re-run with --force.")
                print("If it is NOT, you have just found the regression this gate exists to catch.")
                return 1
        json.dump({"reference_run": REF_TAG, "port_run": port, "nofma_run": nofma,
                   "reference_exe_sha256": provenance(REF_TAG)["exe_sha256"],
                   "port_exe_sha256": provenance(port)["exe_sha256"],
                   "nofma_exe_sha256": provenance(nofma)["exe_sha256"],
                   "port": sp, "nofma": sn}, open(BASELINE, "w"), indent=2)
        print(f"baseline written -> {BASELINE}")
        for k in STATS:
            print(f"  {k:<8} port {sp[k]:.3e}   NOFMA {sn[k]:.3e}   ratio {sp[k]/sn[k]:.2f}x"
                  if sn[k] else f"  {k:<8} port {sp[k]:.3e}")
        return 0

    if mode != "check":
        raise SystemExit(__doc__)
    if not os.path.exists(BASELINE):
        raise SystemExit(f"no baseline at {BASELINE} - run 'baseline <port> <nofma>' first")

    bj = json.load(open(BASELINE))
    tag = sys.argv[2]
    check_fresh([REF_TAG, tag, bj["nofma_run"]])
    for key, t in (("reference_exe_sha256", REF_TAG),
                   ("nofma_exe_sha256", bj["nofma_run"])):
        if bj.get(key) and bj[key] != provenance(t)["exe_sha256"]:
            raise SystemExit(f"baseline recorded against a different {t} binary - re-baseline")

    base, nof = bj["port"], bj["nofma"]
    cur = stats(tag)
    ok = all(cur[k] <= base[k] * DRIFT for k in STATS)

    if "--json" in sys.argv:
        print(json.dumps({"tag": tag, "pass": ok, "baseline": base,
                          "nofma": nof, "current": cur}, indent=2))
    else:
        print(f"DIFFPREP `pa` registration: {tag} vs {REF_TAG}")
        print(f"  reference is bit-identical across independent CUDA runs, so any")
        print(f"  difference here is attributable to the backend.\n")
        print(f"  {'stat':<8}{'current':>12}{'baseline':>12}{'drift lim':>12}"
              f"{'NOFMA':>12}{'vs NOFMA':>10}")
        for k in STATS:
            r = f"{cur[k]/nof[k]:.2f}x" if nof[k] else "-"
            print(f"  {k:<8}{cur[k]:>12.3e}{base[k]:>12.3e}{base[k]*DRIFT:>12.3e}"
                  f"{nof[k]:>12.3e}{r:>10}")
        print(f"\n  {cur['ndiff']}/{cur['live']} live parameters differ "
              f"(113 of the 240 numbers are structurally zero).")
        print("  Gate is DRIFT against this port's own baseline. The NOFMA column is")
        print("  context, not a criterion: the port is 1.7-2.5x NOFMA on bulk statistics")
        print("  and 1.26x on the tail, so no single cross-backend multiplier is founded.")
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
