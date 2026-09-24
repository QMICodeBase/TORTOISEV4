#!/usr/bin/env python3
"""Measure the run-to-run reproducibility floor from several CUDA runs.

CLAUDE.md 5.4 records that the end-to-end floor was first set from n = 2, taking the better
pair, which is not enough to judge a near-miss against. This computes every
pairwise CUDA-vs-CUDA comparison to get a distribution, then places the WebGPU
result against it.

  measure_floor.py fast CUDA CUDA_C CUDA_D CUDA_E --test WebGPU

Reports the CUDA-vs-CUDA range for both statistics and says whether the test
backend falls inside it. It does NOT rewrite any gate - widening a tolerance is a
decision for a human, and this tool only supplies the measurement.
"""
import argparse
import itertools
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Resolve final-output names through compare_outputs, which globs for them. The
# basename is dataset-dependent (`ap_TORTOISE_final.nii` on fast, but
# `sub-..._dir-AP_..._TORTOISE_final.nii` on medium/slow), so hardcoding one
# silently restricted this tool to `fast`.
sys.path.insert(0, HERE)
import compare_outputs  # noqa: E402


def compare(ref, test):
    """Run compare_outputs.py and pull the two statistics out of its report."""
    p = subprocess.run(
        [sys.executable, os.path.join(HERE, "compare_outputs.py"), ref, test],
        capture_output=True, text=True)
    out = p.stdout + p.stderr
    d = re.search(r"max\|diff\|/p99\s+([0-9.eE+-]+)", out)
    r = re.search(r"Pearson r\s+([0-9.eE+-]+)", out)
    f = re.search(r"differing\s+\d+\s+\(([0-9.]+)%\)", out)
    if not (d and r):
        raise RuntimeError("could not parse compare_outputs.py output:\n" + out)
    return float(d.group(1)), float(r.group(1)), float(f.group(1)) if f else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("backends", nargs="+", help="CUDA run folders to cross-compare")
    ap.add_argument("--test", help="backend to place against the measured floor")
    a = ap.parse_args()

    base = os.path.join(ROOT, "benchmark", a.dataset)

    def path(b):
        return compare_outputs.find_output(a.dataset, b)

    # A "run-to-run floor" must come from ONE binary. The original fast floor mixed
    # benchmark/fast/CUDA (ff4b85a8) with CUDA_C/D/E (e692f7d4), so 3 of its 6 pairs
    # were binary-vs-binary - i.e. it folded a source/compiler difference into a
    # measurement of thread-scheduling noise, and did so in the direction that makes
    # the derived gate looser.
    hashes = {}
    for b in a.backends:
        pj = os.path.join(base, b, "provenance.json")
        if not os.path.exists(pj):
            sys.exit(f"{b} has no provenance.json - cannot verify it came from the same binary")
        import json as _j
        hashes[b] = _j.load(open(pj)).get("exe_sha256", "")
    uniq = set(hashes.values())
    if len(uniq) != 1:
        sys.exit("refusing to derive a floor from mixed binaries:\n  " +
                 "\n  ".join(f"{b}: {h[:16]}" for b, h in hashes.items()) +
                 "\n\nA run-to-run floor requires one binary; otherwise it measures a\n"
                 "source/compiler difference as though it were scheduling noise.")

    pairs = list(itertools.combinations(a.backends, 2))
    if not pairs:
        sys.exit("need at least two CUDA runs to form a pair")

    print("CUDA-vs-CUDA pairs (the reproducibility floor)\n")
    print(f"  {'pair':<26} {'max|diff|/p99':>14} {'Pearson r':>16} {'differing %':>12}")
    ds, rs = [], []
    for x, y in pairs:
        d, r, f = compare(path(x), path(y))
        ds.append(d)
        rs.append(r)
        print(f"  {x + ' vs ' + y:<26} {d:>14.3f} {r:>16.6f} {f:>11.2f}%")

    def stats(v):
        n = len(v)
        m = sum(v) / n
        sd = (sum((z - m) ** 2 for z in v) / (n - 1)) ** 0.5 if n > 1 else float("nan")
        return m, sd

    dm, dsd = stats(ds)
    rm, rsd = stats(rs)
    print(f"\n  n = {len(pairs)} pairs from {len(a.backends)} runs")
    print(f"  max|diff|/p99  min {min(ds):.3f}  max {max(ds):.3f}  "
          f"mean {dm:.3f}  sd {dsd:.3f}")
    print(f"  Pearson r      min {min(rs):.6f}  max {max(rs):.6f}  "
          f"mean {rm:.6f}  sd {rsd:.6f}")

    if a.test:
        # Compare the test against EVERY reference run, not one arbitrary run: the
        # floor is a distribution over pairs, so a single test-vs-first comparison
        # is not measured the same way as the range it is placed against.
        print(f"\n{a.test} vs each CUDA run")
        print(f"  {'pair':<26} {'max|diff|/p99':>14} {'Pearson r':>16} {'differing %':>12}")
        tds, trs = [], []
        for b in a.backends:
            d, r, f = compare(path(b), path(a.test))
            tds.append(d)
            trs.append(r)
            print(f"  {a.test + ' vs ' + b:<26} {d:>14.3f} {r:>16.6f} {f:>11.2f}%")
        tdm, _ = stats(tds)
        trm, _ = stats(trs)
        print(f"\n  mean max|diff|/p99  {tdm:.3f}   CUDA mean {dm:.3f} (sd {dsd:.3f})")
        print(f"  mean Pearson r      {trm:.6f}   CUDA mean {rm:.6f} (sd {rsd:.6f})")

        # Deliberately descriptive, NOT a verdict. A min/max containment rule can
        # only ever LOOSEN as runs are added, which would make "add more runs until
        # it passes" a valid strategy - the exact failure the project's standing
        # rule against relaxing tolerances exists to prevent. State where the test
        # sits and let a human decide.
        for name, tv, cv, lower_is_better in (
                ("max|diff|/p99", tdm, ds, True),
                ("Pearson r", trm, rs, False)):
            # "Worse" means the CUDA pair disagrees with itself MORE than the test
            # does: a larger diff, or a lower correlation.
            worse = (sum(1 for z in cv if z > tv) if lower_is_better
                     else sum(1 for z in cv if z < tv))
            print(f"\n  {name}: {worse} of {len(cv)} CUDA pairs are worse than "
                  f"the test mean.")
            if worse == 0:
                print("    The test is worse than EVERY CUDA pair on this statistic,")
                print("    so it is not explained by run-to-run variation.")
        print("\n  This tool reports; it does not adjudicate. Widening a gate to admit "
              "a\n  result is a human decision and must be justified by more than this "
              "output.")


if __name__ == "__main__":
    main()
