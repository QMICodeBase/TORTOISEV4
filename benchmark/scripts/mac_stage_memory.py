#!/usr/bin/env python3
"""Compare per-stage peak memory between two backend runs.

Reads the `[PROFILE] <stage> <sec> peak_MiB <P> grew_MiB <G>` lines that
ProfileScope emits (src/main/tortoise_profile.h) from each run's time.txt and
reports, per stage, how much each backend RAISED the process high-water mark.

The point is localisation: a whole-run peak tells you a backend uses more memory,
not WHERE. grew_MiB attributes the growth to the stage that caused it.

  mac_stage_memory.py <runA-dir> <runB-dir> [--tol-mib N] [--tol-frac F]

Exits non-zero if any stage diverges by more than the tolerance, so it can gate.
"""
import sys, os, re, collections

PAT = re.compile(r"\[PROFILE\] (\S+) ([\d.]+) peak_MiB ([\d.]+) grew_MiB ([\d.]+)"
                 r"(?: foot_MiB ([\d.]+))?")

def load(d):
    grew, peak, foot = collections.OrderedDict(), {}, {}
    for name in ("time.txt", "run.log"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        for line in open(p, errors="replace"):
            m = PAT.search(line)
            if not m:
                continue
            st, pk, gr = m.group(1), float(m.group(3)), float(m.group(4))
            grew[st] = grew.get(st, 0.0) + gr      # a stage can run more than once
            peak[st] = max(peak.get(st, 0.0), pk)
            if m.group(5):                          # phys_footprint incl. GPU
                foot[st] = max(foot.get(st, 0.0), float(m.group(5)))
    return grew, peak, foot

def main():
    argv, args, opts = sys.argv[1:], [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if "=" in a:                      # --tol-mib=256
                k, v = a.split("=", 1); opts[k] = v
            elif i + 1 < len(argv):            # --tol-mib 256
                opts[a] = argv[i + 1]; i += 1
        else:
            args.append(a)
        i += 1
    if len(args) != 2:
        print(__doc__); return 2
    try:
        tol_mib  = float(opts.get("--tol-mib", 256))
        tol_frac = float(opts.get("--tol-frac", 0.25))
    except ValueError:
        print("--tol-mib and --tol-frac take a number"); return 2
    A, B = args
    ga, pa, fa = load(A)
    gb, pb, fb = load(B)
    if not ga or not gb:
        print("no [PROFILE] ... peak_MiB lines found - is the binary instrumented?")
        return 2
    stages = list(ga.keys()) + [s for s in gb if s not in ga]
    na, nb = os.path.basename(A.rstrip("/")), os.path.basename(B.rstrip("/"))
    print("%-32s %10s %10s %10s   %s" % ("stage", na[:10], nb[:10], "diff", "verdict"))
    print("%-32s %10s %10s %10s" % ("", "grew MiB", "grew MiB", "MiB"))
    bad = 0
    for st in stages:
        a, b = ga.get(st, 0.0), gb.get(st, 0.0)
        d = a - b
        lim = max(tol_mib, tol_frac * max(a, b))
        v = "ok"
        if abs(d) > lim:
            v = "DIVERGE"; bad += 1
        print("%-32s %10.1f %10.1f %10.1f   %s" % (st, a, b, d, v))
    print("\n%-32s %10.1f %10.1f %10.1f" % ("TOTAL peak RSS",
          max(pa.values()), max(pb.values()), max(pa.values()) - max(pb.values())))
    if fa and fb:
        # phys_footprint includes GPU/IOKit memory that RSS can miss - the honest
        # like-for-like figure between two GPU backends.
        print("%-32s %10.1f %10.1f %10.1f" % ("TOTAL peak phys_footprint",
              max(fa.values()), max(fb.values()), max(fa.values()) - max(fb.values())))
    print("\n%d stage(s) diverge beyond max(%.0f MiB, %.0f%%)" % (bad, tol_mib, 100 * tol_frac))
    return 1 if bad else 0

sys.exit(main())
