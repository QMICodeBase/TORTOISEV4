#!/usr/bin/env python3
"""Hash the golden-vector record trees so drift becomes detectable.

WHY. The record trees are gitignored (multi-GB), so a fresh clone has none of them
and `git diff` can never tell you a record changed. Nothing bound the captured
vectors to the CUDA binary that produced them, and nothing bound the synthetic
records to the generator that wrote them - so either could be regenerated out of
step with the code and every suite would still report the same counts. An audit
found the manifest.jsonl of an older capture already stale relative to the source
that writes it, which is the same failure with a smaller blast radius.

This writes a small, committable manifest: per-record digests, the capturing
binary's sha256 where provenance records it, and the generator's sha256 for the
synthetic tree. `check` re-hashes and reports what moved.

  manifest.py write            # regenerate benchmark/MANIFEST.json
  manifest.py check            # compare on-disk trees against it
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "benchmark", "MANIFEST.json")
# `slow` was captured 2026-08-21. It was initially missing from this list, so both
# this manifest and revalidate.sh reported PASS while blind to 106 records - the
# exact failure this file exists to prevent, one layer down.
TREES = [("fast", "benchmark/fast/CAPTURE/golden_vectors"),
         ("medium", "benchmark/medium/CAPTURE/golden_vectors"),
         ("slow", "benchmark/slow/CAPTURE/golden_vectors"),
         ("synthetic", "benchmark/synthetic_vectors")]
GENERATOR = "benchmark/scripts/make_synthetic_records.py"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_digest(d):
    """One digest per record, over its json and every blob, in sorted order."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            h.update(name.encode())
            h.update(sha(p).encode())
    return h.hexdigest()


def scan():
    out = {}
    for label, rel in TREES:
        base = os.path.join(ROOT, rel)
        if not os.path.isdir(base):
            out[label] = {"path": rel, "present": False}
            continue
        recs = {}
        for e in sorted(os.listdir(base)):
            p = os.path.join(base, e)
            if os.path.isdir(p):
                recs[e] = record_digest(p)
        entry = {"path": rel, "present": True, "count": len(recs), "records": recs}
        # Bind captured trees to the binary that produced them by HASHING THE
        # BINARY. An earlier version stored provenance.json's exe_sha256 and then
        # compared it to a fresh read of that same file - a tautology: rebuilding
        # the capturing binary with a changed kernel left the manifest reporting
        # "unchanged". Hash the executable itself so a rebuild is visible.
        prov = os.path.join(os.path.dirname(base), "provenance.json")
        if os.path.exists(prov):
            try:
                pj = json.load(open(prov))
                exe = pj.get("executable")
                entry["capture_exe_recorded"] = pj.get("exe_sha256")
                entry["capture_exe_now"] = sha(exe) if exe and os.path.exists(exe) else None
                entry["capture_exe_path"] = exe
            except Exception:
                pass
        if label == "synthetic":
            g = os.path.join(ROOT, GENERATOR)
            if os.path.exists(g):
                entry["generator"] = {"path": GENERATOR, "sha256": sha(g)}
        out[label] = entry
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    cur = scan()

    if mode == "write":
        # Overwriting silently is how a bad change gets laundered into the baseline.
        if os.path.exists(OUT) and "--force" not in sys.argv:
            old = json.load(open(OUT))["trees"]
            deltas = []
            for label, _ in TREES:
                o, c = old.get(label, {}), cur.get(label, {})
                orec, crec = o.get("records", {}), c.get("records", {})
                nch = len(set(orec) ^ set(crec)) + sum(
                    1 for k in set(orec) & set(crec) if orec[k] != crec[k])
                if nch:
                    deltas.append(f"{label}: {nch} record(s) differ")
            if deltas:
                print("refusing to overwrite an existing manifest:")
                for d in deltas:
                    print(f"  {d}")
                print("re-run with --force if this is intended.")
                return 1
        json.dump({"trees": cur}, open(OUT, "w"), indent=2, sort_keys=True)
        print(f"wrote {OUT}")
        for k, v in cur.items():
            if v.get("present"):
                print(f"  {k:<10} {v['count']:>4} records"
                      + (f"   capture exe {v['capture_exe_recorded'][:16]}"
                         if v.get("capture_exe_recorded") else "")
                      + (f"   generator {v['generator']['sha256'][:16]}"
                         if v.get("generator") else ""))
        return 0

    if not os.path.exists(OUT):
        raise SystemExit(f"no manifest at {OUT} - run 'manifest.py write' first")
    old = json.load(open(OUT))["trees"]

    ok = True
    for label, rel in TREES:
        o, c = old.get(label, {}), cur.get(label, {})
        if not c.get("present"):
            print(f"  {label:<10} MISSING on disk"); ok = False; continue
        if not o.get("present"):
            print(f"  {label:<10} not in manifest - run 'write'"); ok = False; continue
        orec, crec = o.get("records", {}), c.get("records", {})
        added = sorted(set(crec) - set(orec))
        removed = sorted(set(orec) - set(crec))
        changed = sorted(k for k in set(orec) & set(crec) if orec[k] != crec[k])
        if o.get("generator") and c.get("generator") and \
           o["generator"]["sha256"] != c["generator"]["sha256"]:
            print(f"  {label:<10} GENERATOR CHANGED since these records were written -"
                  f" regenerate or the records describe old semantics"); ok = False
        # Two distinct checks: (a) the capturing binary still exists and still
        # hashes to what produced these records; (b) it has not changed since the
        # manifest was written.
        rec_h, now_h = c.get("capture_exe_recorded"), c.get("capture_exe_now")
        if rec_h and now_h and rec_h != now_h:
            print(f"  {label:<10} CAPTURING BINARY CHANGED since these records were "
                  f"captured ({rec_h[:16]} -> {now_h[:16]}) - records describe an "
                  f"older build"); ok = False
        elif rec_h and now_h is None:
            print(f"  {label:<10} capturing binary {c.get('capture_exe_path')} is gone -"
                  f" cannot verify these records"); ok = False
        if added or removed or changed:
            ok = False
            print(f"  {label:<10} {len(added)} added, {len(removed)} removed, "
                  f"{len(changed)} changed")
            for k in (added + removed + changed)[:6]:
                print(f"      {k}")
        else:
            print(f"  {label:<10} {c['count']:>4} records, unchanged")
    print(f"\nRESULT: {'PASS' if ok else 'DRIFT DETECTED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
