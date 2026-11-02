#!/bin/bash
# mac_run.sh <label> <command...>
# Runs the command under /usr/bin/time -l, appends one JSONL record to
# benchmark/mac/runs.jsonl and keeps raw stdout/stderr beside it.
# Exit status is the command's own, so a caller can gate on it.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$REPO/benchmark/mac
mkdir -p "$OUT/logs"
LABEL=${1:?usage: mac_run.sh <label> <command...>}; shift
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$OUT/logs/${LABEL}_${STAMP}.log
TIMEF=$(mktemp)

# caffeinate -i prevents macOS idle sleep from inflating wall time - see the note in
# run_cuda_reference.sh. A measurement taken while the machine slept is void.
/usr/bin/time -l caffeinate -i "$@" >"$LOG" 2>>"$TIMEF"
RC=$?
# /usr/bin/time writes to stderr, so the command's stderr is in $TIMEF too.
cat "$TIMEF" >> "$LOG"

python3 - "$LABEL" "$STAMP" "$RC" "$LOG" "$TIMEF" "$REPO" "$@" <<'PY'
import hashlib, json, os, re, subprocess, sys
label, stamp, rc, log, timef, repo = sys.argv[1:7]
cmd = sys.argv[7:]
t = open(timef, errors="replace").read()
def grab(pat, cast=float):
    m = re.search(pat, t)
    return cast(m.group(1)) if m else None
def sha(p):
    try:
        with open(p, "rb") as f: return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError: return None
def git(*a):
    try: return subprocess.check_output(["git", "-C", repo, *a], text=True).strip()
    except Exception: return None
rec = {
  "ts": stamp, "label": label, "exit": int(rc), "cmd": cmd, "log": os.path.relpath(log, repo),
  "wall_s": grab(r"([\d.]+)\s+real"),
  "peak_rss_bytes": grab(r"(\d+)\s+maximum resident set size", int),
  "exe_sha256_16": sha(cmd[0]),
  "git_rev": git("rev-parse", "--short", "HEAD"),
  "git_dirty": bool(git("status", "--porcelain")),
}
p = os.path.join(repo, "benchmark/mac/runs.jsonl")
with open(p, "a") as f: f.write(json.dumps(rec) + "\n")
rss = rec["peak_rss_bytes"]
print("[mac_run] %s exit=%s wall=%ss peak_rss=%s" % (
    label, rc, rec["wall_s"], "%.2f GiB" % (rss/2**30) if rss else "?"))
PY
rm -f "$TIMEF"
exit $RC
