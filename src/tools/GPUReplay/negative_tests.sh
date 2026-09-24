#!/bin/bash
# gpu_replay must reject unusable records loudly rather than silently passing.
# Builds damaged copies of a real record and checks each one fails with a
# specific message and a nonzero exit.
#
# Usage: negative_tests.sh <a_valid_record_dir>
set -u

SRC=${1:?usage: negative_tests.sh <record_dir>}
REPLAY=${REPLAY:-$(cd "$(dirname "$0")/../../.." && pwd)/bin/gpu_replay}
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

fails=0
check() {           # check <name> <dir> <expected substring>
    local name=$1 dir=$2 want=$3
    local out rc
    out=$("$REPLAY" "$dir" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then
        echo "  FAIL  $name: exited 0, should have refused the record"; fails=$((fails+1)); return
    fi
    if ! grep -qi -- "$want" <<<"$out"; then
        echo "  FAIL  $name: exit $rc but message did not mention '$want':"; echo "        $out"
        fails=$((fails+1)); return
    fi
    echo "  ok    $name -> $(grep -oi -m1 "[A-Za-z' ]*$want[^;]*" <<<"$out" | head -1)"
}

# 1. unknown operation
cp -r "$SRC" "$TMP/unknown"
python3 - "$TMP/unknown/record.json" <<'PY'
import json,sys
p=sys.argv[1]; d=json.load(open(p)); d["op"]="NoSuchOperation"; json.dump(d,open(p,"w"))
PY
check "unknown op" "$TMP/unknown" "no replay registered"

# 2. incompatible schema version
cp -r "$SRC" "$TMP/schema"
python3 - "$TMP/schema/record.json" <<'PY'
import json,sys
p=sys.argv[1]; d=json.load(open(p)); d["schema_version"]=99; json.dump(d,open(p,"w"))
PY
check "incompatible schema" "$TMP/schema" "schema_version"

# 3. truncated data blob
cp -r "$SRC" "$TMP/trunc"
blob=$(ls "$TMP/trunc"/in_*.f32 2>/dev/null | head -1)
if [ -n "$blob" ]; then
    truncate -s -64 "$blob"
    check "truncated blob" "$TMP/trunc" "truncated"
else
    echo "  skip  truncated blob: record has no input blob"
fi

# 4. corrupted data blob (right size, wrong bytes) - caught by the digest
cp -r "$SRC" "$TMP/corrupt"
blob=$(ls "$TMP/corrupt"/in_*.f32 2>/dev/null | head -1)
if [ -n "$blob" ]; then
    printf 'wrecked!' | dd of="$blob" bs=1 seek=16 conv=notrunc status=none
    check "corrupted blob" "$TMP/corrupt" "digest"
else
    echo "  skip  corrupted blob: record has no input blob"
fi

# 5. missing blob entirely
cp -r "$SRC" "$TMP/missing"
blob=$(ls "$TMP/missing"/in_*.f32 2>/dev/null | head -1)
if [ -n "$blob" ]; then
    rm -f "$blob"
    check "missing blob" "$TMP/missing" "missing blob"
else
    echo "  skip  missing blob: record has no input blob"
fi

# 6. malformed json
cp -r "$SRC" "$TMP/badjson"
echo '{ this is not json' > "$TMP/badjson/record.json"
check "malformed json" "$TMP/badjson" "not valid JSON"

echo
if [ $fails -eq 0 ]; then echo "negative tests: all passed"; else echo "negative tests: $fails FAILED"; fi
exit $fails
