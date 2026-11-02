#!/usr/bin/env bash
# Record CUDA's residual against the synthetic EXACT references, and persist it.
#
# WHY THIS EXISTS: the claim "CUDA is roughly three orders of magnitude less
# accurate than the port on QuadraticTransformImageC" was produced by an ad-hoc
# `gpu_replay --all benchmark/synthetic_vectors` invocation and left no artefact.
# No committed command reproduced it and no file recorded it, so the number was
# asserted rather than evidenced. `revalidate.sh` runs `gpu_replay` only against
# the CAPTURED vectors, never the synthetic ones.
#
# It is deliberately NOT a pass/fail gate. `gpu_replay` compares bitwise, and
# nothing is bit-exact against an analytic reference, so every line reports FAIL by
# construction — the residual is the datum. Gating on it would be meaningless.
#
#   benchmark/scripts/record_cuda_vs_exact.sh
#
# Writes benchmark/synthetic_vectors/cuda_vs_exact.txt.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=benchmark/synthetic_vectors/cuda_vs_exact.txt

for l in /proc/[0-9]*/exe; do
    t=$(readlink "$l" 2>/dev/null) || continue
    case ${t##*/} in
        TORTOISEProcess*|DRBUDDI*|gpu_replay*|webgpu_replay)
            echo "A GPU job is running (${l#/proc/} $t) - refusing to contend"; exit 1;;
    esac
done

. benchmark/scripts/gpu_env.sh

{
    echo "# CUDA vs the synthetic EXACT references"
    echo "# generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by $(basename "$0")"
    echo "#"
    echo "# gpu_replay compares BITWISE, so FAIL is expected on every line - nothing is"
    echo "# bit-exact against an analytic reference. The 'rel=' figure is the datum."
    echo "# Compare against the WebGPU residuals from:"
    echo "#   bin/webgpu_replay --all benchmark/synthetic_vectors"
    echo
    ./bin/gpu_replay --all benchmark/synthetic_vectors 2>&1
} > "$OUT"

echo "wrote $OUT"
grep -E "^(PASS|FAIL)" "$OUT" | sed 's/^/  /' | head -30
