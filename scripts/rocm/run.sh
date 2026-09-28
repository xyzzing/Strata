#!/bin/sh
# strata-rocm run - launch the validated configuration on loopback.
#
# There is no validated engine binary yet: the HIP backend covers 19 kernel
# parity targets, not a forward pass (Stage 4 has not been reached, and the
# feasibility gate S3 has not been passed). This command therefore checks the
# preconditions it is contracted to check and then refuses with the exact
# blocker, rather than starting something unvalidated or pretending to.
#
# It never stops the existing server on port 8080 and never frees its memory.
#
# Usage: strata-rocm run [--port N] [--dry-run]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

PORT="${STRATA_PORT:-8081}"
DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" $EX_USAGE ;;
    esac
done

echo "strata-rocm run - preconditions"
echo "------------------------------------------------------------"

# 0. the port that is never ours, even if it looks free: 8080 belongs to the
# user's existing model server, and this rollout refuses to bind it by rule.
if [ "$PORT" = "8080" ]; then
    die "refusing --port 8080: that port belongs to the existing model server and this rollout never binds it, even when it is free (the loopback target is 8081)" $EX_USAGE
fi

# 1. port
if have ss && ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$PORT\$"; then
    die "port $PORT is already in use; choose another with --port (the server on 8080 is left alone)" $EX_BLOCKED
fi
echo "  port $PORT            free (loopback target)"

# 2. the existing service we must not disturb
if have ss && ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE '[:.]8080$'; then
    echo "  existing server        still listening on 8080 and left untouched"
else
    echo "  existing server        not detected on 8080"
fi

# 3. GPU visibility
if [ "$(gpu_devices_hidden)" = "true" ]; then
    die "GPU device nodes are hidden by this shell namespace; run outside the sandbox" $EX_BLOCKED
fi
echo "  GPU devices            visible"

# 4. free VRAM, only reported: freeing it would mean stopping the user's server
if have rocm-smi; then
    FREE=$(timeout 30 rocm-smi --showmeminfo vram 2>/dev/null | awk '
        /VRAM Total Memory/ {t=$NF} /VRAM Total Used Memory/ {u=$NF} END {if (t!="") print t-u}')
    echo "  free VRAM              ${FREE:-unknown} bytes (the existing server holds the rest; it is not ours to stop)"
fi

# 5. the thing that would actually be launched
# Only HIP-build candidates: $STRATA_DIR/build is the upstream CUDA tree, and
# launching a CUDA binary here would silently measure the wrong backend.
ENGINE=""
for cand in "$BUILD_DIR/cmake/strata" "$ROCM_ROOT/build/strata"; do
    [ -x "$cand" ] && ENGINE=$cand && break
done

if [ -z "$ENGINE" ]; then
    echo
    echo "  BLOCKED: no HIP engine binary exists yet."
    echo "  The port covers the kernel layer: the parity suite passes with declared"
    echo "  skips (exact counts: artifacts/rocm/results/test-kernels.json, or"
    echo "  './strata-rocm status'). src/core (device discovery, graphs, session) and"
    echo "  src/prefill are converted and tested at the kernel level, but no forward"
    echo "  pass has ever run, so there is nothing to launch that would produce a token."
    echo
    echo "  next: ./strata-rocm status"
    echo "        when the model shards arrive and verify: TASKS.md S4.1"
    state_set next_action '"when the model download to /mnt/LINUXWINSHARE lands: sha256-verify vs MODEL-DOWNLOAD-PROPOSAL.md, then S4.1 first inference on 8081"' >/dev/null
    state_set blocker '{"kind":"awaiting-user-download","detail":"no HIP engine binary has run a forward pass; kernel layer tested (see test-kernels.json), model weights are the user in-flight download"}' >/dev/null
    exit $EX_BLOCKED
fi

if [ "$DRY" = 1 ]; then
    echo "  would launch: HIP_VISIBLE_DEVICES=0 $ENGINE --port $PORT"
    exit $EX_OK
fi

echo "  launching $ENGINE on 127.0.0.1:$PORT"
HIP_VISIBLE_DEVICES=0 exec "$ENGINE" --host 127.0.0.1 --port "$PORT"
