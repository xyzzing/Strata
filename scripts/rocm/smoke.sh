#!/bin/sh
# strata-rocm smoke - build and run the Stage 0 HIP smoke test on the real card.
#
# This is the only check that proves HIP can allocate, launch and compute on the
# RX 7900 XTX. It must run where /dev/kfd and /dev/dri are visible; inside the
# DSH workspace sandbox they are not, and the test reports that as blocked
# rather than failed (see PORTING.md, "Running GPU work").
#
# Usage: strata-rocm smoke [--arch gfx1100] [--timeout 120] [--build-only]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

ARCH=gfx1100
LIMIT=120
BUILD_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --arch) ARCH=$2; shift 2 ;;
        --timeout) LIMIT=$2; shift 2 ;;
        --build-only) BUILD_ONLY=1; shift ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" $EX_USAGE ;;
    esac
done

SRC="$TESTS_DIR/smoke/smoke_hip.cpp"
OUT_DIR="$ART_DIR/smoke"
BIN="$OUT_DIR/smoke_hip.$ARCH"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOGS_DIR/smoke-$TS.log"

mkdir -p "$OUT_DIR"
[ -f "$SRC" ] || die "missing $SRC" $EX_BLOCKED
have hipcc || die "hipcc not on PATH" $EX_BLOCKED

: >"$LOG_FILE"
log "smoke: compiling $SRC for $ARCH"
hipcc --offload-arch="$ARCH" -O2 -Wall "$SRC" $HIP_RUNTIME_LIB -o "$BIN" >>"$LOG_FILE" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    warn "smoke compile FAILED (see $LOG_FILE)"
    state_record "smoke-compile" "{\"status\":\"compile_failed\",\"arch\":\"$ARCH\",\"log\":\"$LOG_FILE\"}"
    state_gate S0-hardware-hip-smoke failed "$LOG_FILE"
    exit $EX_FAIL
fi
log "smoke: compiled OK -> $BIN"

if [ "$BUILD_ONLY" = 1 ]; then
    state_record "smoke-compile" "{\"status\":\"compiled\",\"arch\":\"$ARCH\",\"binary\":\"$BIN\"}"
    exit $EX_OK
fi

if [ "$(gpu_devices_hidden)" = "true" ]; then
    warn "GPU device nodes are not visible in this namespace; cannot run"
    state_record "smoke-run" "{\"status\":\"blocked\",\"reason\":\"device_nodes_hidden\",\"binary\":\"$BIN\"}"
    state_gate S0-hardware-hip-smoke blocked "device nodes hidden by sandbox; rerun outside"
    exit $EX_BLOCKED
fi

log "smoke: running (limit ${LIMIT}s)"
run_bounded "$LIMIT" "$LOG_FILE" "$BIN" --gfx "$ARCH"
rc=$?
OUT=$(sed -n 's/^/  /p' "$LOG_FILE" | grep -E 'device_name|gcn_arch|int_exact|float_worst|SMOKE_|SMOKE FAIL|SMOKE RESULT|warp_size' || true)
printf '%s\n' "$OUT"

case $rc in
    0)
        printf '\nSMOKE: pass on %s\n' "$ARCH"
        state_record "smoke-run" "{\"status\":\"pass\",\"arch\":\"$ARCH\",\"binary\":\"$BIN\",\"log\":\"$LOG_FILE\"}"
        state_gate S0-hardware-hip-smoke passed "$LOG_FILE"
        exit $EX_OK
        ;;
    $EX_TIMEOUT)
        printf '\nSMOKE: TIMEOUT after %ss (not a compute failure)\n' "$LIMIT"
        state_record "smoke-run" "{\"status\":\"timeout\",\"arch\":\"$ARCH\",\"limit_s\":$LIMIT,\"log\":\"$LOG_FILE\"}"
        state_gate S0-hardware-hip-smoke failed "$LOG_FILE"
        exit $EX_TIMEOUT
        ;;
    *)
        printf '\nSMOKE: FAIL (exit %s)\n' "$rc"
        state_record "smoke-run" "{\"status\":\"fail\",\"arch\":\"$ARCH\",\"exit\":$rc,\"log\":\"$LOG_FILE\"}"
        state_gate S0-hardware-hip-smoke failed "$LOG_FILE"
        exit $EX_FAIL
        ;;
esac
