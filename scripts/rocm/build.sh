#!/bin/sh
# strata-rocm build - configure and build the HIP backend for the detected GPU.
#
# Separate build directory (build-hip/), separate CMake project (never the
# upstream one), complete logs, and a timeout that is reported as a timeout
# rather than as a compiler error.
#
# Usage: strata-rocm build [--jobs N] [--timeout S] [--target NAME] [--clean]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

JOBS=$(state_get limits.compile_parallelism 2>/dev/null || echo 4)
[ -n "$JOBS" ] || JOBS=4
TIMEOUT=$(state_get limits.build_timeout_s 2>/dev/null || echo 1800)
[ -n "$TIMEOUT" ] || TIMEOUT=1800
TARGET=""
CLEAN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --jobs|-j) JOBS=$2; shift 2 ;;
        --timeout) TIMEOUT=$2; shift 2 ;;
        --target) TARGET=$2; shift 2 ;;
        --clean) CLEAN=1; shift ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" $EX_USAGE ;;
    esac
done

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOGS_DIR/build-$TS.log"
CMAKE_DIR="$BUILD_DIR/cmake"
SRC_DIR="$SCRIPTS_DIR/hip"

have cmake || die "cmake not on PATH" $EX_BLOCKED
have hipcc || die "hipcc not on PATH" $EX_BLOCKED
[ -d "$BUILD_DIR/hipify" ] || die "no HIPIFY tree; run './strata-rocm hipify'" $EX_BLOCKED

[ "$CLEAN" = 1 ] && rm -rf "$CMAKE_DIR"
mkdir -p "$CMAKE_DIR"
: >"$LOG_FILE"

log "build: configure (jobs=$JOBS, timeout=${TIMEOUT}s)"
# No bare `time` here: it is a bash keyword, not POSIX, and the wrapper must
# run under dash too. The duration is in the log timestamps and state.json.
cmake -S "$SRC_DIR" -B "$CMAKE_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
    -DSTRATA_HIP_ROOT="$BUILD_DIR/hipify" \
    -DSTRATA_LLAMA_DIR="$BUILD_DIR/llama.cpp" \
    >>"$LOG_FILE" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    warn "configure FAILED - see $LOG_FILE"
    tail -30 "$LOG_FILE" >&2
    state_record "build-configure" "{\"status\":\"configure_failed\",\"log\":\"$LOG_FILE\"}"
    exit $EX_FAIL
fi
log "build: configure OK"

# -k 0: keep going after a failed translation unit. One build then reports the
# *whole* failure inventory instead of hiding 40 errors behind the first one,
# which is what makes the coverage claim in the report checkable.
if [ -n "$TARGET" ]; then
    BUILD_ARGS="$TARGET"
else
    BUILD_ARGS=""
fi
log "build: compiling (keep-going)"
run_bounded "$TIMEOUT" "$LOG_FILE" ninja -C "$CMAKE_DIR" -k 0 -j "$JOBS" $BUILD_ARGS
rc=$?

ERRS=$(grep -cE '^FAILED:|error:' "$LOG_FILE" 2>/dev/null)
[ -n "$ERRS" ] || ERRS=0
FAILED_TARGETS=$(grep -oE '^FAILED: [^ ]+' "$LOG_FILE" 2>/dev/null | sed 's/^FAILED: //' | sort -u | tr '\n' ' ')

case $rc in
    0)  STATUS=ok ;;
    $EX_TIMEOUT) STATUS=timeout ;;
    *)  STATUS=failed ;;
esac

BIN_COUNT=$(find "$CMAKE_DIR" -maxdepth 2 -type f -executable ! -name '*.so' ! -name '*.a' 2>/dev/null | wc -l | tr -d ' ')
python3 - "$RESULTS_DIR/build.json" "$TS" "$STATUS" "$rc" "$ERRS" "$BIN_COUNT" "$LOG_FILE" "$JOBS" "$TIMEOUT" <<'PY'
import json, sys
out, ts, status, rc, errs, bins, log, jobs, timeout = sys.argv[1:10]
json.dump({"name": "build", "at": ts, "status": status, "exit": int(rc),
           "compiler_errors": int(errs), "executables": int(bins),
           "jobs": int(jobs), "timeout_s": int(timeout), "log": log},
          open(out, "w"), indent=2)
PY
state_record "build" "{\"status\":\"$STATUS\",\"exit\":$rc,\"compiler_errors\":$ERRS,\"executables\":$BIN_COUNT}"

printf '\nbuild: %s (exit %s, %s compiler error lines, %s executables)\n' "$STATUS" "$rc" "$ERRS" "$BIN_COUNT"
[ -n "$FAILED_TARGETS" ] && printf 'failed targets: %s\n' "$FAILED_TARGETS"
if [ -s "$CMAKE_DIR/unsupported-sources.txt" ] && grep -q . "$CMAKE_DIR/unsupported-sources.txt"; then
    printf 'excluded as unsupported (must be reported, not forgotten):\n'
    sed 's/^/  - /' "$CMAKE_DIR/unsupported-sources.txt"
fi
printf 'log: %s\n' "$LOG_FILE"

[ "$STATUS" = timeout ] && exit $EX_TIMEOUT
[ "$STATUS" = failed ] && exit $EX_FAIL
exit $EX_OK
