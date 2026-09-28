#!/bin/sh
# Shared helpers for the Strata ROCm rollout scripts.
#
# POSIX sh only (no bashisms) so the wrapper runs under dash as well as bash.
# Every script sources this after ROCM_ROOT is known; if a script is executed
# directly, ROCM_ROOT is derived from $0 instead.
#
# Deliberately a plain shell library, not a framework: the whole rollout is
# shell + python3 + cmake, nothing else to install.

set -u

if [ -z "${ROCM_ROOT:-}" ]; then
    # <root>/scripts/rocm/lib/common.sh -> <root>
    ROCM_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
fi
export ROCM_ROOT

SCRIPTS_DIR="$ROCM_ROOT/scripts/rocm"
STRATA_DIR="${STRATA_DIR:-$ROCM_ROOT/Strata}"
ART_DIR="${ART_DIR:-$ROCM_ROOT/artifacts/rocm}"
BUILD_DIR="${BUILD_DIR:-$ROCM_ROOT/build-hip}"
TESTS_DIR="$ROCM_ROOT/tests/rocm"
VENDOR_DIR="$SCRIPTS_DIR/vendor"

STATE_FILE="$ART_DIR/state.json"
MANIFEST_FILE="$ART_DIR/env/manifest.json"
RESULTS_DIR="$ART_DIR/results"
LOGS_DIR="$ART_DIR/logs"

export ROCM_ROOT SCRIPTS_DIR STRATA_DIR ART_DIR BUILD_DIR TESTS_DIR VENDOR_DIR
export STATE_FILE MANIFEST_FILE RESULTS_DIR LOGS_DIR

# ---- exit codes -------------------------------------------------------------
# A timeout and a compiler error must be distinguishable by the caller, so they
# are different codes rather than "nonzero".
EX_OK=0
EX_FAIL=1
EX_USAGE=2
EX_BLOCKED=3        # a prerequisite is missing; nothing was attempted
EX_TESTFAIL=4       # a test ran and failed
EX_SKIP=5           # nothing failed, but required coverage was skipped: not a pass
EX_TIMEOUT=124

# ---- logging ----------------------------------------------------------------
LOG_FILE="${LOG_FILE:-}"
_ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

_log() {  # _log <level> <message...>
    _lvl=$1
    shift
    _line="[$(_ts)] $_lvl: $*"
    printf '%s\n' "$_line" >&2
    if [ -n "$LOG_FILE" ]; then
        mkdir -p "$(dirname -- "$LOG_FILE")" 2>/dev/null || true
        printf '%s\n' "$_line" >>"$LOG_FILE" 2>/dev/null || true
    fi
}
log()  { _log INFO "$@"; }
warn() { _log WARN "$@"; }
die()  { _log ERROR "$@"; exit "${2:-$EX_FAIL}"; }

have() { command -v "$1" >/dev/null 2>&1; }

# Fedora packages ROCm under /usr (not /opt/rocm), but hipconfig still reports
# ROCM_PATH=/opt/rocm, which does not exist here. Consequence: hipcc compiles
# device code fine and then fails to link, with "undefined symbol:
# hipLaunchKernel / __hipRegisterFatBinary". The runtime library itself is in
# the default search path, so naming it explicitly is the whole fix.
# Dropping ROCM_PATH=/usr does NOT work - hipcc then omits the hip link step
# entirely. Recorded in PORTING.md.
HIP_RUNTIME_LIB="${HIP_RUNTIME_LIB:--lamdhip64}"

# ---- bounded execution ------------------------------------------------------
# run_bounded <seconds> <logfile> <cmd...>
# Records the exit status; distinguishes timeout (EX_TIMEOUT) from failure.
run_bounded() {
    _limit=$1
    _logfile=$2
    shift 2
    mkdir -p "$(dirname -- "$_logfile")" 2>/dev/null || true
    log "run (limit ${_limit}s): $*" >>"$_logfile" 2>&1
    timeout --signal=TERM --kill-after=30 "$_limit" "$@" >>"$_logfile" 2>&1
    _rc=$?
    if [ "$_rc" -eq 124 ]; then
        warn "TIMEOUT after ${_limit}s: $* (see $_logfile)"
    elif [ "$_rc" -ne 0 ]; then
        warn "exit $_rc: $* (see $_logfile)"
    fi
    return "$_rc"
}

# ---- exit-code aggregation --------------------------------------------------
# worst_rc <rc>... : print the most severe exit code of a group.
#
# Severity (low to high): 0 ok, 3 blocked, 5 skip, 124 timeout, 1 failure,
# 4 test failure. A test failure outranks everything because it is the suite's
# actual verdict; a definitive failure outranks a timeout because "what do I do
# next" must not hide behind an ambiguous hang; a skip outranks a block because
# more ground was covered. An unknown nonzero ranks as a failure but keeps its
# own code, so a wrapper's novel status is never silently swallowed.
worst_rc() {
    _worst=0 _worst_rank=0
    for _rc in "$@"; do
        case "$_rc" in
            0) _rank=0 ;;
            3) _rank=1 ;;
            5) _rank=2 ;;
            124) _rank=3 ;;
            1) _rank=4 ;;
            4) _rank=5 ;;
            *) _rank=4 ;;
        esac
        if [ "$_rank" -gt "$_worst_rank" ]; then
            _worst=$_rc
            _worst_rank=$_rank
        fi
    done
    printf '%s' "$_worst"
}

# ---- python state helper ----------------------------------------------------
_py() { python3 "$SCRIPTS_DIR/lib/state.py" "$@"; }

state_init()      { _py init; }
state_set()       { _py set "$1" "$2"; }
state_get()       { _py get "$1"; }
state_gate()      { _py gate "$1" "$2" "${3:-}"; }
state_record()    { _py record "$1" "$2"; }
state_budget()    { _py budget "$1" "${2:-1}"; }
state_attempt()   { _py attempt "$1" "$2"; }   # attempt <signature> <outcome>

# ---- environment capture ----------------------------------------------------
# sha256 of a file, or "-" when unavailable. Used for vendored tools.
sha256_of() {
    [ -f "$1" ] || { printf '%s' "-"; return; }
    if have sha256sum; then sha256sum "$1" | awk '{print $1}';
    else python3 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1";
    fi
}

strata_revision() {
    git -C "$STRATA_DIR" rev-parse HEAD 2>/dev/null || printf '%s' "unknown"
}

strata_dirty() {
    # tracked modifications only: our own rollout files live outside the checkout
    if [ -n "$(git -C "$STRATA_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
        printf 'true'
    else
        printf 'false'
    fi
}

# True when the current process cannot see the GPU device nodes. The DSH
# sandbox mounts its own minimal /dev, so this is a property of the *shell
# namespace*, not of the machine: /sys/class/drm still shows the card.
gpu_devices_hidden() {
    if [ -e /dev/kfd ] && [ -e /dev/dri/renderD128 ]; then
        printf 'false'
    else
        printf 'true'
    fi
}

free_mib() {  # free_mib <mountpoint>
    df -Pm "$1" 2>/dev/null | awk 'NR==2 {print $4}'
}

mkdir -p "$ART_DIR" "$RESULTS_DIR" "$LOGS_DIR" "$ART_DIR/env" 2>/dev/null || true
