#!/bin/sh
# strata-rocm test - run the independent tests and report honestly.
#
# Stages:
#   smoke    the Stage 0 HIP smoke test (build + run)
#   kernels  the in-tree parity suite: each kernel against its host / float64 /
#            ggml-derived reference
#   model    model-level checks; needs approved weights and a reference, and is
#            therefore blocked until Stage 4
#
# Three outcomes, never conflated: passed, failed, skipped. A test that could not
# run is skipped (exit 5), not a pass. A declared-unavailable test is still run,
# and is only counted skipped when its own output matches the declared reason.
#
# Usage: strata-rocm test --stage kernels [--timeout S] [--filter PATTERN] [--list]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

STAGE=""
TIMEOUT=$(state_get limits.small_gpu_test_timeout_s 2>/dev/null || echo 120)
[ -n "$TIMEOUT" ] || TIMEOUT=120
FILTER=""
LIST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE=$2; shift 2 ;;
        --timeout) TIMEOUT=$2; shift 2 ;;
        --filter) FILTER=$2; shift 2 ;;
        --list) LIST=1; shift ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" $EX_USAGE ;;
    esac
done
[ -n "$STAGE" ] || die "--stage is required (smoke|kernels|model|all)" $EX_USAGE

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOGS_DIR/test-$STAGE-$TS.log"
BUILD="$BUILD_DIR/cmake"
UNAVAILABLE="$TESTS_DIR/unavailable.json"
: >"$LOG_FILE"

run_smoke() {
    log "test: stage smoke"
    sh "$SCRIPTS_DIR/smoke.sh" --arch gfx1100 --timeout "$TIMEOUT"
}

run_kernels() {
    [ -d "$BUILD" ] || die "no build in $BUILD; run './strata-rocm build'" $EX_BLOCKED

    if [ "$LIST" = 1 ]; then
        python3 "$SCRIPTS_DIR/lib/testrun.py" --dir "$BUILD" --list ${FILTER:+--filter "$FILTER"}
        exit $EX_OK
    fi

    if [ "$(gpu_devices_hidden)" = "true" ]; then
        warn "GPU device nodes are not visible here; the parity suite cannot run"
        state_record "test-kernels" \
            '{"stage":"kernels","status":"blocked","reason":"device_nodes_hidden"}' >/dev/null
        state_set blocker '{"kind":"sandbox","detail":"GPU device nodes hidden by the shell namespace; run this outside the sandbox"}' >/dev/null
        return $EX_BLOCKED
    fi

    # iq_parity compares against fixtures upstream does not publish. They are
    # generated here (idempotent) rather than left as a permanently skipped test.
    if ! sh "$SCRIPTS_DIR/fixtures.sh" >>"$LOG_FILE" 2>&1; then
        warn "fixture generation failed; iq_parity will report missing fixtures (see $LOG_FILE)"
    fi

    log "test: stage kernels (per-test timeout ${TIMEOUT}s, serial)"
    # Serial on purpose: one GPU, and the existing model server holds most of its
    # memory. Running the suite in parallel would measure contention.
    # Not piped into tee: `cmd | tee` makes $? tee's status, so a failing suite
    # would report success. Capture first, echo after.
    OUT=$(mktemp)
    python3 "$SCRIPTS_DIR/lib/testrun.py" \
        --dir "$BUILD" \
        --timeout "$TIMEOUT" \
        --out "$RESULTS_DIR/test-kernels.json" \
        --unavailable "$UNAVAILABLE" \
        --args "$TESTS_DIR/testargs.json" \
        --logs "$ART_DIR/testlogs" \
        ${FILTER:+--filter "$FILTER"} >"$OUT" 2>&1
    RC=$?
    cat "$OUT"
    cat "$OUT" >>"$LOG_FILE"
    rm -f "$OUT"

    python3 - "$RESULTS_DIR/test-kernels.json" "$TS" "$BUILD" <<'PY'
import json, os, sys
path, ts, build = sys.argv[1:4]
d = json.load(open(path))
d["name"] = "test-kernels"
d["at"] = ts
unsup = os.path.join(build, "unsupported-sources.txt")
d["coverage"] = {
    "suite": "in-tree parity tests (src/kernels/*_parity.cpp) on gfx1100",
    "not_run": [f"{t['name']}: {t['detail']}" for t in d["tests"] if t["status"] == "skipped"],
    "unsupported_sources": [l.strip() for l in open(unsup) if l.strip()] if os.path.exists(unsup) else [],
    "caveat": "passing here means each kernel agrees with its own in-tree reference. It does not "
              "certify the quantization formats whose fixtures are missing, and it says nothing "
              "about the shape of a full forward pass.",
}
json.dump(d, open(path, "w"), indent=2)
PY

    for k in passed failed skipped; do
        V=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
            "$RESULTS_DIR/test-kernels.json" "$k")
        case "$V" in
            ''|*[!0-9]*) die "test-kernels.json has a non-numeric $k count: $V" $EX_FAIL ;;
        esac
        eval "N_$k=\$V"
    done
    state_set tests "{\"passed\":$N_passed,\"failed\":$N_failed,\"skipped\":$N_skipped,\"last\":\"kernels\"}" >/dev/null

    printf '\n  passed %s   failed %s   skipped %s\n' "$N_passed" "$N_failed" "$N_skipped"
    printf '  results: %s\n  logs:    %s\n' "$RESULTS_DIR/test-kernels.json" "$ART_DIR/testlogs"

    case $RC in
        0) state_gate S2-kernel-suite-gfx1100 passed "$RESULTS_DIR/test-kernels.json" >/dev/null ;;
        5) state_gate S2-kernel-suite-gfx1100 passed_with_skips "$RESULTS_DIR/test-kernels.json" >/dev/null ;;
        *) state_gate S2-kernel-suite-gfx1100 failed "$RESULTS_DIR/test-kernels.json" >/dev/null ;;
    esac
    if [ "$RC" -eq 0 ] || [ "$RC" -eq 5 ]; then
        state_set blocker null >/dev/null
        # Never move the stage backwards: a stage reached by other work must not
        # be undone by re-running this suite.
        CUR=$(state_get stage 2>/dev/null || echo 0)
        case "$CUR" in ''|*[!0-9]*) CUR=0 ;; esac
        [ "$CUR" -lt 2 ] && state_set stage '"2"' >/dev/null
        state_set next_action '"S3: replace the NVIDIA PTX in native_qsa_score.cu with a HIP reference path, then src/prefill (TASKS.md)"' >/dev/null
    fi
    return $RC
}

run_model() {
    warn "model stage is not available yet: it needs approved weights and an independent reference"
    state_record "test-model" \
        '{"stage":"model","status":"skipped","reason":"no approved weights and no reference: the Stage 3 feasibility gate has not been reached"}'
    return $EX_SKIP
}

RC=0
case "$STAGE" in
    smoke)   run_smoke; RC=$? ;;
    kernels) run_kernels; RC=$? ;;
    model)   run_model; RC=$? ;;
    all)     run_smoke; S1=$?; run_kernels; S2=$?
             # worst-of: a kernel-suite failure (4) must outrank a blocked or
             # skipped smoke run, or the suite's verdict would be masked.
             RC=$(worst_rc "$S1" "$S2") ;;
    *) die "unknown stage: $STAGE" $EX_USAGE ;;
esac

case $RC in
    0)  printf '\ntest --stage %s: all selected tests passed\n' "$STAGE" ;;
    $EX_SKIP) printf '\ntest --stage %s: passed with skipped coverage (NOT a full pass)\n' "$STAGE" ;;
    $EX_BLOCKED) printf '\ntest --stage %s: BLOCKED (nothing was proven)\n' "$STAGE" ;;
    $EX_TIMEOUT) printf '\ntest --stage %s: TIMEOUT\n' "$STAGE" ;;
    *)  printf '\ntest --stage %s: FAILED\n' "$STAGE" ;;
esac
exit $RC
