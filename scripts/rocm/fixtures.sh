#!/bin/sh
# strata-rocm fixtures - build the IQ/QK fixtures that iq_parity needs.
#
# Upstream does not publish them (its generator lived in the omitted tools/ tree),
# so they are built here: `make_iq_fixtures` produces VALID blocks with ggml's own
# encoders, then `ggufpy_reference.py` replaces the reference values with
# gguf-py's - a separate implementation in a different language from the kernels
# under test, which is what makes the comparison worth anything.
#
# Idempotent: existing fixtures are kept unless --force. `test.sh` calls this so
# the suite is reproducible from a bare checkout.
#
# Usage: strata-rocm fixtures [--force]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

OUT="$ART_DIR/iq_fixture"
BUILDER="$BUILD_DIR/cmake/make_iq_fixtures"
REFERENCE="$SCRIPTS_DIR/ggufpy_reference.py"
GGUF_PY="$BUILD_DIR/llama.cpp/gguf-py"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOGS_DIR/fixtures-$TS.log"

# The ten names iq_parity looks for.
NAMES="IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ1_M IQ4_NL IQ4_XS Q2_0 Q3_K"
complete() {
    for n in $NAMES; do
        [ -s "$OUT/$n.bin" ] && [ -s "$OUT/$n.f32" ] || return 1
    done
    return 0
}

if [ "$FORCE" = 0 ] && complete; then
    log "fixtures: present and complete at $OUT"
    exit $EX_OK
fi

[ -x "$BUILDER" ] || die "no fixture builder at $BUILDER; run './strata-rocm build'" $EX_BLOCKED
mkdir -p "$OUT"

log "fixtures: generating blocks with ggml's encoders"
if ! "$BUILDER" "$OUT" >>"$LOG_FILE" 2>&1; then
    tail -20 "$LOG_FILE" >&2
    die "fixture generation failed; see $LOG_FILE" $EX_FAIL
fi
tail -1 "$LOG_FILE" | sed 's/^/  /'

# The reference must not be a second copy of the thing under test. When gguf-py
# cannot run, the fixtures keep ggml's C values - still a working comparison,
# but no longer independent of the kernels - and that downgrade is recorded in
# results/, not just warned into a log nobody reconciles.
VENV_PY="$ROCM_ROOT/.venv-rocm/bin/python"
REF_KIND="ggml-c-not-independent"
if [ -d "$GGUF_PY" ] && [ -x "$VENV_PY" ]; then
    log "fixtures: replacing reference values with gguf-py's"
    REF_OUT=$(mktemp)
    if PYTHONPATH="$GGUF_PY" "$VENV_PY" "$REFERENCE" "$OUT" \
           --json "$RESULTS_DIR/fixtures-reference.json" >"$REF_OUT" 2>&1; then
        REF_KIND="gguf-py"
    else
        warn "gguf-py reference step failed; fixtures keep ggml's C values (see $LOG_FILE)"
    fi
    sed 's/^/  /' "$REF_OUT" | tee -a "$LOG_FILE"
    rm -f "$REF_OUT"
else
    warn "gguf-py or the project venv is missing; the .f32 files keep ggml's C values,"
    warn "so the reference is NOT independent of the kernels. Run './strata-rocm prepare'."
fi

complete || die "fixtures incomplete after generation" $EX_FAIL
state_record "fixtures" "{\"dir\":\"$OUT\",\"types\":\"$NAMES\",\"reference\":\"$REF_KIND\",\"provenance\":\"results/fixtures-reference.json\"}" >/dev/null
log "fixtures: complete at $OUT (reference: $REF_KIND)"
exit $EX_OK
