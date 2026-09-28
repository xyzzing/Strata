#!/bin/sh
# strata-rocm selftest - run the wrapper's own test suite (tests/harness/).
#
# These tests exercise the rollout tooling (testrun, state, report, help
# extraction, exit-code aggregation, fixture provenance) against throwaway
# ART_DIRs. They are the S1.6 harness self-check in standing, re-runnable form:
# the suite injects a defect (a fake binary that fails) and requires the
# classifier to call it failed.
#
# Exit codes follow the house convention: 0 all passed, 4 any failure,
# 5 nothing failed but some tests were skipped (not a full pass), 1 broken run.

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

HARNESS="$ROCM_ROOT/tests/harness"
[ -d "$HARNESS" ] || die "no harness test directory at $HARNESS" $EX_BLOCKED

OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT
# No network, no GPU, no real state: the discovery run is sandbox-safe.
python3 -m unittest discover -s "$HARNESS" -v >"$OUT" 2>&1
rc=$?
tail -n 20 "$OUT"

SKIPPED=$(sed -n 's/.*skipped=\([0-9][0-9]*\).*/\1/p' "$OUT" | head -1)

if [ "$rc" -ne 0 ]; then
    printf '\nselftest: FAILED (unittest rc %s)\n' "$rc"
    exit $EX_TESTFAIL
fi
if [ "${SKIPPED:-0}" -gt 0 ]; then
    printf '\nselftest: passed with %s skipped (skipped is not a full pass)\n' "$SKIPPED"
    exit $EX_SKIP
fi
printf '\nselftest: all harness tests passed\n'
exit $EX_OK
