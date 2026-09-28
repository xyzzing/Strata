#!/bin/sh
# strata-rocm bench - bounded, recorded measurement.
#
# IMPORTANT SCOPE: the engine links but cannot load a model -- no weights, and the
# CPU expert half is switched off (TASKS S2.7c) -- so this cannot and does not
# produce time-to-first-token, prompt processing rate or tokens per second.
# Reporting those numbers now would be inventing them. What it measures today is kernel-level wall time for the ported
# targets, with the settings and the resource state recorded alongside, and it
# labels the result as such.
#
# Usage: strata-rocm bench [--target NAME] [--reps N] [--timeout S]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

TARGET=""
REPS=3
LIMIT=120
while [ $# -gt 0 ]; do
    case "$1" in
        --target) TARGET=$2; shift 2 ;;
        --reps) REPS=$2; shift 2 ;;
        --timeout) LIMIT=$2; shift 2 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" $EX_USAGE ;;
    esac
done

BUILD="$BUILD_DIR/cmake"
[ -d "$BUILD" ] || die "no build in $BUILD; run './strata-rocm build'" $EX_BLOCKED
[ "$(gpu_devices_hidden)" = "true" ] && die "GPU device nodes hidden by this shell namespace; run outside the sandbox" $EX_BLOCKED

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOGS_DIR/bench-$TS.log"
: >"$LOG_FILE"

# Default targets: the heaviest ported kernels, which are the ones worth timing.
if [ -z "$TARGET" ]; then
    TARGETS="kv_stream_parity qsa_parity s_gemv_parity sampler_parity"
else
    TARGETS=$TARGET
fi

VRAM_BEFORE=$(timeout 30 rocm-smi --showmeminfo vram 2>/dev/null | awk '/VRAM Total Used Memory/ {print $NF}')
log "bench: targets '$TARGETS', $REPS repetitions, ${LIMIT}s each, serial"

BENCH_RC=0
python3 - "$BUILD" "$TARGETS" "$REPS" "$LIMIT" "$RESULTS_DIR/bench.json" "$TS" "$LOG_FILE" "$VRAM_BEFORE" <<'PY' || BENCH_RC=$?
import json, os, subprocess, sys, time

build, targets, reps, limit, out, ts, log, vram_before = sys.argv[1:9]
reps, limit = int(reps), int(limit)

def vram_used():
    try:
        r = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True,
                           text=True, timeout=30)
        for line in r.stdout.splitlines():
            if "VRAM Total Used Memory" in line:
                return int(line.split()[-1])
    except Exception:
        pass
    return None

entries = []
for name in targets.split():
    path = os.path.join(build, name)
    if not os.path.isfile(path):
        entries.append({"target": name, "status": "unavailable",
                        "reason": "not built"})
        continue
    times, rcs = [], []
    for _ in range(reps):
        t0 = time.time()
        try:
            p = subprocess.run([path, "--selftest"], capture_output=True, text=True, timeout=limit)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        times.append(round(time.time() - t0, 4))
        rcs.append(rc)
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    entries.append({
        "target": name, "status": "ok" if all(r == 0 for r in rcs) else "failed",
        "reps": reps, "seconds": times, "median_s": median,
        "spread_s": round(max(times) - min(times), 4),
        "exit_codes": rcs,
    })

doc = {
    "name": "bench", "at": ts,
    "scope": ("kernel-level wall time for the ported parity targets. NOT a model "
              "benchmark: the engine links but cannot load a model (no weights; the "
              "CPU expert path is off, TASKS S2.7c), so time-to-first-token, "
              "prompt-processing rate and tokens/second are deliberately absent "
              "rather than estimated."),
    "settings": {"reps": reps, "per_run_timeout_s": limit, "serial": True,
                 "arch": "gfx1100", "concurrent_load": "an existing model server on port 8080 holds most VRAM"},
    "vram_used_bytes_before": vram_before,
    "vram_used_bytes_after": vram_used(),
    "results": entries,
}
json.dump(doc, open(out, "w"), indent=2)

for e in entries:
    if e.get("status") == "unavailable":
        print(f"  UNAVAILABLE  {e['target']}: {e['reason']}")
    else:
        print(f"  {e['status'].upper():<11}  {e['target']:<22} median {e['median_s']:.3f}s "
              f"spread {e['spread_s']:.3f}s over {e['reps']} reps")
print()
print("  scope:", doc["scope"])
print("  results:", out)
PY

state_record "bench" "{\"log\":\"$LOG_FILE\",\"reps\":$REPS}" >/dev/null
# A crashed harness must not read as a completed benchmark, and neither may a
# target that ran and failed: bench.json's per-target statuses decide the exit
# code (any failed -> 4; everything unavailable -> 3), so a bench of failures
# can never exit 0.
if [ "$BENCH_RC" -ne 0 ]; then
    die "bench harness failed (rc $BENCH_RC); no benchmark was recorded - see $LOG_FILE" $EX_FAIL
fi
BENCH_FAILED=$(python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
rs=d.get("results",[])
f=sum(1 for e in rs if e.get("status")=="failed")
u=sum(1 for e in rs if e.get("status")=="unavailable")
print(f"{f} {u} {len(rs)}")' "$RESULTS_DIR/bench.json" 2>/dev/null || echo "0 0 0")
set -- $BENCH_FAILED
if [ "${1:-0}" -gt 0 ]; then
    warn "$1 bench target(s) FAILED - see $RESULTS_DIR/bench.json"
    exit $EX_TESTFAIL
fi
if [ "${2:-0}" -gt 0 ] && [ "${2:-0}" -eq "${3:-0}" ]; then
    warn "every bench target is unavailable (not built) - nothing was measured"
    exit $EX_BLOCKED
fi
printf '\nbench: complete (results: %s)\n' "$RESULTS_DIR/bench.json"
exit $EX_OK
