#!/bin/bash
# r31e-gemv-bench-window.sh - V01's missing measurement, done properly in a window (2026-10-18)
# The accident: `q8_gemv_bench --help` ran the bench against the live card (production healthy after,
# disclosed in the campaign log). This window stops 8081, runs the real native_mmvq Q8_0 projection
# bench at the engine's shapes across cols 1..6, and restores. No production change.
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
BENCH=/home/zacch/projects/strata-r145/build-0145/q8_gemv_bench
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r31e-gemv-bench-$STAMP.log
mkdir -p "$ROOT/artifacts/rocm/logs"; exec > >(tee -a "$LOG") 2>&1
restore_8081() {
    systemctl --user reset-failed strata-8081.service 2>/dev/null
    systemctl --user start strata-8081.service 2>/dev/null
    for i in $(seq 1 45); do
        curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' \
            && { echo "RESTORE_8081 ok"; return 0; }; sleep 4
    done
    echo "RESTORE_8081 FAILED"; return 1
}
trap restore_8081 EXIT
vram() { local v; v=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo 999999999999); echo $((v / 1048576)); }
echo "== stop 8081 =="; systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "VRAM $(vram) MiB"
echo "== q8_gemv_bench (native_mmvq, engine shapes, cols 1,2,3,4,6,8) =="
env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib timeout 900 \
    "$BENCH" --cols 1,2,3,4,6,8 --iters 100 --bw 750
echo "== done (trap restores 8081) =="
