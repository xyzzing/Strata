#!/bin/bash
# r31j-hc-up-bench.sh - the HC up stop/go discriminator (2026-10-18)
# The engine's up call reads bf16 10240x320 = 6.55 MB at ~15.6 us (420 GB/s). This window runs the bench's
# grid/ILP variants on that exact read; >=650 GB/s on the best variant means an engine change is worth it.
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
BENCH=/home/zacch/projects/strata-r145/build-0145/hc_up_bench
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r31j-hc-up-bench-$STAMP.log
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
memav() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
vram() { local v; v=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo 999999999999); echo $((v / 1048576)); }
echo "== preflight ($STAMP) =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(memav)" -ge 66000 ] && [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "MemAvailable $(memav) MiB; VRAM $(vram) MiB"
env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib timeout 600 "$BENCH" all 300
echo "== done (trap restores 8081) =="
