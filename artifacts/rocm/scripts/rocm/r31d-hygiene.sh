#!/bin/bash
# r31d-hygiene.sh - item 4: close the hygiene gaps (2026-10-18)
#   (a) spec-off: is it possible on a native IQ pack? two cells at 1K: --spec 0, and no --spec flag
#   (b) >LLC working set: native_expert_bench G=64 T=1 (141 MB of expert blobs, past the 96 MB
#       Infinity Cache) against G=8 T=1 (18 MB), same layers/formats
#   (c) the graph-replay item is covered by the engine windows themselves (the verify windows are
#       captured graphs and EXPDEC ran inside them in window A) - recorded, not re-run
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
WT=/home/zacch/projects/strata-r145; BUILD=build-0145; EXE=$WT/$BUILD/strata
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r31d-hygiene-$STAMP.log
T1K=$ROOT/artifacts/rocm/workloads/qsa-profile-1024.tokens
SHARD=/home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
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
echo "== preflight ($STAMP) =="; "$EXE" --version
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(memav)" -ge 64000 ] && [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "MemAvailable $(memav) MiB; VRAM $(vram) MiB"
echo "== (b) >LLC bench (runs first, small) =="
BENCH=$WT/$BUILD/native_expert_bench
for G in 8 64; do
    OUT=$ROOT/artifacts/rocm/logs/r31d-bench-G$G-$STAMP.log
    timeout 300 $BENCH "$SHARD" 0,1,2,8,12,17,21,47 "$G" 1 -1 -1 100 > "$OUT" 2>&1
    echo "BENCH G=$G:"; grep -oE "all +[0-9.]+ us +gu +[0-9.]+ us \([^)]*\) +down +[0-9.]+ us" "$OUT" | head -2
    grep -oE "[0-9]+ differing" "$OUT" | head -1
done
cell_spec() {  # cell_spec <tag> [mode]
    local tag="$1" mode="${2:-0}"
    local OUT=$ROOT/artifacts/rocm/logs/r31d-cell-$tag-$STAMP.log
    local sargs=()
    if [ "$mode" = "0" ]; then sargs=(--spec 0); fi
    if [ "$mode" = "live" ]; then sargs=(--spec 5 --spec-min-p 0.70 --lookup-chain 2); fi
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_PRIOR=0.5 STRATA_GDN_CHUNKED=1 STRATA_PF_GEMM=1 STRATA_PF_PAD=1 \
        systemd-run --user --scope -p MemoryMax=64G \
        "$EXE" --tokens-file "$T1K" --max-new 128 --max-context 139264 \
        --pack $EXP/packs/iq3_s \
        --native /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        --ple-gguf /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
        --mtp /mnt/LINUXWINSHARE/models/mtp/rt \
        --expert-profile $EXP/data/expert-profile.bin \
        --expert-cache auto --kv int8 --kv-resident 32768 --prefill 8192 \
        --prompt-cache 6 --vram-reserve-mib 3072 "${sargs[@]}" > "$OUT" 2>&1
    local rc=$? d
    d=$(grep -oE "decode +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    echo "SPEC $tag rc=$rc decode=$d $(grep -m1 -iE 'spec|refus|error' "$OUT" | cut -c1-120)"
}
echo "== (a) spec-off =="
cell_spec live live
cell_spec off0 0
cell_spec offdefault default
echo "== done (trap restores 8081) =="
