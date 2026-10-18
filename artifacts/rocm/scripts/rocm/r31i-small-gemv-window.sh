#!/bin/bash
# r31i-small-gemv-window.sh - the runner-up's decision window (2026-10-18)
# "One window of small gemv": the only separate small GEMVs left after the GDN fusion are the shared
# expert's gate/up (and the k/v, router ones are already batched). This window measures (a) the live
# stage split on the reverted binary (one 1K/256 profile) and (b) the isolated cost of the shared
# shapes at the live T via q8_gemv_bench, so the batch's ceiling is computed, not guessed.
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
WT=/home/zacch/projects/strata-r145; BUILD=build-0145; EXE=$WT/$BUILD/strata; BENCH=$WT/$BUILD/q8_gemv_bench
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r31i-small-gemv-$STAMP.log
T1K=$ROOT/artifacts/rocm/workloads/qsa-profile-1024.tokens
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
for i in $(seq 1 45); do [ "$(memav)" -ge 66000 ] && [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "MemAvailable $(memav) MiB; VRAM $(vram) MiB"
echo "== (a) live profile, 1K/256 (reverted binary) =="
env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
    STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
    STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
    STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_PRIOR=0.5 STRATA_GDN_CHUNKED=1 STRATA_PF_GEMM=1 STRATA_PF_PAD=1 \
    STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 \
    systemd-run --user --scope -p MemoryMax=64G \
    "$EXE" --tokens-file "$T1K" --max-new 256 --max-context 139264 \
    --pack $EXP/packs/iq3_s \
    --native /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --ple-gguf /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
    --mtp /mnt/LINUXWINSHARE/models/mtp/rt --expert-profile $EXP/data/expert-profile.bin \
    --expert-cache auto --spec 5 --kv int8 --kv-resident 32768 --prefill 8192 \
    --prompt-cache 6 --spec-min-p 0.70 --vram-reserve-mib 3072 --lookup-chain 2 \
    > $ROOT/artifacts/rocm/logs/r31i-profile-$STAMP.log 2>&1
grep -a "strata decode timing" $ROOT/artifacts/rocm/logs/r31i-profile-$STAMP.log | tail -1 | cut -c1-300
grep -a "strata decode GPU stages" $ROOT/artifacts/rocm/logs/r31i-profile-$STAMP.log | tail -1 | cut -c1-1100
echo "== (b) isolated shared/kv shapes at cols 1,3,4 =="
env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib timeout 300 "$BENCH" --cols 1,3,4 --iters 100 --bw 750 2>&1 | grep -A6 -E "shared gate/up|shared down|QSA k/v|HC up"
echo "== (c) bound =="
python3 - "$STAMP" <<'PY'
import glob, re, sys
stamp = sys.argv[1]; root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
prof = glob.glob(f"{root}/r31i-profile-{stamp}.log")
t = open(prof[0]).read() if prof else ""
# the small, non-quantization stages that are separate launches (gate row, k/v) and the shared block
def stage(name):
    m = re.search(re.escape(name) + r" ([\d.]+)", t)
    return float(m.group(1)) if m else None
print("shared+quant (in-situ):", stage("shared+quant"), "ms/window  (GDN half only shown; QSA half in the same line)")
bench = glob.glob(f"{root}/r31i-small-gemv-{stamp}.log")
print("bench rows above are per-call; a gate/up batch could at best halve their fixed launch cost,")
print("i.e. save ~1 launch per layer ~= 1.5-2 us x 48 = 0.07-0.10 ms/window = 0.2-0.3% of the 35 ms round.")
PY
echo "== done (trap restores 8081) =="
