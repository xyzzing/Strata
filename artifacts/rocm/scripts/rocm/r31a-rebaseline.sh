#!/bin/bash
# r31a-rebaseline.sh - item 1: re-baseline and re-profile decode on the live env (2026-10-18)
#
# The headline numbers in GFX1100.md predate the three adoptions (route prior 0.5, GDN chunked,
# PF_GEMM+PAD). This window measures the live config end to end and re-profiles the decode window:
#   cells @1K x3, 32K x2, 128K x2, 262K x1  (production args, LIVE env)
#   serve-path profile: STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1, one ~3K-token request
# Live env: ROUTE_PRIOR=0.5 GDN_CHUNKED=1 PF_GEMM=1 PF_PAD=1 + the previous adopted set.
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
WT=/home/zacch/projects/strata-r145; BUILD=build-0145; EXE=$WT/$BUILD/strata
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r31a-rebaseline-$STAMP.log
CFG=/tmp/serve-r31a-$STAMP.json
T1K=$ROOT/artifacts/rocm/workloads/qsa-profile-1024.tokens
T32K=$ROOT/artifacts/rocm/workloads/qsa-profile-32768.tokens
T128K=$ROOT/artifacts/rocm/workloads/qsa-profile-131072.tokens
T262K=$ROOT/artifacts/rocm/workloads/qsa-profile-261888.tokens
mkdir -p "$ROOT/artifacts/rocm/logs"; exec > >(tee -a "$LOG") 2>&1
restore_8081() {
    local p; p=$(pgrep -f "serve/server.py --engine strata .*$STAMP" || true); [ -n "${p:-}" ] && kill $p 2>/dev/null
    sleep 2; systemctl --user reset-failed strata-8081.service 2>/dev/null
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
python3 - "$EXP/serve-8081.json" "$CFG" "$EXE" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["exe"] = sys.argv[3]; c["log"] = sys.argv[2] + ".engine.log"
c["note"] = "r31a re-baseline copy"; json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
echo "== stop 8081 + drain =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(memav)" -ge 66000 ] && [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "MemAvailable $(memav) MiB (gate 66000); VRAM $(vram) MiB"
LIVE=(STRATA_ROUTE_PRIOR=0.5 STRATA_GDN_CHUNKED=1 STRATA_PF_GEMM=1 STRATA_PF_PAD=1)
cell() {  # cell <tag> <tok> <new> <maxctx>
    local tag="$1" tok="$2" new="$3" ctx="$4"
    local OUT=$ROOT/artifacts/rocm/logs/r31a-cell-$tag-$STAMP.log
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 "${LIVE[@]}" \
        systemd-run --user --scope -p MemoryMax=64G \
        "$EXE" --tokens-file "$tok" --max-new "$new" --max-context "$ctx" \
        --pack $EXP/packs/iq3_s \
        --native /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        --ple-gguf /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
        --mtp /mnt/LINUXWINSHARE/models/mtp/rt \
        --expert-profile $EXP/data/expert-profile.bin \
        --expert-cache auto --spec 5 --kv int8 --kv-resident 32768 --prefill 8192 \
        --prompt-cache 6 --spec-min-p 0.70 --vram-reserve-mib 3072 --lookup-chain 2 > "$OUT" 2>&1
    local rc=$? p d sel acc cpu pcie
    p=$(grep -oE "prefill +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    sel=$(grep -oE "qsa select [0-9]+" "$OUT" | head -1)
    d=$(grep -oE "decode +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    acc=$(grep -oE "drafts accepted [0-9]+ of [0-9]+" "$OUT" | head -1)
    cpu=$(grep -oE "CPU experts [0-9.]+ distinct / [0-9.]+ routed" "$OUT" | head -1)
    pcie=$(grep -oE "pcie experts +[0-9.]+ distinct" "$OUT" | head -1)
    echo "CELL $tag rc=$rc prefill=$p ${sel:+| $sel} decode=$d $acc | $cpu | $pcie"
    [ "$rc" = "0" ] || echo "  (failed - $OUT)"
}
echo "== cells, live env =="
for i in 1 2 3; do cell 1k-$i "$T1K" 256 139264; done
for i in 1 2; do cell 32k-$i "$T32K" 128 139264; cell 128k-$i "$T128K" 64 139264; done
cell 262k-1 "$T262K" 64 262144

echo "== serve-path profile (live env + timing) =="
python3 - > /tmp/r31a-req.json <<'PY'
import json
para = ("The quick brown fox jumps over the lazy dog while the engineer measures the decode window "
        "and writes down every stage of the pipeline, because a number without its provenance is a rumour. ")
text = para * 60
print(json.dumps({"model": "qwen3.8-flash-next", "max_tokens": 128, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": text}]}))
PY
setsid env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
    STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
    STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
    STRATA_ROUTE_TAIL_SKIP=7 "${LIVE[@]}" STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 \
    $EXP/venv/bin/python $EXP/hipify/serve/server.py --engine strata \
    --config $CFG --port 8081 --host 127.0.0.1 >> "$CFG.engine.log" 2>&1 &
SRV=$!
for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break; sleep 3; done
curl -s -m 300 -o /tmp/r31a-resp.json -X POST http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/r31a-req.json
sleep 2
kill -- -$SRV 2>/dev/null
for i in $(seq 1 30); do [ "$(vram)" -le 2600 ] && break; sleep 2; done
p=$(pgrep -f "$BUILD/strata --serve" || true); [ -n "${p:-}" ] && kill $p 2>/dev/null
grep -a "strata decode timing" "$CFG.engine.log" | tail -1 | cut -c1-400
grep -a "strata decode GPU stages" "$CFG.engine.log" | tail -1 | cut -c1-1200
echo "== done (trap restores 8081) =="
