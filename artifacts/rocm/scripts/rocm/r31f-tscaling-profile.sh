#!/bin/bash
# r31f-tscaling-profile.sh - V01 calibration: does the projection stage scale with T? (2026-10-18)
# The reviewer's decisive question: the profile's byte basis rides on the token factor. Two serve-path
# profiles on the live env, the same request, only --spec differs (2 vs 8), so avg T differs; the
# per-stage ms/window tells whether the weight-bound stages amortize (flat) or scale with rows.
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
WT=/home/zacch/projects/strata-r145; BUILD=build-0145; EXE=$WT/$BUILD/strata
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r31f-tscaling-$STAMP.log
mkdir -p "$ROOT/artifacts/rocm/logs"; exec > >(tee -a "$LOG") 2>&1
restore_8081() {
    local p; p=$(pgrep -f "serve/server.py --engine strata .*r31f" || true); [ -n "${p:-}" ] && kill $p 2>/dev/null
    sleep 2; systemctl --user reset-failed strata-8081.service 2>/dev/null
    systemctl --user start strata-8081.service 2>/dev/null
    for i in $(seq 1 45); do
        curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' \
            && { echo "RESTORE_8081 ok"; return 0; }; sleep 4
    done
    echo "RESTORE_8081 FAILED"; return 1
}
trap restore_8081 EXIT
vram() { local v; v=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo 999999999999); echo $((v / 1048576)); }
python3 - > /tmp/r31f-req.json <<'PY'
import json
para = ("The quick brown fox jumps over the lazy dog while the engineer measures the decode window "
        "and writes down every stage of the pipeline, because a number without its provenance is a rumour. ")
print(json.dumps({"model": "qwen3.8-flash-next", "max_tokens": 128, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": para * 60}]}))
PY
for SPEC in 2 8; do
    CFG=/tmp/serve-r31f-spec$SPEC-$STAMP.json
    python3 - "$EXP/serve-8081.json" "$CFG" "$EXE" "$SPEC" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["exe"] = sys.argv[3]; c["log"] = sys.argv[2] + ".engine.log"
c["note"] = "r31f T-scaling copy (spec " + sys.argv[4] + ")"
args = c["args"]
for i, a in enumerate(args):
    if a == "--spec": args[i+1] = sys.argv[4]
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
    systemctl --user stop strata-8081.service
    for i in $(seq 1 45); do [ "$(vram)" -le 2600 ] && break; sleep 4; done
    setsid env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_PRIOR=0.5 STRATA_GDN_CHUNKED=1 STRATA_PF_GEMM=1 STRATA_PF_PAD=1 \
        STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 \
        $EXP/venv/bin/python $EXP/hipify/serve/server.py --engine strata \
        --config "$CFG" --port 8081 --host 127.0.0.1 > "$CFG.out" 2>&1 &
    SRV=$!
    for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break; sleep 3; done
    curl -s -m 400 -o /tmp/r31f-resp-$SPEC.json -X POST http://127.0.0.1:8081/v1/chat/completions \
        -H "Content-Type: application/json" -d @/tmp/r31f-req.json
    sleep 2
    kill -- -$SRV 2>/dev/null
    for i in $(seq 1 30); do [ "$(vram)" -le 2600 ] && break; sleep 2; done
    p=$(pgrep -f "$BUILD/strata --serve" || true); [ -n "${p:-}" ] && kill $p 2>/dev/null
    sleep 3
    echo "===== spec $SPEC: decode timing ====="
    grep -a "strata decode timing" "$CFG.engine.log" | tail -1 | cut -c1-420
    echo "===== spec $SPEC: GPU stages ====="
    grep -a "strata decode GPU stages" "$CFG.engine.log" | tail -1 | cut -c1-1400
done
echo "== done (trap restores 8081) =="
