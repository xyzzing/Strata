#!/bin/bash
# r30j-prefill-gate.sh - quality gate for the two prefill candidates (2026-10-18)
#   GDN chunked: +5.85% prefill @32K (STRATA_GDN_CHUNKED=1, FP16 operands in the recurrence)
#   PF combo:    +4.41% prefill @128K (STRATA_PF_GEMM=1 STRATA_PF_PAD=1, FP16 activations)
# Both are output-changing and PREFILL-only; a short-prompt battery never reaches either path, so
# the gate is: (a) the GDN chunked parity test on-card, (b) an 8K-token needle through the serve
# API, A-B-A-B (base, GDN, PF, base) - the answer must be the needle number in every arm.
# ADOPTION if (a) passes and every needle answer is correct.
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
WT=/home/zacch/projects/strata-r145; BUILD=build-0145; EXE=$WT/$BUILD/strata
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r30j-prefill-gate-$STAMP.log
CFG=/tmp/serve-r30j-$STAMP.json
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
c["note"] = "r30j prefill gate copy"; json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
echo "== stop 8081 =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(vram)" -le 2600 ] && [ "$(memav)" -ge 64000 ] && break; sleep 4; done
echo "MemAvailable $(memav) MiB; VRAM $(vram) MiB"
echo "== (a) GDN chunked parity (on card) =="
cd "$WT/$BUILD" || exit 9
GATES_OK=1
for t in hip_gdn_chunked iq_parity; do
    if LD_LIBRARY_PATH=$EXP/sdk/lib ctest -R "^${t}\$" --output-on-failure > "ctest-r30j-${t}.log" 2>&1 \
       && grep -q "100% tests passed" "ctest-r30j-${t}.log" && ! grep -qi "No tests were found" "ctest-r30j-${t}.log"; then
        echo "CTEST $t: PASS"
    else echo "CTEST $t: FAIL (tail: $(tail -3 "ctest-r30j-${t}.log" | tr '\n' ' '))"; GATES_OK=0; fi
done
[ "$GATES_OK" = "1" ] || { echo "PARITY FAILED"; exit 4; }

# the 8K-token needle: deterministic filler, a number in the middle, one question
python3 - > /tmp/r30j-needle.json <<'PY'
import json
para = ("The northern shipping lanes stay open through the winter only when the icebreakers clear the "
        "channel twice a week, and the harbour master logs every transit in the seasonal record. ")
text = para * 420
mid = len(text) // 2
text = text[:mid] + " The access code for the harbour archive is 739154. " + text[mid:]
q = ("What is the access code for the harbour archive? Give only the number.\n\n" + text)
print(json.dumps({"model": "qwen3.8-flash-next", "max_tokens": 32, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": q}]}))
PY
serve_arm() {  # serve_arm <tag> [env...]
    local tag="$1"; shift
    local logf=$ROOT/artifacts/rocm/logs/r30j-engine-$tag-$STAMP.log
    : > "$CFG.engine.log"
    setsid env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_PRIOR=0.5 "$@" \
        $EXP/venv/bin/python $EXP/hipify/serve/server.py --engine strata \
        --config $CFG --port 8081 --host 127.0.0.1 > "$logf" 2>&1 &
    SRV=$!
    local up=0
    for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && { up=1; break; }; sleep 3; done
    [ "$up" = "1" ] || { echo "SERVE $tag FAILED"; return 1; }
    local t0=$(date +%s)
    curl -s -m 400 -o /tmp/r30j-needle-$tag.json -X POST http://127.0.0.1:8081/v1/chat/completions         -H "Content-Type: application/json" -d @/tmp/r30j-needle.json
    local t1=$(date +%s)
    python3 - "$tag" "$((t1 - t0))" "/tmp/r30j-needle-$tag.json" <<'PYN'
import json, sys
tag, elapsed, path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path)); c = d["choices"][0]
    content = c.get("message", {}).get("content") or ""
    ok = "739154" in content
    print(f"NEEDLE {tag}: {'PASS' if ok else 'FAIL'} ({elapsed}s) content={content[:60]!r} finish={c.get('finish_reason')}")
except Exception as e:
    print(f"NEEDLE {tag}: ERROR {e}")
PYN
    kill -- -$SRV 2>/dev/null
    for i in $(seq 1 30); do [ "$(vram)" -le 2600 ] && break; sleep 2; done
    local p; p=$(pgrep -f "$BUILD/strata --serve" || true); [ -n "${p:-}" ] && kill $p 2>/dev/null
    sleep 3
}
echo "== (b) 8K needle, A-B-A-B =="
serve_arm base1
serve_arm gdn STRATA_GDN_CHUNKED=1
serve_arm pf STRATA_PF_GEMM=1 STRATA_PF_PAD=1
serve_arm base2
echo "== done (trap restores 8081) =="
