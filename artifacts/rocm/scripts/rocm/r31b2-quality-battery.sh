#!/bin/bash
# r31b2-quality-battery.sh - item 2b: 20-prompt quality battery, prior off vs live 0.5
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
WT=/home/zacch/projects/strata-r145; BUILD=build-0145; EXE=$WT/$BUILD/strata
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r31b2-battery-$STAMP.log
CFG=/tmp/serve-r31b2-$STAMP.json
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
vram() { local v; v=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo 999999999999); echo $((v / 1048576)); }
python3 - "$EXP/serve-8081.json" "$CFG" "$EXE" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["exe"] = sys.argv[3]; c["log"] = sys.argv[2] + ".engine.log"
c["note"] = "r31b2 battery copy"; json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(vram)" -le 2600 ] && break; sleep 4; done
serve_arm() {  # serve_arm <tag> <prior>
    local tag="$1"
    local prior="$2"
    local logf=$ROOT/artifacts/rocm/logs/r31b2-engine-$tag-$STAMP.log
    : > "$CFG.engine.log"
    setsid env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_PRIOR=$prior STRATA_GDN_CHUNKED=1 STRATA_PF_GEMM=1 STRATA_PF_PAD=1 \
        $EXP/venv/bin/python $EXP/hipify/serve/server.py --engine strata \
        --config $CFG --port 8081 --host 127.0.0.1 > "$logf" 2>&1 &
    SRV=$!
    for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break; sleep 3; done
    curl -s -m 180 http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" \
        -d '{"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"warm"}]}' > /dev/null
    echo "SERVE $tag up"
}
serve_stop() {
    kill -- -$SRV 2>/dev/null
    for i in $(seq 1 30); do [ "$(vram)" -le 2600 ] && break; sleep 2; done
    local p; p=$(pgrep -f "$BUILD/strata --serve" || true); [ -n "${p:-}" ] && kill $p 2>/dev/null
    sleep 3
}
battery() {  # battery <tag>
    local tag="$1"
    python3 - "$tag" "$STAMP" <<'PY'
import json, re, urllib.request, sys
tag, stamp = sys.argv[1], sys.argv[2]
prompts = [
    ("paris", r"\bparis\b", "Name the capital of France in one word."),
    ("391", r"\b391\b", "What is 17 multiplied by 23? Give only the number."),
    ("jupiter", r"\bjupiter\b", "Which planet is the largest in our solar system? Answer with one word."),
    ("1989", r"\b1989\b", "In what year did the Berlin Wall fall? Give only the year."),
    ("17", r"\b17\b", "Continue the sequence of primes: 2, 3, 5, 7, 11, 13. Give only the next prime."),
    ("iron", r"\biron\b", "Which chemical element has the symbol Fe? Answer with one word."),
    ("h2o", r"h\s*2\s*o|h\u2082o", "What is the chemical formula of water? Give only the formula."),
    ("7", r"\bseven\b|\b7\b", "How many continents are there on Earth? Give only the number."),
    ("144", r"\b144\b", "What is 12 multiplied by 12? Give only the number."),
    ("63", r"\b63\b", "What is 100 minus 37? Give only the number."),
    ("pacific", r"\bpacific\b", "Which ocean is the largest on Earth? Answer with one word."),
    ("2", r"\b2\b", "What is the smallest prime number? Give only the number."),
    ("100", r"\b100\b", "At what temperature in Celsius does water boil at sea level? Give only the number."),
    ("shakespeare", r"shakespeare", "Who wrote Hamlet? Answer with the surname only."),
    ("au", r"\bau\b", "What is the chemical symbol for gold? Give only the symbol."),
    ("1024", r"\b1024\b", "What is 2 to the power of 10? Give only the number."),
    ("tokyo", r"\btokyo\b", "What is the capital of Japan? Answer with one word."),
    ("366", r"\b366\b", "How many days are in a leap year? Give only the number."),
    ("56", r"\b56\b", "What is 7 multiplied by 8? Give only the number."),
    ("blue", r"\bblue\b", "What colour is a clear daytime sky? Answer with one word."),
]
out = open(f"/tmp/r31b2-battery-{tag}-{stamp}.jsonl", "w")
for key, pat, q in prompts:
    req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",
        data=json.dumps({"max_tokens": 64, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False},
                         "messages": [{"role": "user", "content": q}]}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=180)); c = d["choices"][0]
        content = (c["message"]["content"] or "")
        rec = {"key": key, "pass": re.search(pat, content[:80].lower()) is not None, "content": content[:120]}
    except Exception as e:
        rec = {"key": key, "pass": False, "error": str(e)[:120]}
    out.write(json.dumps(rec) + "\n"); out.flush()
out.close(); print(f"battery {tag} done")
PY
}
serve_arm off 0;    battery off;    serve_stop
serve_arm live 0.5; battery live;   serve_stop
python3 - "$STAMP" <<'PY'
import json, sys
stamp = sys.argv[1]
def b(tag): return {r["key"]: r.get("pass", False) for r in [json.loads(l) for l in open(f"/tmp/r31b2-battery-{tag}-{stamp}.jsonl")]}
o, l = b("off"), b("live")
fails = [k for k in o if o[k] and not l[k]]
gains = [k for k in o if l[k] and not o[k]]
print(f"20-prompt battery: off {sum(o.values())}/20 | live 0.5 {sum(l.values())}/20")
print(f"live fails prompts off passes: {fails or 'none'} | live gains prompts off fails: {gains or 'none'}")
PY
echo "== done (trap restores 8081) =="
