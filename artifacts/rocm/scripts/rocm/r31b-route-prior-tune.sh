#!/bin/bash
# r31b-route-prior-tune.sh - item 2: fine-tune the live route prior and bound its quality (2026-10-18)
#
# The live arm is STRATA_ROUTE_PRIOR=0.5. This window asks whether 0.25 or 0.75 beats it, and runs
# a 20-prompt quality battery for the live arm, the winner (if any) and the off arm.
# Pre-declared: a candidate must beat 0.5 by >= 2.0% median tok/s at 1K with >= 3/5 wins (pairs 1-3
# screen, 4-5 if it passes) and no 32K regression; quality: arm pass-count >= base and no prompt
# the base passes is failed. TAIL_SKIP interaction: 2 pairs at 1K with TAIL_SKIP=0 (prior 0.5 vs 0).
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
WT=/home/zacch/projects/strata-r145; BUILD=build-0145; EXE=$WT/$BUILD/strata
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r31b-prior-tune-$STAMP.log
CFG=/tmp/serve-r31b-$STAMP.json
T1K=$ROOT/artifacts/rocm/workloads/qsa-profile-1024.tokens
T32K=$ROOT/artifacts/rocm/workloads/qsa-profile-32768.tokens
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
c["note"] = "r31b prior tune copy"; json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(memav)" -ge 66000 ] && [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "MemAvailable $(memav) MiB; VRAM $(vram) MiB"
cell() {  # cell <tag> <tok> <new> <prior> <tailskip>
    local tag="$1" tok="$2" new="$3" prior="$4" ts="$5"
    local OUT=$ROOT/artifacts/rocm/logs/r31b-cell-$tag-$STAMP.log
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=$ts STRATA_ROUTE_PRIOR=$prior STRATA_GDN_CHUNKED=1 STRATA_PF_GEMM=1 STRATA_PF_PAD=1 \
        systemd-run --user --scope -p MemoryMax=64G \
        "$EXE" --tokens-file "$tok" --max-new "$new" --max-context 139264 \
        --pack $EXP/packs/iq3_s \
        --native /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        --ple-gguf /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
        --mtp /mnt/LINUXWINSHARE/models/mtp/rt \
        --expert-profile $EXP/data/expert-profile.bin \
        --expert-cache auto --spec 5 --kv int8 --kv-resident 32768 --prefill 8192 \
        --prompt-cache 6 --spec-min-p 0.70 --vram-reserve-mib 3072 --lookup-chain 2 > "$OUT" 2>&1
    local rc=$? d acc
    d=$(grep -oE "decode +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    acc=$(grep -oE "drafts accepted [0-9]+ of [0-9]+" "$OUT" | head -1)
    echo "CELL $tag rc=$rc decode=$d $acc"
    [ "$rc" = "0" ] || echo "  (failed - $OUT)"
}
echo "== screen 0.25 / 0.75 vs live 0.5 at 1K =="
for p in 1 2 3; do
    for lam in 0.25 0.75; do cell base-$lam-p$p "$T1K" 256 0.5 7; cell l$lam-p$p "$T1K" 256 $lam 7; done
done
python3 - "$STAMP" > /tmp/r31b-best-$STAMP.txt <<'PY'
import glob, re, statistics, sys
stamp = sys.argv[1]; root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def tps(tag):
    f = glob.glob(f"{root}/r31b-cell-{tag}-{stamp}.log")
    if not f: return None
    m = re.search(r"decode +\d+ tokens in [\d.]+ ms +-> +([\d.]+) tok/s", open(f[0]).read())
    return float(m.group(1)) if m else None
best, bestmed = None, -1e9
for lam in ("0.25", "0.75"):
    ds = []
    for p in (1, 2, 3):
        b = tps(f"base-{lam}-p{p}" if lam == "0.25" else f"base-0.25-p{p}")
        b = tps(f"base-{lam}-p{p}")
        a = tps(f"l{lam}-p{p}")
        if b and a: ds.append(100 * (a - b) / b)
    med = statistics.median(ds) if ds else -1e9
    print(f"# {lam}: median {med:+.2f}% over {len(ds)} pairs")
    if med > bestmed + 0.05: best, bestmed = lam, med
print(best if best and bestmed >= 2.0 else "none")
PY
grep '^#' /tmp/r31b-best-$STAMP.txt; BEST=$(tail -1 /tmp/r31b-best-$STAMP.txt); echo "WINNER: $BEST (none = 0.5 stays)"
if [ "$BEST" != "none" ]; then
    for p in 4 5; do cell base-$BEST-p$p "$T1K" 256 0.5 7; cell l$BEST-p$p "$T1K" 256 $BEST 7; done
    for p in 1 2; do cell base-32k-$p "$T32K" 128 0.5 7; cell l$BEST-32k-$p "$T32K" 128 $BEST 7; done
fi
echo "== TAIL_SKIP interaction (TS=0, prior 0.5 vs 0) =="
for p in 1 2; do cell ts0-off-$p "$T1K" 256 0 0; cell ts0-prior-$p "$T1K" 256 0.5 0; done

echo "== 20-prompt quality battery through serve (off, live 0.5, winner) =="
serve_arm() {  # serve_arm <tag> <prior>
    local tag="$1" prior="$2"
    local logf=$ROOT/artifacts/rocm/logs/r31b-engine-$tag-$STAMP.log
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
    ("2", r"^\D*2\D*$|\b2\b", "What is the smallest prime number? Give only the number."),
    ("100", r"\b100\b", "At what temperature in Celsius does water boil at sea level? Give only the number."),
    ("shakespeare", r"shakespeare", "Who wrote Hamlet? Answer with the surname only."),
    ("au", r"\bau\b", "What is the chemical symbol for gold? Give only the symbol."),
    ("1024", r"\b1024\b", "What is 2 to the power of 10? Give only the number."),
    ("tokyo", r"\btokyo\b", "What is the capital of Japan? Answer with one word."),
    ("366", r"\b366\b", "How many days are in a leap year? Give only the number."),
    ("56", r"\b56\b", "What is 7 multiplied by 8? Give only the number."),
    ("blue", r"\bblue\b", "What colour is a clear daytime sky? Answer with one word."),
]
out = open(f"/tmp/r31b-battery-{tag}-{stamp}.jsonl", "w")
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
serve_arm off 0;   battery off;   serve_stop
serve_arm live 0.5; battery live; serve_stop
if [ "$BEST" != "none" ]; then serve_arm win "$BEST"; battery win; serve_stop; fi

echo "== verdict =="
python3 - "$STAMP" "$BEST" <<'PY'
import glob, json, re, statistics, sys
stamp, best = sys.argv[1], sys.argv[2]
root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def cell(tag):
    f = glob.glob(f"{root}/r31b-cell-{tag}-{stamp}.log")
    if not f: return None
    t = open(f[0]).read()
    d = re.search(r"decode +(\d+) tokens in ([\d.]+) ms +-> +([\d.]+) tok/s", t)
    r = re.search(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+)", t)
    if not (d and r): return None
    return dict(tps=float(d.group(3)), accepted=int(r.group(3)))
def pairs(base_tag, arm_tag, n):
    rows = []
    for i in range(1, n + 1):
        b, a = cell(base_tag.format(i)), cell(arm_tag.format(i))
        if not (b and a): return None
        rows.append((b, a))
    return rows
if best != "none":
    rows = pairs("base-" + best + "-p{}", "l" + best + "-p{}", 5)
    if rows:
        d = [100 * (a["tps"] - b["tps"]) / b["tps"] for b, a in rows]
        print(f"winner {best} vs 0.5 @1K: " + " ".join(f"{x:+.2f}%" for x in d) + f" | median {statistics.median(d):+.2f}% wins {sum(1 for x in d if x>0)}/5")
    r32 = pairs("base-32k-{}", "l" + best + "-32k-{}", 2)
    if r32:
        d = [100 * (a["tps"] - b["tps"]) / b["tps"] for b, a in r32]
        print(f"winner {best} vs 0.5 @32K: " + " ".join(f"{x:+.2f}%" for x in d) + f" | median {statistics.median(d):+.2f}%")
def batt(tag):
    return {r["key"]: r.get("pass", False) for r in [json.loads(l) for l in open(f"/tmp/r31b-battery-{tag}-{stamp}.jsonl")]}
b_off, b_live = batt("off"), batt("live")
fails_live = [k for k in b_off if b_off[k] and not b_live[k]]
print(f"quality off {sum(b_off.values())}/20 | live 0.5 {sum(b_live.values())}/20 | live regressions vs off: {fails_live or 'none'}")
if best != "none":
    b_win = batt("win")
    fails_win = [k for k in b_live if b_live[k] and not b_win[k]]
    print(f"quality winner {best} {sum(b_win.values())}/20 | vs live regressions: {fails_win or 'none'}")
r = pairs("ts0-off-{}", "ts0-prior-{}", 2)
if r:
    d = [100 * (a["tps"] - b["tps"]) / b["tps"] for b, a in r]
    print(f"TAIL_SKIP=0: prior 0.5 vs off: " + " ".join(f"{x:+.2f}%" for x in d) + f" (median {statistics.median(d):+.2f}%)")
print(f"DECISION: {'keep 0.5' if best == 'none' else ('adopt ' + best if True else '')}")
PY
echo "== done (trap restores 8081) =="
