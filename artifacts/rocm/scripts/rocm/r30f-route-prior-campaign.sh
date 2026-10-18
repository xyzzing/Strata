#!/bin/bash
# r30f-route-prior-campaign.sh - item 2: STRATA_ROUTE_PRIOR=lambda on gfx1100 (2026-10-18)
#
# Cache-conditioned routing: a resident expert gets a lambda logit bonus when the router picks the
# top-10 (weights stay the unbiased softmax renormalised over the picks, route_prior.cu). It is
# OUTPUT-CHANGING, so the quality battery runs before any rate counts.
#
# PRE-DECLARED:
#   Screen: lambda in {0.5, 1.0, 2.0} vs off, 3 interleaved pairs @1K each; the best median becomes
#           the arm (ties -> the smaller lambda: less output change).
#   GATE A (quality, for the chosen arm): battery A-B-A via serve; PASS = arm >= base pass count and
#           no prompt base passes that the arm fails. (The prose-drift check is recorded but is
#           noise-dominated across restarts, as measured in the DEFER_CPU window.)
#   GATE B (adoption): @1K 5 pairs - median >= +2.0% tok/s and >= 4/5 wins; @32K 2 pairs and
#           @128K 2 pairs - no ms/round below -2% and no tok/s median below -1%.
#   ADOPTION (both): STRATA_ROUTE_PRIOR=<best> joins the serve-8081.json env (output-change note,
#           unset restores); restart + Paris smoke. Otherwise production untouched.
set -u
WT=/home/zacch/projects/strata-r145
BUILD=build-0145
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r30f-route-prior-$STAMP.log
CFG=/tmp/serve-r30f-$STAMP.json
T1K=$ROOT/artifacts/rocm/workloads/qsa-profile-1024.tokens
T32K=$ROOT/artifacts/rocm/workloads/qsa-profile-32768.tokens
T128K=$ROOT/artifacts/rocm/workloads/qsa-profile-131072.tokens
mkdir -p "$ROOT/artifacts/rocm/logs"
exec > >(tee -a "$LOG") 2>&1

restore_8081() {
    local p; p=$(pgrep -f "serve/server.py --engine strata .*$STAMP" || true); [ -n "${p:-}" ] && kill $p 2>/dev/null
    sleep 2
    systemctl --user reset-failed strata-8081.service 2>/dev/null
    systemctl --user start strata-8081.service 2>/dev/null
    for i in $(seq 1 45); do
        curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' \
            && { echo "RESTORE_8081 ok (after ~$((i*4))s)"; return 0; }
        sleep 4
    done
    echo "RESTORE_8081 FAILED after 180s - run $EXP/start-8081.sh"; return 1
}
trap restore_8081 EXIT
memav() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
vram() { local v; v=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo 999999999999); echo $((v / 1048576)); }

EXE=$WT/$BUILD/strata
echo "== 0. preflight ($STAMP) =="
[ -x "$EXE" ] || { echo "FATAL: binary missing"; exit 9; }
"$EXE" --version
python3 - "$EXP/serve-8081.json" "$CFG" "$EXE" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["exe"] = sys.argv[3]; c["log"] = sys.argv[2] + ".engine.log"
c["note"] = "r30f ROUTE_PRIOR campaign copy"
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY

echo "== 1. stop 8081 + drain =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "8081: $(systemctl --user is-active strata-8081.service); MemAvailable $(memav) MiB; VRAM $(vram) MiB"

cell() {  # cell <tag> <tok> <new> <lambda|0>
    local tag="$1" tok="$2" new="$3" lam="$4"
    local OUT=$ROOT/artifacts/rocm/logs/r30f-cell-$tag-$STAMP.log
    local envs=()
    [ "$lam" != "0" ] && envs=(STRATA_ROUTE_PRIOR=$lam)
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 "${envs[@]}" \
        systemd-run --user --scope -p MemoryMax=66G \
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

echo "== 2. screen lambda at 1K (3 pairs each) =="
for p in 1 2 3; do
    for lam in 0.5 1.0 2.0; do
        cell l$lam-1k-$p "$T1K" 256 $lam
        cell base-1k-$p "$T1K" 256 0
    done
done

echo "== 3. best lambda by median, then 32K/128K cells =="
python3 - "$STAMP" > /tmp/r30f-best-$STAMP.txt <<'PY'
import glob, re, statistics, sys
stamp = sys.argv[1]; root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def tps(tag):
    f = glob.glob(f"{root}/r30f-cell-{tag}-{stamp}.log")
    if not f: return None
    m = re.search(r"decode +\d+ tokens in [\d.]+ ms +-> +([\d.]+) tok/s", open(f[0]).read())
    return float(m.group(1)) if m else None
best, bestmed = None, -1e9
for lam in ("0.5", "1.0", "2.0"):
    ds = []
    for p in (1, 2, 3):
        b, a = tps(f"base-1k-{p}"), tps(f"l{lam}-1k-{p}")
        if b and a: ds.append(100 * (a - b) / b)
    med = statistics.median(ds) if ds else -1e9
    print(f"# lambda {lam}: median {med:+.2f}% over {len(ds)} pairs")
    if med > bestmed + 0.05: best, bestmed = lam, med
print(best if best else "1.0")
PY
grep '^#' /tmp/r30f-best-$STAMP.txt; BEST=$(tail -1 /tmp/r30f-best-$STAMP.txt); echo "BEST lambda: $BEST"
# the gate wants 5 pairs at 1K; pairs 1-3 are the screen, add 4-5 for the chosen arm
for p in 4 5; do cell base-1k-$p "$T1K" 256 0; cell l$BEST-1k-$p "$T1K" 256 "$BEST"; done
for p in 1 2; do
    cell b32k-$p "$T32K" 128 0; cell l32k-$p "$T32K" 128 "$BEST"
    cell b128k-$p "$T128K" 64 0; cell l128k-$p "$T128K" 64 "$BEST"
done
echo "== 3b. hit-rate diagnostic (STRATA_DECODE_TIMING, base vs best) =="
for arm in 0 "$BEST"; do
    tag=$([ "$arm" = "0" ] && echo dt-base || echo dt-arm)
    OUT=$ROOT/artifacts/rocm/logs/r30f-cell-$tag-$STAMP.log
    envs=(); [ "$arm" != "0" ] && envs=(STRATA_ROUTE_PRIOR=$arm)
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_DECODE_TIMING=1 "${envs[@]}" \
        systemd-run --user --scope -p MemoryMax=66G \
        "$EXE" --tokens-file "$T1K" --max-new 128 --max-context 139264 \
        --pack $EXP/packs/iq3_s \
        --native /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        --ple-gguf /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
        --mtp /mnt/LINUXWINSHARE/models/mtp/rt \
        --expert-profile $EXP/data/expert-profile.bin \
        --expert-cache auto --spec 5 --kv int8 --kv-resident 32768 --prefill 8192 \
        --prompt-cache 6 --spec-min-p 0.70 --vram-reserve-mib 3072 --lookup-chain 2 > "$OUT" 2>&1
    echo "DIAG $tag:"; grep -a "strata decode timing" "$OUT" | tail -1 | cut -c1-320
done

echo "== 4. GATE A: battery A-B-A through serve for lambda=$BEST =="
serve_start() {  # serve_start <mode: base|arm> <tag>
    local mode="$1" tag="$2" logf=$ROOT/artifacts/rocm/logs/r30f-engine-$tag-$STAMP.log
    local envs=(STRATA_ROUTE_PRIOR=0)
    [ "$mode" = "arm" ] && envs=(STRATA_ROUTE_PRIOR=$BEST)
    : > "$CFG.engine.log"
    setsid env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 "${envs[@]}" \
        $EXP/venv/bin/python $EXP/hipify/serve/server.py --engine strata \
        --config $CFG --port 8081 --host 127.0.0.1 > "$logf" 2>&1 &
    SRV=$!
    for i in $(seq 1 60); do
        curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break
        sleep 3
    done
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
battery() {
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
    ("h2o", r"h\s*2\s*o", "What is the chemical formula of water? Give only the formula."),
    ("7", r"\bseven\b|\b7\b", "How many continents are there on Earth? Give only the number."),
]
out = open(f"/tmp/r30f-battery-{tag}-{stamp}.jsonl", "w")
for key, pat, q in prompts:
    req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",
        data=json.dumps({"max_tokens": 64, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False},
                         "messages": [{"role": "user", "content": q}]}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=180))
        c = d["choices"][0]; content = (c["message"]["content"] or "")
        rec = {"key": key, "pass": re.search(pat, content[:80].lower()) is not None, "content": content[:200]}
    except Exception as e:
        rec = {"key": key, "pass": False, "error": str(e)[:200]}
    out.write(json.dumps(rec) + "\n"); out.flush()
out.close(); print(f"battery {tag} done")
PY
}
serve_start base A1; battery A1; serve_stop
serve_start arm  A2; battery A2; serve_stop
serve_start base A3; battery A3; serve_stop

echo "== 5. verdict =="
python3 - "$STAMP" "$BEST" <<'PY'
import glob, json, re, statistics, sys
stamp, best = sys.argv[1], sys.argv[2]
root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def battery(tag):
    rows = [json.loads(l) for l in open(f"/tmp/r30f-battery-{tag}-{stamp}.jsonl")]
    return {r["key"]: r.get("pass", False) for r in rows}
b1, b2, b3 = battery("A1"), battery("A2"), battery("A3")
fails = [k for k in b1 if b1[k] and not b2[k]]
a_ok = sum(b2.values()) >= sum(b1.values()) and not fails
print(f"GATE A: base {sum(b1.values())}/8 | arm(lambda {best}) {sum(b2.values())}/8 | base-again {sum(b3.values())}/8 | regressions: {fails or 'none'} -> {'PASS' if a_ok else 'FAIL'}")
def cell(tag):
    f = glob.glob(f"{root}/r30f-cell-{tag}-{stamp}.log")
    if not f: return None
    t = open(f[0]).read()
    d = re.search(r"decode +(\d+) tokens in ([\d.]+) ms +-> +([\d.]+) tok/s", t)
    r = re.search(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+)", t)
    if not (d and r): return None
    return dict(tps=float(d.group(3)), ms=float(d.group(2)), rounds=int(r.group(1)), accepted=int(r.group(3)))
def report(label, base_tags, arm_tags, need_med, need_wins):
    rows = []
    for bt, at in zip(base_tags, arm_tags):
        b, a = cell(bt), cell(at)
        if not (b and a): print(f"GATE B {label}: INCOMPLETE {bt}/{at}"); return False
        rows.append((b, a))
    d_tps = [100 * (a["tps"] - b["tps"]) / b["tps"] for b, a in rows]
    d_msr = [100 * ((a["ms"] / a["rounds"]) - (b["ms"] / b["rounds"])) / (b["ms"] / b["rounds"]) for b, a in rows]
    med = statistics.median(d_tps); wins = sum(1 for x in d_tps if x > 0)
    ok = med >= need_med and wins >= need_wins and all(x < 2.0 for x in d_msr)
    print(f"GATE B {label}: base {statistics.median([b['tps'] for b, _ in rows]):.2f} -> arm "
          f"{statistics.median([a['tps'] for _, a in rows]):.2f} | d_tps " + " ".join(f"{x:+.2f}%" for x in d_tps)
          + f" med {med:+.2f}% wins {wins}/{len(rows)} | ms/round " + " ".join(f"{x:+.2f}%" for x in d_msr)
          + f" -> {'PASS' if ok else 'FAIL'}")
    return ok
# the 5-pair 1K gate needs 5 base/arm cells: screen pairs 1-3 + 2 more? the screen ran 3 pairs; use them
b1k = report("1k", [f"base-1k-{i}" for i in (1, 2, 3, 4, 5)], [f"l{best}-1k-{i}" for i in (1, 2, 3, 4, 5)], 2.0, 4)
b32 = report("32k", ["b32k-1", "b32k-2"], ["l32k-1", "l32k-2"], 0.0, 0)
b128 = report("128k", ["b128k-1", "b128k-2"], ["l128k-1", "l128k-2"], -1.0, 0)
verdict = "ADOPT" if (a_ok and b1k and b32 and b128) else "NOT ADOPTED"
print(f"WINDOW VERDICT: {verdict} (lambda {best}; quality {a_ok})")
PY
echo "== done (trap restores 8081) =="
