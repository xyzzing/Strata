#!/bin/bash
# r30d-defer-cpu-window.sh - STRATA_VERIFY_DEFER_CPU (#2690) gate on gfx1100 (2026-10-18).
#
# Arm binary: strata-r146/build-0146 (refresh-0145 + #2514 + the #2690 merge). The feature is
# output-changing and opt-in; the default path is untouched. PRE-DECLARED GATES (declared here,
# before any number counts):
#   GATE A (quality, must pass before rates count):
#     A1 task battery A-B-A via the serve API (8 objective prompts, greedy, thinking off):
#        PASS = arm pass-count >= base pass-count AND no prompt base passes that the arm fails.
#     A2 drift: prose continuation A-B-A across restarts; arm-vs-base divergence is read against
#        the base-vs-base restart divergence (the recorded cross-restart greedy drift). Arm
#        divergence far above the restart baseline = manual review, not auto-adoption.
#     KL caveat (declared, same as R25): --dump-logits does not fire on native packs; the gate is
#        the battery + drift + the PR's own KL table (IQ3_S: 0.088 mean, the ROUTE_TAIL_SKIP scale).
#   GATE B (adoption): paired one-shot decode on build-0146, alternating arms:
#     @1K: 5 pairs - median >= +2.0% tok/s and >= 4/5 wins
#     @32K: 2 pairs - median >= 0%
#     @128K: 2 pairs - median >= -1% and no pair below -2%
#   ADOPTION (both): serve-8081.json exe -> build-0146 + env STRATA_VERIFY_DEFER_CPU=1, .bak kept,
#     restart + Paris smoke. Otherwise production stays on build-0145 untouched.
# 8080 is never touched. The trap always restarts strata-8081 from the unchanged config.
set -u
WT=/home/zacch/projects/strata-r146
BUILD=build-0146
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r30d-defer-window-$STAMP.log
CFG=/tmp/serve-r146-$STAMP.json
mkdir -p "$ROOT/artifacts/rocm/logs"
exec > >(tee -a "$LOG") 2>&1

restore_8081() {
    # kill any manual server still bound to 8081
    local p
    p=$(pgrep -f "serve/server.py --engine strata .*$STAMP" || true)
    [ -n "${p:-}" ] && kill $p 2>/dev/null
    sleep 2
    systemctl --user reset-failed strata-8081.service 2>/dev/null
    systemctl --user start strata-8081.service 2>/dev/null
    for i in $(seq 1 45); do
        if curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"'; then
            echo "RESTORE_8081 ok (after ~$((i*4))s)"; return 0
        fi
        sleep 4
    done
    echo "RESTORE_8081 FAILED after 180s - run $EXP/start-8081.sh"; return 1
}
trap restore_8081 EXIT
memav() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
vram() { local v; v=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo 999999999999); echo $((v / 1048576)); }

EXE=$WT/$BUILD/strata
T1K=$ROOT/artifacts/rocm/workloads/qsa-profile-1024.tokens
T32K=$ROOT/artifacts/rocm/workloads/qsa-profile-32768.tokens
T128K=$ROOT/artifacts/rocm/workloads/qsa-profile-131072.tokens

echo "== 0. preflight ($STAMP) =="
[ -x "$EXE" ] || { echo "FATAL: build-0146 binary missing"; exit 9; }
"$EXE" --version
sha256sum "$EXE"

# the arm config: the production config with the exe and log swapped, never the live file
python3 - "$EXP/serve-8081.json" "$CFG" "$EXE" <<'PY'
import json, sys
src, dst, exe = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(src))
c["exe"] = exe
c["log"] = dst + ".engine.log"
c["note"] = "campaign r30d (STRATA_VERIFY_DEFER_CPU gate) - a copy; production is the unit's config"
json.dump(c, open(dst, "w"), indent=1)
print("arm config:", dst)
PY

echo "== 1. stop 8081 + drain =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "8081: $(systemctl --user is-active strata-8081.service); VRAM $(vram) MiB"

echo "== 2. strict parity on the ported binary (default path) =="
cd "$WT/$BUILD" || exit 9
GATES_OK=1
for t in iq_parity native_grouped_parity iq_multi_parity qsa_select_wmma_parity; do
    if LD_LIBRARY_PATH=$EXP/sdk/lib ctest -R "^${t}\$" --output-on-failure > "ctest-r30d-${t}.log" 2>&1 \
       && grep -q "100% tests passed" "ctest-r30d-${t}.log" && ! grep -qi "No tests were found" "ctest-r30d-${t}.log"; then
        echo "CTEST $t: PASS"
    else
        echo "CTEST $t: FAIL (tail: $(tail -3 "ctest-r30d-${t}.log" | tr '\n' ' ' | head -c 240))"; GATES_OK=0
    fi
done

serve_start() {  # serve_start <defer: on|off|check> <tag>
    local mode="$1" tag="$2"
    local envs=(STRATA_VERIFY_DEFER_CPU=0)
    if [ "$mode" = "on" ]; then envs=(STRATA_VERIFY_DEFER_CPU=1); fi
    if [ "$mode" = "check" ]; then envs=(STRATA_VERIFY_DEFER_CPU=check); fi
    local logf=$ROOT/artifacts/rocm/logs/r30d-engine-$tag-$STAMP.log
    : > "$CFG.engine.log"
    setsid env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 "${envs[@]}" \
        $EXP/venv/bin/python $EXP/hipify/serve/server.py --engine strata \
        --config $CFG --port 8081 --host 127.0.0.1 > "$logf" 2>&1 &
    SRV=$!
    local up=0
    for i in $(seq 1 60); do
        curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && { up=1; break; }
        sleep 3
    done
    if [ "$up" = "1" ]; then echo "SERVE $tag: up"; else echo "SERVE $tag: FAILED"; return 1; fi
    curl -s -m 180 http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" \
        -d '{"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"warm"}]}' > /dev/null
    grep -m2 "STRATA_VERIFY_DEFER_CPU" "$CFG.engine.log" 2>/dev/null | sed 's/^/  /'
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
    ("h2o", r"h\s*2\s*o", "What is the chemical formula of water? Give only the formula."),
    ("7", r"\bseven\b|\b7\b", "How many continents are there on Earth? Give only the number."),
]
out = open(f"/tmp/r30d-battery-{tag}-{stamp}.jsonl", "w")
for key, pat, q in prompts:
    req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",
        data=json.dumps({"max_tokens": 64, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False},
                         "messages": [{"role": "user", "content": q}]}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=180))
        c = d["choices"][0]
        content = (c["message"]["content"] or "")
        rec = {"key": key, "pass": re.search(pat, content[:80].lower()) is not None,
               "content": content[:200], "finish": c["finish_reason"]}
    except Exception as e:
        rec = {"key": key, "pass": False, "error": str(e)[:200]}
    out.write(json.dumps(rec) + "\n"); out.flush()
out.close()
print(f"battery {tag} done")
PY
}
prose() {  # prose <tag>
    local tag="$1"
    python3 - "$tag" "$STAMP" <<'PY'
import json, urllib.request, sys
tag, stamp = sys.argv[1], sys.argv[2]
para = ("The striped hummingbird migrates across the coastal range each autumn, "
        "following the flowering eucalyptus groves inland. Researchers tag a few hundred birds "
        "each season; the recapture rate has held near eleven percent for a decade. ")
req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",
    data=json.dumps({"max_tokens": 256, "temperature": 0,
                     "chat_template_kwargs": {"enable_thinking": False},
                     "messages": [{"role": "user", "content": para * 8 + "\nContinue this passage with one more paragraph."}]}).encode(),
    headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=300))
c = d["choices"][0]
open(f"/tmp/r30d-prose-{tag}-{stamp}.txt", "w").write(c["message"]["content"] or "")
print(f"prose {tag} done ({len(c['message']['content'] or '')} chars)")
PY
}

echo "== 3. GATE A: A-B-A through serve (base, defer, base-again) =="
serve_start off  A1
battery A1; prose A1; serve_stop
serve_start on   A2
battery A2; prose A2; serve_stop
serve_start off  A3
battery A3; prose A3; serve_stop

echo "== 4. GATE B: paired one-shot cells on build-0146 =="
cell() {  # cell <tag> <tok> <new> <defer:0|1>
    local tag="$1" tok="$2" new="$3" defer="$4"
    local OUT=$ROOT/artifacts/rocm/logs/r30d-cell-$tag-$STAMP.log
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_VERIFY_DEFER_CPU=$defer \
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
    [ "$rc" = "0" ] || { echo "  (failed - $OUT)"; GATES_OK=0; }
}
for p in 1 2 3 4 5; do
    if [ $((p % 2)) = 1 ]; then cell b1k-$p "$T1K" 256 0; cell d1k-$p "$T1K" 256 1
    else                         cell d1k-$p "$T1K" 256 1; cell b1k-$p "$T1K" 256 0; fi
done
for p in 1 2; do
    if [ $((p % 2)) = 1 ]; then
        cell b32k-$p "$T32K" 128 0; cell d32k-$p "$T32K" 128 1
        cell b128k-$p "$T128K" 64 0; cell d128k-$p "$T128K" 64 1
    else
        cell d32k-$p "$T32K" 128 1; cell b32k-$p "$T32K" 128 0
        cell d128k-$p "$T128K" 64 1; cell b128k-$p "$T128K" 64 0
    fi
done

echo "== 5. verdict =="
python3 - "$STAMP" <<'PY'
import difflib, glob, json, re, statistics, sys
stamp = sys.argv[1]
def battery(tag):
    rows = [json.loads(l) for l in open(f"/tmp/r30d-battery-{tag}-{stamp}.jsonl")]
    return {r["key"]: r.get("pass", False) for r in rows}
b1, b2, b3 = battery("A1"), battery("A2"), battery("A3")
base_pass, arm_pass, base2_pass = sum(b1.values()), sum(b2.values()), sum(b3.values())
fails = [k for k in b1 if b1[k] and not b2[k]]
a1_ok = arm_pass >= base_pass and not fails
print(f"GATE A1: base {base_pass}/8 | defer {arm_pass}/8 | base-again {base2_pass}/8 | defer regressions: {fails or 'none'} -> {'PASS' if a1_ok else 'FAIL'}")
def txt(tag): return open(f"/tmp/r30d-prose-{tag}-{stamp}.txt").read()
def ratio(a, b): return difflib.SequenceMatcher(None, a, b).ratio()
restart_floor = 1.0 - ratio(txt("A1"), txt("A3")) if txt("A3") else 1.0
arm_drift = 1.0 - ratio(txt("A1"), txt("A2"))
a2_lim = restart_floor * 1.6 + 0.05
print(f"GATE A2: prose divergence defer-vs-base {arm_drift:.3f} vs restart floor {restart_floor:.3f} (limit {a2_lim:.3f}) -> {'PASS' if arm_drift <= a2_lim else 'REVIEW'}")
a2_ok = arm_drift <= a2_lim
def cell(tag):
    f = glob.glob(f"/home/zacch/projects/strata-amd/artifacts/rocm/logs/r30d-cell-{tag}-{stamp}.log")
    if not f: return None
    t = open(f[0]).read()
    d = re.search(r"decode +(\d+) tokens in ([\d.]+) ms +-> +([\d.]+) tok/s", t)
    r = re.search(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+)", t)
    if not (d and r): return None
    return dict(tps=float(d.group(3)), rounds=int(r.group(1)), accepted=int(r.group(3)), offered=int(r.group(4)),
                ms_per_round=float(d.group(2)) / int(r.group(1)))
def report(tier, n, need_wins, med_gate, no_reg):
    rows = []
    for i in range(1, n + 1):
        b, a = cell(f"b{tier}-{i}"), cell(f"d{tier}-{i}")
        if not (b and a):
            print(f"GATE B {tier}: INCOMPLETE (pair {i})"); return False
        rows.append((b, a))
    d_tps = [100 * (a["tps"] - b["tps"]) / b["tps"] for b, a in rows]
    d_msr = [100 * (a["ms_per_round"] - b["ms_per_round"]) / b["ms_per_round"] for b, a in rows]
    wins = sum(1 for x in d_tps if x > 0)
    med = statistics.median(d_tps)
    print(f"GATE B {tier}: base {statistics.median([b['tps'] for _, b in rows]):.2f} -> defer "
          f"{statistics.median([a['tps'] for _, a in rows]):.2f} tok/s | d_tps " + " ".join(f"{x:+.2f}%" for x in d_tps)
          + f" med {med:+.2f}% wins {wins}/{n}")
    print(f"   ms/round " + " ".join(f"{x:+.2f}%" for x in d_msr)
          + f" | acceptance base {[b['accepted'] for b, _ in rows]} defer {[a['accepted'] for _, a in rows]}")
    ok = med >= med_gate and wins >= need_wins and all(x > no_reg for x in d_msr)
    print(f"   -> {'PASS' if ok else 'FAIL'} (median >= {med_gate:+.1f}%, >= {need_wins}/{n} wins, no ms/round < {no_reg:+.0f}%)")
    return ok
b1_ok = report("1k", 5, 4, 2.0, -2.0)
b32_ok = report("32k", 2, 0, 0.0, -2.0)
b128_ok = report("128k", 2, 0, -1.0, -2.0)
verdict = "ADOPT" if (a1_ok and a2_ok and b1_ok and b32_ok and b128_ok) else "NOT ADOPTED"
print(f"WINDOW VERDICT: {verdict} (quality A1={a1_ok} A2={a2_ok}; perf 1K={b1_ok} 32K={b32_ok} 128K={b128_ok})")
PY
echo "== done (trap restores 8081 from the unchanged production config) =="
