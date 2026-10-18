#!/bin/bash
# r30b-iq3s-window.sh - IQ3_S decode campaign, window B (2026-10-18).
#
# Window A proved EXPDEC gate/up is slower on gfx1100 and found the down win (-27% at the real
# G34 T1 envelope). Window B does what window A could not:
#   1. the GPU stage profile, via the SERVE path (one-shot does not print the stage line):
#      server.py on 8081 with STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1, one fixed request
#   2. the campaign baseline cells (production env, no timing): 1K x5, 32K x2, 128K x2
#   3. the down arm (STRATA_EXPDEC=1 STRATA_EXPDEC_DOWN=1 STRATA_EXPDEC_D20=1): 1K x5, 32K x2,
#      128K x2 (bit-exact arm, so acceptance must be identical)
#   4. verdict under the pre-declared gate: adopt iff median paired committed tok/s >= +2.0%,
#      acceptance identical and no depth cell worse by >2%
# 8080 is never touched. The trap always restarts strata-8081 and checks /health.
set -u
WT=/home/zacch/projects/strata-r145
BUILD=build-0145
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r30b-iq3s-window-$STAMP.log
ATTRLOG=$ROOT/artifacts/rocm/logs/r30b-iq3s-attr-$STAMP.log
mkdir -p "$ROOT/artifacts/rocm/logs"
exec > >(tee -a "$LOG") 2>&1

restore_8081() {
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
[ -x "$EXE" ] || { echo "FATAL: binary missing"; exit 9; }
"$EXE" --version

echo "== 1. stop 8081 + drain gates =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do
    [ "$(memav)" -ge 68000 ] && [ "$(vram)" -le 2600 ] && break
    sleep 4
done
MEM=$(memav); VR=$(vram)
echo "8081: $(systemctl --user is-active strata-8081.service); MemAvailable ${MEM} MiB (gate 68000); VRAM ${VR} MiB (gate 2600)"
[ "$MEM" -ge 68000 ] && [ "$VR" -le 2600 ] || { echo "FATAL: drain gates not met (a skip is not a pass)"; exit 9; }

echo "== 2. GPU stage profile through the serve path (STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1) =="
python3 - > /tmp/r30b-attr-request.json <<'PY'
import json
para = ("The quick brown fox jumps over the lazy dog while the engineer measures the decode window "
        "and writes down every stage of the pipeline, because a number without its provenance is a rumour. ")
text = para * 80
print(json.dumps({"model": "qwen3.8-flash-next", "max_tokens": 128, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": text}]}))
PY
setsid env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
    STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
    STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
    STRATA_ROUTE_TAIL_SKIP=7 \
    STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 \
    $EXP/venv/bin/python $EXP/hipify/serve/server.py --engine strata \
    --config $EXP/serve-8081.json --port 8081 --host 127.0.0.1 > "$ATTRLOG" 2>&1 &
SRV=$!
UP=0
for i in $(seq 1 60); do
    curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && { UP=1; break; }
    sleep 3
done
if [ "$UP" = "1" ]; then
    echo "attribution server up; posting one request"
    curl -s -m 400 -X POST http://127.0.0.1:8081/v1/chat/completions \
        -H "Content-Type: application/json" -d @/tmp/r30b-attr-request.json > /tmp/r30b-attr-response.json
    echo "response bytes: $(wc -c < /tmp/r30b-attr-response.json)"
    sleep 2
else
    echo "ATTR FAIL: server did not come up (log $ATTRLOG)"
fi
kill -- -$SRV 2>/dev/null
for i in $(seq 1 30); do [ "$(vram)" -le 2600 ] && break; sleep 2; done
if [ "$(vram)" -gt 2600 ]; then
    PIDS=$(pgrep -f "$BUILD/strata --serve" || true)
    echo "engine still holding VRAM (pids: ${PIDS:-none}) - terminating it"
    [ -n "${PIDS:-}" ] && kill $PIDS 2>/dev/null
    sleep 8
fi
echo "VRAM after attribution: $(vram) MiB"
echo "--- attribution lines ---"
grep -E "strata decode timing|strata decode GPU stages|verify window|pcie experts|mtp " "$ATTRLOG" | tail -8

cell() {  # cell <tag> <tokens-file> <max-new> <extra-env...>
    local tag="$1" tok="$2" new="$3"; shift 3
    local OUT=$ROOT/artifacts/rocm/logs/r30b-iq3s-cell-$tag-$STAMP.log
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 \
        "$@" \
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
    echo "CELL $tag: rc=$rc decode=$d $acc"
    [ "$rc" = "0" ] || echo "  (cell failed - log $OUT)"
}

echo "== 3. baseline cells =="
for i in 1 2 3 4 5; do cell base-1k-$i "$T1K" 256; done
for i in 1 2;     do cell base-32k-$i "$T32K" 128; done
for i in 1 2;     do cell base-128k-$i "$T128K" 64; done

echo "== 4. down arm (EXPDEC + _DOWN + _D20) =="
ARM=(STRATA_EXPDEC=1 STRATA_EXPDEC_DOWN=1 STRATA_EXPDEC_D20=1)
for i in 1 2 3 4 5; do cell down-1k-$i "$T1K" 256 "${ARM[@]}"; done
for i in 1 2;     do cell down-32k-$i "$T32K" 128 "${ARM[@]}"; done
for i in 1 2;     do cell down-128k-$i "$T128K" 64 "${ARM[@]}"; done

echo "== 5. verdict =="
python3 - "$STAMP" <<'PY'
import glob, re, statistics, sys
stamp = sys.argv[1]
root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"

def cell(tag):
    f = glob.glob(f"{root}/r30b-iq3s-cell-{tag}-{stamp}.log")
    if not f: return None
    t = open(f[0]).read()
    d = re.search(r"decode +(\d+) tokens in ([\d.]+) ms +-> +([\d.]+) tok/s", t)
    r = re.search(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+) "
                  r"\(([\d.]+)\), ([\d.]+) tokens per round", t)
    if not (d and r): return None
    return dict(tps=float(d.group(3)), rounds=int(r.group(1)), accepted=int(r.group(3)),
                offered=int(r.group(4)), tok_per_round=float(r.group(6)),
                ms_per_round=float(d.group(2)) / int(r.group(1)))

def report(tier, n, gate):
    ps = []
    for i in range(1, n + 1):
        b, e = cell(f"base-{tier}-{i}"), cell(f"down-{tier}-{i}")
        if not (b and e):
            print(f"VERDICT INCOMPLETE: {tier} pair {i} missing"); return None
        ps.append((b, e))
    d_tps = [100 * (e["tps"] - b["tps"]) / b["tps"] for b, e in ps]
    d_msr = [100 * (e["ms_per_round"] - b["ms_per_round"]) / b["ms_per_round"] for b, e in ps]
    acc_ok = all(e["accepted"] == b["accepted"] and e["tok_per_round"] == b["tok_per_round"] for b, e in ps)
    med_tps, med_msr = statistics.median(d_tps), statistics.median(d_msr)
    spread = max(abs(x - statistics.median(d_tps)) for x in d_tps) if len(d_tps) > 1 else 0.0
    print(f"{tier}: base {statistics.median([b['tps'] for _, b in ps]):.2f} -> down "
          f"{statistics.median([e['tps'] for _, e in ps]):.2f} tok/s | d_tps " + " ".join(f"{x:+.2f}%" for x in d_tps)
          + f" med {med_tps:+.2f}% (spread {spread:.2f})")
    print(f"   ms/round " + " ".join(f"{x:+.2f}%" for x in d_msr) + f" med {med_msr:+.2f}% | acceptance identical: {acc_ok}")
    depth_ok = all(x > -2.0 for x in d_msr)
    if gate:
        ok = med_tps >= 2.0 and acc_ok and depth_ok
        print(f"GATE {tier}: {'PASS' if ok else 'FAIL'} (need >= +2.0% tok/s, identical acceptance, no ms/round < -2%)")
        return ok
    print(f"DEPTH CHECK {tier}: {'PASS' if depth_ok else 'FAIL'}")
    return depth_ok

ok1 = report("1k", 5, True)
ok32 = report("32k", 2, False)
ok128 = report("128k", 2, False)
verdict = "ADOPT" if (ok1 and ok32 and ok128) else "NOT ADOPTED"
print(f"WINDOW VERDICT: {verdict} (down arm: STRATA_EXPDEC=1 _DOWN=1 _D20=1)")
sys.exit(0 if verdict == "ADOPT" else 3)
PY
echo "== done (trap restores 8081) =="
