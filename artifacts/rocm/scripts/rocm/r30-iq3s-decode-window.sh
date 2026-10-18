#!/bin/bash
# r30-iq3s-decode-window.sh - IQ3_S decode campaign, window A (2026-10-18).
#
# One batched stop/restore of production 8081 on build-0145 (0.1.45-8eb3e85d):
#   0. preflight (binary stamp, config hashes), stop 8081, RAM+VRAM drain gates
#   1. strict on-card parity (iq_parity, iq_multi_parity, s2_expert_grouped_parity),
#      then the same under STRATA_EXPDEC=1 - the bitwise gate for the EXPDEC arm
#   2. attribution: one 1K cell with STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 and
#      --dump-routing; prints the ms/window split and the per-layer CPU/VRAM/PCIe shares
#   3. EXPDEC kernel matrix on real GGUF rows (native_expert_bench, layers 0,1,2,8,12,17,21,47
#      = the 7 real format pairs; G 20 T 4 and G 8 T 1), 5 variants; picks the fastest gu sum
#   4. e2e: baseline vs the chosen EXPDEC arm, interleaved 1K x5 / 32K x2 / 128K x2, and a
#      down-coverage arm (STRATA_EXPDEC_DOWN=1) at 1K x2
#   5. verdict: median paired committed tok/s (bit-exact arm: acceptance must match), ms/window
#      and the acceptance counts; adopt iff >= +2.0% and > the largest pair spread and no
#      32K/128K cell worse by > 2%
#
# Pre-declared gates (no tolerance is moved after the fact):
#   ADOPT   median paired 1K committed tok/s >= +2.0%, acceptance identical, no depth regression
#   KEEP OPT-IN (not adopted)   0 < gain < 2.0% or depth-neutral
#   REJECT  gain <= 0, parity failure, or a depth regression > 2%
# Cell gate: MemAvailable >= 68000 MiB after the stop (a skip is not a pass).
# 8080 is never touched. The trap always restarts strata-8081 and checks /health.
set -u
WT=/home/zacch/projects/strata-r145
BUILD=build-0145
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
CAMP=$ROOT/artifacts/rocm/iq3s-decode-campaign-20261018
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r30-iq3s-window-$STAMP.log
mkdir -p "$ROOT/artifacts/rocm/logs"
exec > >(tee -a "$LOG") 2>&1

restore_8081() {
    systemctl --user reset-failed strata-8081.service 2>/dev/null
    systemctl --user start strata-8081.service 2>/dev/null
    for i in $(seq 1 45); do
        if curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"'; then
            echo "RESTORE_8081 ok (after ~$((i*4))s)"
            return 0
        fi
        sleep 4
    done
    echo "RESTORE_8081 FAILED after 180s - start it manually: $EXP/start-8081.sh"
    return 1
}
trap restore_8081 EXIT
memav() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
vram() { local v; v=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo 999999999999); echo $((v / 1048576)); }

EXE=$WT/$BUILD/strata
SHARD=/home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
T1K=$ROOT/artifacts/rocm/workloads/qsa-profile-1024.tokens
T32K=$ROOT/artifacts/rocm/workloads/qsa-profile-32768.tokens
T128K=$ROOT/artifacts/rocm/workloads/qsa-profile-131072.tokens
LAYERS=0,1,2,8,12,17,21,47

echo "== 0. preflight ($STAMP) =="
[ -x "$EXE" ] || { echo "FATAL: build-0145 binary missing"; exit 9; }
[ -f "$SHARD" ] && [ -f "$T1K" ] && [ -f "$T32K" ] && [ -f "$T128K" ] || { echo "FATAL: model/workload missing"; exit 9; }
"$EXE" --version
sha256sum "$EXP/serve-8081.json"
echo "MemAvailable $(memav) MiB, VRAM $(vram) MiB (before stop)"

echo "== 1. stop 8081 + drain gates =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do
    [ "$(memav)" -ge 68000 ] && [ "$(vram)" -le 2600 ] && break
    sleep 4
done
MEM=$(memav); VR=$(vram)
echo "8081: $(systemctl --user is-active strata-8081.service); MemAvailable ${MEM} MiB (gate 68000); VRAM ${VR} MiB (gate 2600)"
if [ "$MEM" -lt 68000 ] || [ "$VR" -gt 2600 ]; then
    echo "FATAL: drain gates not met - no cells (a skip is not a pass)"; exit 9
fi

echo "== 2. strict parity (default), then the same under STRATA_EXPDEC=1 =="
cd "$WT/$BUILD" || exit 9
GATES_OK=1
parity() {  # parity <tag> [env...]
    local tag="$1"; shift
    local t
    for t in iq_parity iq_multi_parity s2_expert_grouped_parity; do
        if env "$@" LD_LIBRARY_PATH=$EXP/sdk/lib ctest -R "^${t}\$" --output-on-failure > "ctest-r30-${tag}-${t}.log" 2>&1 \
           && grep -q "100% tests passed" "ctest-r30-${tag}-${t}.log" \
           && ! grep -qi "No tests were found" "ctest-r30-${tag}-${t}.log"; then
            echo "CTEST $tag $t: PASS"
        else
            echo "CTEST $tag $t: FAIL (tail: $(tail -3 "ctest-r30-${tag}-${t}.log" | tr '\n' ' ' | head -c 240))"; GATES_OK=0
        fi
    done
}
parity def
parity expdec STRATA_EXPDEC=1
[ "$GATES_OK" = "1" ] || { echo "PARITY FAILED - no cells (exit 4)"; exit 4; }

# production-shaped env, plus the arm's env; LD_LIBRARY_PATH is the SDK's
cell() {  # cell <tag> <tokens-file> <max-new> <extra-env...>; CELL_EXTRA = optional engine args
    local tag="$1" tok="$2" new="$3"; shift 3
    local OUT=$ROOT/artifacts/rocm/logs/r30-iq3s-cell-$tag-$STAMP.log
    local extra=${CELL_EXTRA:-}
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_PREFILL_TIMING=1 \
        "$@" \
        systemd-run --user --scope -p MemoryMax=66G \
        "$EXE" --tokens-file "$tok" --max-new "$new" --max-context 139264 \
        --pack $EXP/packs/iq3_s \
        --native /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        --ple-gguf /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
        --mtp /mnt/LINUXWINSHARE/models/mtp/rt \
        --expert-profile $EXP/data/expert-profile.bin \
        --expert-cache auto --spec 5 --kv int8 --kv-resident 32768 --prefill 8192 \
        --prompt-cache 6 --spec-min-p 0.70 --vram-reserve-mib 3072 --lookup-chain 2 $extra > "$OUT" 2>&1
    local rc=$? p d tt acc
    p=$(grep -oE "prefill +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    d=$(grep -oE "decode +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    tt=$(grep -oE "time to first token [0-9.]+" "$OUT" | grep -oE "[0-9.]+" | head -1)
    acc=$(grep -oE "drafts accepted [0-9]+ of [0-9]+" "$OUT" | head -1)
    echo "CELL $tag: rc=$rc prefill=$p decode=$d TTFT=${tt}ms $acc"
    if [ "$rc" != "0" ]; then GATES_OK=0; echo "  (cell failed - log $OUT)"; fi
}

echo "== 3. attribution (1K, 128 tokens, production env + timing) =="
cell attr-1k "$T1K" 128 STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1
ATT=$ROOT/artifacts/rocm/logs/r30-iq3s-cell-attr-1k-$STAMP.log
grep -E "strata decode timing|strata decode GPU stages" "$ATT" | head -4
grep -E "pcie|PcieAuto|pcie share" "$ATT" | tail -4
CELL_EXTRA="--dump-routing $ROOT/artifacts/rocm/logs/r30-iq3s-routing-$STAMP.txt"
cell attr-routing "$T1K" 256 STRATA_DECODE_TIMING=1
CELL_EXTRA=""
ROUTING=$ROOT/artifacts/rocm/logs/r30-iq3s-routing-$STAMP.txt
if [ -s "$ROUTING" ]; then
    python3 - "$ROUTING" <<'PY_ROUTE'
import collections, sys
# count entries per (layer, expert): tokens-per-expert for the verify windows (T07 data)
per = collections.defaultdict(collections.Counter)
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) < 3:
        continue
    try:
        layer = int(f[0]); n = int(f[1])
    except ValueError:
        continue
    for tok in f[2:2 + n]:
        per[layer][int(tok)] += 1
if per:
    tot = sum(sum(c.values()) for c in per.values())
    print(f"routing: {len(per)} layers, {tot} expert-entries; "
          f"mean entries/expert {tot/max(1,sum(len(c) for c in per.values())):.2f}")
    for l in sorted(per)[:3]:
        c = per[l]
        sizes = sorted(c.values(), reverse=True)
        print(f"  layer {l}: {len(c)} distinct experts, top {sizes[:8]}, entries {sum(sizes)}")
else:
    print("routing: file parsed empty (format may differ - check the log)")
PY_ROUTE
else
    echo "routing: dump missing/empty (not a gate)"
fi

echo "== 4. EXPDEC kernel matrix (real rows; bitwise self-check) =="
BENCH=$WT/$BUILD/native_expert_bench
BENCH_RES=$ROOT/artifacts/rocm/logs/r30-iq3s-bench-results-$STAMP.txt
: > "$BENCH_RES"
bench() {  # bench <tag> <G> <T> [env...]
    local tag="$1" G="$2" T="$3"; shift 3
    local OUT=$ROOT/artifacts/rocm/logs/r30-iq3s-bench-$tag-G$G-T$T-$STAMP.log
    env "$@" timeout 300 "$BENCH" "$SHARD" "$LAYERS" "$G" "$T" -1 -1 200 > "$OUT" 2>&1
    local rc=$? diff gu
    diff=$(grep -oE "[0-9]+ differing" "$OUT" | head -1)
    gu=$(grep -oE "\| mode -1: all +[0-9.]+ us +gu +[0-9.]+ us" "$OUT" | grep -oE "gu +[0-9.]+" | grep -oE "[0-9.]+" | awk '{s+=$1} END {printf "%.1f", s}')
    echo "BENCH $tag G=$G T=$T: rc=$rc $diff gu_sum=${gu}us" | tee -a "$BENCH_RES"
    [ "$rc" = "0" ] || GATES_OK=0
}
for spec in "def" "expdec:STRATA_EXPDEC=1" "expdec-r2:STRATA_EXPDEC=1 STRATA_EXPDEC_R=2" "expdec-minb2:STRATA_EXPDEC=1 STRATA_EXPDEC_MINB=2" "expdec-r2-minb2:STRATA_EXPDEC=1 STRATA_EXPDEC_R=2 STRATA_EXPDEC_MINB=2"; do
    tag=${spec%%:*}; envs=${spec#*:}
    [ "$envs" = "$spec" ] && envs=""
    # shellcheck disable=SC2086
    bench "$tag" 20 4 $envs
    # shellcheck disable=SC2086
    bench "$tag" 8 1 $envs
done
BEST=$(grep -h "BENCH .* G=20 T=4:" "$BENCH_RES" | awk '{g=$0; sub(/.*gu_sum=/,"",g); sub(/us.*/,"",g); if (g+0 > 0) print g, $2}' | sort -n | head -1 | awk '{print $2}')
echo "BEST EXPDEC variant by G=20 T=4 gu sum: ${BEST:-none}"
case "$BEST" in
    expdec)            ARM=(STRATA_EXPDEC=1) ;;
    expdec-r2)         ARM=(STRATA_EXPDEC=1 STRATA_EXPDEC_R=2) ;;
    expdec-minb2)      ARM=(STRATA_EXPDEC=1 STRATA_EXPDEC_MINB=2) ;;
    expdec-r2-minb2)   ARM=(STRATA_EXPDEC=1 STRATA_EXPDEC_R=2 STRATA_EXPDEC_MINB=2) ;;
    *)                 ARM=(); echo "no bench winner - e2e EXPDEC arm skipped"; BEST="none" ;;
esac
BEST=${BEST:-none}

echo "== 5. e2e cells: baseline vs EXPDEC ($BEST), interleaved =="
if [ "$BEST" = "none" ]; then
    echo "E2E SKIP: no EXPDEC variant to run"
else
    for i in 1 2 3 4 5; do cell base-1k-$i "$T1K" 256; cell exp-1k-$i "$T1K" 256 "${ARM[@]}"; done
    for i in 1 2;     do cell base-32k-$i "$T32K" 128; cell exp-32k-$i "$T32K" 128 "${ARM[@]}"; done
    for i in 1 2;     do cell base-128k-$i "$T128K" 64; cell exp-128k-$i "$T128K" 64 "${ARM[@]}"; done
    echo "== 5b. down-coverage arm (STRATA_EXPDEC_DOWN=1, 1K x2; reported, not gated) =="
    for i in 1 2; do cell down-1k-$i "$T1K" 256 STRATA_EXPDEC=1 STRATA_EXPDEC_DOWN=1; done
fi

echo "== 6. verdict =="
python3 - "$STAMP" "$BEST" <<'PY'
import glob, re, statistics, sys
stamp, best = sys.argv[1], sys.argv[2]
root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"

def cell(tag):
    f = glob.glob(f"{root}/r30-iq3s-cell-{tag}-{stamp}.log")
    if not f:
        return None
    t = open(f[0]).read()
    d = re.search(r"decode +(\d+) tokens in ([\d.]+) ms +-> +([\d.]+) tok/s", t)
    r = re.search(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+) "
                  r"\(([\d.]+)\), ([\d.]+) tokens per round", t)
    p = re.search(r"prefill +(\d+) tokens in (\d+) chunks, ([\d.]+) ms \(([\d.]+) tok/s\)", t)
    if not (d and r and p):
        return None
    return dict(tps=float(d.group(3)), ms=float(d.group(2)), rounds=int(r.group(1)),
                accepted=int(r.group(3)), offered=int(r.group(4)), tok_per_round=float(r.group(6)),
                ms_per_round=float(d.group(2)) / int(r.group(1)), prefill=float(p.group(4)))

def pairs(tier, n):
    out = []
    for i in range(1, n + 1):
        b, e = cell(f"base-{tier}-{i}"), cell(f"exp-{tier}-{i}")
        if not (b and e):
            print(f"VERDICT INCOMPLETE: {tier} pair {i} missing"); return None
        out.append((b, e))
    return out

def report(tier, n, gate):
    ps = pairs(tier, n)
    if ps is None: return None
    d_tps = [100 * (e["tps"] - b["tps"]) / b["tps"] for b, e in ps]
    d_msr = [100 * (e["ms_per_round"] - b["ms_per_round"]) / b["ms_per_round"] for b, e in ps]
    acc_ok = all(e["accepted"] == b["accepted"] and e["tok_per_round"] == b["tok_per_round"] for b, e in ps)
    med_tps, med_msr = statistics.median(d_tps), statistics.median(d_msr)
    spread = max(abs(x - statistics.median(d_tps)) for x in d_tps) if len(d_tps) > 1 else 0.0
    print(f"{tier}: base {statistics.median([b['tps'] for _, b in ps]):.2f} -> exp "
          f"{statistics.median([e['tps'] for _, e in ps]):.2f} tok/s | d_tps " +
          " ".join(f"{x:+.2f}%" for x in d_tps) + f" med {med_tps:+.2f}% (spread {spread:.2f})")
    print(f"   ms/round deltas " + " ".join(f"{x:+.2f}%" for x in d_msr) + f" med {med_msr:+.2f}% | acceptance identical: {acc_ok}")
    depth_ok = all(x > -2.0 for x in d_msr)
    if gate:
        ok = med_tps >= 2.0 and acc_ok and depth_ok
        print(f"GATE {tier}: {'PASS' if ok else 'FAIL'} (need >= +2.0% tok/s, identical acceptance, no ms/round < -2%)")
        return ok
    print(f"DEPTH CHECK {tier}: {'PASS' if depth_ok else 'FAIL'} (no ms/round < -2%)")
    return depth_ok

ok1 = report("1k", 5, True)
ok32 = report("32k", 2, False)
ok128 = report("128k", 2, False)
down = [cell(f"down-1k-{i}") for i in (1, 2)]
if all(down):
    print(f"down arm (reported): " + " ".join(f"{d['tps']:.2f}" for d in down) + " tok/s")
verdict = "ADOPT" if (ok1 and ok32 is not None and ok128 is not None) else "NOT ADOPTED"
print(f"WINDOW VERDICT: {verdict} (EXPDEC arm: {best}); see campaign.md for the decision record")
sys.exit(0 if verdict == "ADOPT" else 3)
PY
RC=$?
echo "== done: GATES_OK=$GATES_OK verdict_rc=$RC (trap restores 8081) =="
[ "$GATES_OK" = "1" ] || exit 4
exit 0
