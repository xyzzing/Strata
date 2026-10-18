#!/bin/bash
# r30c-iq3s-262k-topk-window.sh - IQ3_S #2514 port gate at 262K (2026-10-18).
#
# The port (src/kernels/cuda/qsa_select.cu, STRATA_TOPK_DECODE_REG=1) changes ONLY the 262K-context
# decode top-k: 65,538 blocks > TK_T*TK_PER (33,792) -> the wide kernel in the base; with the env the
# PER_MAX register kernel holds them (fit 67,584). At 131K the dispatch is the register kernel either way.
#
# Window:
#   1. qsa_topk_parity (default and STRATA_TOPK_DECODE_REG=1) + qsa_topk_parity_reg_always - same ids
#   2. paired 262K decode cells, interleaved: base (wide) vs STRATA_TOPK_DECODE_REG=1 (register), 3 pairs
#   3. verdict: accepted counts identical (same ids), median ms/round improvement >= 3% = ADOPT
# 8080 untouached; the trap restores strata-8081 and checks /health.
set -u
WT=/home/zacch/projects/strata-r145
BUILD=build-0145
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r30c-topk-window-$STAMP.log
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
T262=$ROOT/artifacts/rocm/workloads/qsa-profile-261888.tokens

echo "== 0. preflight ($STAMP) =="
[ -x "$EXE" ] || { echo "FATAL: binary missing"; exit 9; }
[ -f "$T262" ] || { echo "FATAL: 262K workload missing"; exit 9; }
"$EXE" --version | tee /tmp/r30c-version.txt

echo "== 1. stop 8081 + drain gates =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do
    [ "$(memav)" -ge 68000 ] && [ "$(vram)" -le 2600 ] && break
    sleep 4
done
MEM=$(memav); VR=$(vram)
echo "8081: $(systemctl --user is-active strata-8081.service); MemAvailable ${MEM} MiB (gate 68000); VRAM ${VR} MiB (gate 2600)"
[ "$MEM" -ge 68000 ] && [ "$VR" -le 2600 ] || { echo "FATAL: drain gates not met"; exit 9; }

echo "== 2. top-k parity, same ids (default and REG=1) =="
cd "$WT/$BUILD" || exit 9
GATES_OK=1
run_test() {  # run_test <label> <env...>
    local label="$1"; shift
    local t
    for t in qsa_topk_parity qsa_topk_parity_reg_always; do
        if env "$@" LD_LIBRARY_PATH=$EXP/sdk/lib ctest -R "^${t}\$" --output-on-failure > "ctest-r30c-${label}-${t}.log" 2>&1 \
           && grep -q "100% tests passed" "ctest-r30c-${label}-${t}.log" \
           && ! grep -qi "No tests were found" "ctest-r30c-${label}-${t}.log"; then
            echo "CTEST $label $t: PASS"
        else
            echo "CTEST $label $t: FAIL (tail: $(tail -3 "ctest-r30c-${label}-${t}.log" | tr '\n' ' ' | head -c 240))"; GATES_OK=0
        fi
    done
}
run_test def
run_test reg STRATA_TOPK_DECODE_REG=1
[ "$GATES_OK" = "1" ] || { echo "PARITY FAILED - no cells"; exit 4; }

cell() {  # cell <tag> <extra-env...>
    local tag="$1"; shift
    local OUT=$ROOT/artifacts/rocm/logs/r30c-topk-cell-$tag-$STAMP.log
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 \
        "$@" \
        systemd-run --user --scope -p MemoryMax=66G \
        "$EXE" --tokens-file "$T262" --max-new 64 --max-context 262144 \
        --pack $EXP/packs/iq3_s \
        --native /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        --ple-gguf /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
        --mtp /mnt/LINUXWINSHARE/models/mtp/rt \
        --expert-profile $EXP/data/expert-profile.bin \
        --expert-cache auto --spec 5 --kv int8 --kv-resident 32768 --prefill 8192 \
        --prompt-cache 6 --spec-min-p 0.70 --vram-reserve-mib 3072 --lookup-chain 2 > "$OUT" 2>&1
    local rc=$? d acc p
    d=$(grep -oE "decode +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    p=$(grep -oE "prefill +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    acc=$(grep -oE "drafts accepted [0-9]+ of [0-9]+" "$OUT" | head -1)
    echo "CELL $tag: rc=$rc prefill=$p decode=$d $acc"
    [ "$rc" = "0" ] || { echo "  (cell failed - log $OUT)"; GATES_OK=0; }
}

echo "== 3. paired 262K cells (3 pairs, interleaved) =="
for i in 1 2 3; do
    cell base-262k-$i
    cell reg-262k-$i STRATA_TOPK_DECODE_REG=1
done

echo "== 4. verdict =="
python3 - "$STAMP" "$(cat /tmp/r30c-version.txt)" <<'PY'
import glob, re, statistics, sys
stamp, version = sys.argv[1], sys.argv[2]
root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"

def cell(tag):
    f = glob.glob(f"{root}/r30c-topk-cell-{tag}-{stamp}.log")
    if not f: return None
    t = open(f[0]).read()
    d = re.search(r"decode +(\d+) tokens in ([\d.]+) ms +-> +([\d.]+) tok/s", t)
    r = re.search(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+)", t)
    if not (d and r): return None
    return dict(tps=float(d.group(3)), ms=float(d.group(2)), rounds=int(r.group(1)),
                accepted=int(r.group(3)), offered=int(r.group(4)))

ps = []
for i in (1, 2, 3):
    b, e = cell(f"base-262k-{i}"), cell(f"reg-262k-{i}")
    if not (b and e):
        print(f"VERDICT INCOMPLETE: pair {i} missing"); sys.exit(4)
    ps.append((b, e))
d_tps = [100 * (e["tps"] - b["tps"]) / b["tps"] for b, e in ps]
d_msr = [100 * (e["ms"] / e["rounds"] - b["ms"] / b["rounds"]) / (b["ms"] / b["rounds"]) for b, e in ps]
acc_ok = all(e["accepted"] == b["accepted"] and e["offered"] == b["offered"] for b, e in ps)
med_tps, med_msr = statistics.median(d_tps), statistics.median(d_msr)
spread = max(abs(x - med_msr) for x in d_msr)
print(f"base {statistics.median([b['tps'] for _, b in ps]):.2f} -> reg {statistics.median([e['tps'] for _, e in ps]):.2f} tok/s")
print("d_tps " + " ".join(f"{x:+.2f}%" for x in d_tps) + f" med {med_tps:+.2f}%")
print("d_ms/round " + " ".join(f"{x:+.2f}%" for x in d_msr) + f" med {med_msr:+.2f}% (spread {spread:.2f})")
print(f"acceptance identical: {acc_ok}")
ok = med_msr <= -3.0 and acc_ok and all(x <= 2.0 for x in d_msr)
print(f"GATE: {'PASS - register kernel wins at 262K' if ok else 'FAIL'} (need median ms/round <= -3%, same ids/acceptance, no pair > +2%)")
print(f"WINDOW VERDICT: {'ADOPT STRATA_TOPK_DECODE_REG=1' if ok else 'NOT ADOPTED'} on {version}")
sys.exit(0 if ok else 3)
PY
echo "== done (trap restores 8081) =="
