#!/bin/bash
# r30i-route-prior-confirm.sh - clean interleaved confirmation of STRATA_ROUTE_PRIOR=0.5 (2026-10-18)
# The r30f screen reused base tags and its verdict had an inverted ms/round sign; the raw numbers
# and the battery (7/8 = 7/8, no regressions) point to a real candidate. This window confirms with
# proper alternation before any production change:
#   @1K 5 pairs, @32K 2 pairs, @128K 2 pairs; base = env unset, arm = STRATA_ROUTE_PRIOR=0.5.
# Gate: median tok/s >= +2.0% at 1K with >= 4/5 wins; 32K/128K median >= -1%; no tier median
# ms/round > +2% slower. Acceptance and the CPU/PCIe shares are reported, not gated.
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
WT=/home/zacch/projects/strata-r145; BUILD=build-0145; EXE=$WT/$BUILD/strata
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r30i-confirm-$STAMP.log
T1K=$ROOT/artifacts/rocm/workloads/qsa-profile-1024.tokens
T32K=$ROOT/artifacts/rocm/workloads/qsa-profile-32768.tokens
T128K=$ROOT/artifacts/rocm/workloads/qsa-profile-131072.tokens
mkdir -p "$ROOT/artifacts/rocm/logs"; exec > >(tee -a "$LOG") 2>&1
restore_8081() {
    systemctl --user reset-failed strata-8081.service 2>/dev/null
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
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(memav)" -ge 68000 ] && [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "gates: MemAvailable $(memav) MiB, VRAM $(vram) MiB"
cell() {  # cell <tag> <tok> <new> <arm 0|1>
    local tag="$1" tok="$2" new="$3" arm="$4"
    local OUT=$ROOT/artifacts/rocm/logs/r30i-cell-$tag-$STAMP.log
    local envs=(); [ "$arm" = "1" ] && envs=(STRATA_ROUTE_PRIOR=0.5)
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
    local rc=$? d acc cpu pcie
    d=$(grep -oE "decode +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    acc=$(grep -oE "drafts accepted [0-9]+ of [0-9]+" "$OUT" | head -1)
    cpu=$(grep -oE "CPU experts [0-9.]+ distinct / [0-9.]+ routed" "$OUT" | head -1)
    pcie=$(grep -oE "pcie experts +[0-9.]+ distinct" "$OUT" | head -1)
    echo "CELL $tag rc=$rc decode=$d $acc | $cpu | $pcie"
    [ "$rc" = "0" ] || echo "  (failed - $OUT)"
}
echo "== 1K x5, alternating order =="
for p in 1 2 3 4 5; do
    if [ $((p % 2)) = 1 ]; then cell b1k-$p "$T1K" 256 0; cell a1k-$p "$T1K" 256 1
    else cell a1k-$p "$T1K" 256 1; cell b1k-$p "$T1K" 256 0; fi
done
echo "== 32K x2 and 128K x2 =="
for p in 1 2; do
    if [ $((p % 2)) = 1 ]; then cell b32k-$p "$T32K" 128 0; cell a32k-$p "$T32K" 128 1
    else cell a32k-$p "$T32K" 128 1; cell b32k-$p "$T32K" 128 0; fi
    if [ $((p % 2)) = 1 ]; then cell b128k-$p "$T128K" 64 0; cell a128k-$p "$T128K" 64 1
    else cell a128k-$p "$T128K" 64 1; cell b128k-$p "$T128K" 64 0; fi
done
echo "== verdict =="
python3 - "$STAMP" <<'PY'
import glob, re, statistics, sys
stamp = sys.argv[1]; root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def cell(tag):
    f = glob.glob(f"{root}/r30i-cell-{tag}-{stamp}.log")
    if not f: return None
    t = open(f[0]).read()
    d = re.search(r"decode +(\d+) tokens in ([\d.]+) ms +-> +([\d.]+) tok/s", t)
    r = re.search(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+)", t)
    if not (d and r): return None
    return dict(tps=float(d.group(3)), ms=float(d.group(2)), rounds=int(r.group(1)),
                accepted=int(r.group(3)), offered=int(r.group(4)))
def report(label, n, need_med, need_wins):
    rows = []
    for i in range(1, n + 1):
        b, a = cell(f"b{label}-{i}"), cell(f"a{label}-{i}")
        if not (b and a): print(f"{label}: INCOMPLETE pair {i}"); return False
        rows.append((b, a))
    d_tps = [100 * (a["tps"] - b["tps"]) / b["tps"] for b, a in rows]
    d_msr = [100 * ((a["ms"] / a["rounds"]) - (b["ms"] / b["rounds"])) / (b["ms"] / b["rounds"]) for b, a in rows]
    med = statistics.median(d_tps); med_msr = statistics.median(d_msr); wins = sum(1 for x in d_tps if x > 0)
    ok = med >= need_med and wins >= need_wins and med_msr <= 2.0
    print(f"{label}: base {statistics.median([b['tps'] for b, _ in rows]):.2f} -> arm "
          f"{statistics.median([a['tps'] for _, a in rows]):.2f} tok/s | d_tps " + " ".join(f"{x:+.2f}%" for x in d_tps)
          + f" med {med:+.2f}% wins {wins}/{n} | d_ms/round " + " ".join(f"{x:+.2f}%" for x in d_msr)
          + f" med {med_msr:+.2f}% | accepted base {[b['accepted'] for b, _ in rows]} arm {[a['accepted'] for _, a in rows]}"
          + f" -> {'PASS' if ok else 'FAIL'}")
    return ok
ok1 = report("1k", 5, 2.0, 4)
ok32 = report("32k", 2, -1.0, 0)
ok128 = report("128k", 2, -1.0, 0)
print(f"CONFIRMATION: {'ADOPT' if (ok1 and ok32 and ok128) else 'DO NOT ADOPT'} (battery already passed 7/8 = 7/8)")
PY
echo "== done (trap restores 8081) =="
