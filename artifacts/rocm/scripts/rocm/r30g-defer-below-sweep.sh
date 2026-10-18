#!/bin/bash
# r30g-defer-below-sweep.sh - item 3: bound STRATA_VERIFY_DEFER_CPU_BELOW (2026-10-18)
#
# The #2690 port (window D) was rejected at the default BELOW=0.1: 1K -11.41%, 32K -3.57%,
# because the output change cost drafter acceptance, while the 32K ROUND was -15.5% ms/round.
# Smaller BELOW defers fewer (lighter) rows -> less divergence, less gain. Bounded sweep:
#   BELOW in {0.05, 0.02}, 3 interleaved 1K pairs each; the better median gets pairs 4-5;
#   then 32K x2. Gate: @1K median >= +2.0% with >= 4/5 wins (after the added pairs), @32K median
#   >= 0% and no pair below -2%; quality battery (A-B-A) only if the throughput gate passes.
# Binary: strata-r146/build-0146 (the #2690 port; default path untouched).
# Both arms run the now-live ROUTE_PRIOR=0.5 (adopted just before this sweep), so the question is
# "does a smaller BELOW add anything on top of the live config": base = live config, arm = live + defer.
set -u
WT=/home/zacch/projects/strata-r146
BUILD=build-0146
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r30g-below-$STAMP.log
T1K=$ROOT/artifacts/rocm/workloads/qsa-profile-1024.tokens
T32K=$ROOT/artifacts/rocm/workloads/qsa-profile-32768.tokens
mkdir -p "$ROOT/artifacts/rocm/logs"
exec > >(tee -a "$LOG") 2>&1
restore_8081() {
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
[ -x "$EXE" ] || { echo "FATAL: build-0146 missing"; exit 9; }
"$EXE" --version
echo "== 1. stop 8081 + drain =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "8081: $(systemctl --user is-active strata-8081.service); MemAvailable $(memav) MiB; VRAM $(vram) MiB"

cell() {  # cell <tag> <tok> <new> <defer 0|1> <below|0>
    local tag="$1" tok="$2" new="$3" defer="$4" below="$5"
    local OUT=$ROOT/artifacts/rocm/logs/r30g-cell-$tag-$STAMP.log
    local envs=(STRATA_VERIFY_DEFER_CPU=$([ "$defer" = "1" ] && echo 1 || echo 0))
    [ "$defer" = "1" ] && [ "$below" != "0" ] && envs+=(STRATA_VERIFY_DEFER_CPU_BELOW=$below)
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_PRIOR=0.5 "${envs[@]}" \
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

echo "== 2. screen BELOW at 1K (3 pairs each; base = defer off) =="
for p in 1 2 3; do
    for b in 0.05 0.02; do
        cell b$b-1k-$p "$T1K" 256 1 $b
        cell base-1k-$p "$T1K" 256 0 0
    done
done
python3 - "$STAMP" > /tmp/r30g-best-$STAMP.txt <<'PY'
import glob, re, statistics, sys
stamp = sys.argv[1]; root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def tps(tag):
    f = glob.glob(f"{root}/r30g-cell-{tag}-{stamp}.log")
    if not f: return None
    m = re.search(r"decode +\d+ tokens in [\d.]+ ms +-> +([\d.]+) tok/s", open(f[0]).read())
    return float(m.group(1)) if m else None
best, bestmed = "0.05", -1e9
for b in ("0.05", "0.02"):
    ds = []
    for p in (1, 2, 3):
        base, arm = tps(f"base-1k-{p}"), tps(f"b{b}-1k-{p}")
        if base and arm: ds.append(100 * (arm - base) / base)
    med = statistics.median(ds) if ds else -1e9
    print(f"# BELOW {b}: median {med:+.2f}% over {len(ds)} pairs")
    if med > bestmed + 0.05: best, bestmed = b, med
print(best)
PY
grep '^#' /tmp/r30g-best-$STAMP.txt; BEST=$(tail -1 /tmp/r30g-best-$STAMP.txt); echo "BEST BELOW: $BEST"
for p in 4 5; do cell base-1k-$p "$T1K" 256 0 0; cell b$BEST-1k-$p "$T1K" 256 1 "$BEST"; done
for p in 1 2; do cell base-32k-$p "$T32K" 128 0 0; cell b$BEST-32k-$p "$T32K" 128 1 "$BEST"; done

echo "== 3. verdict =="
python3 - "$STAMP" "$BEST" <<'PY'
import glob, re, statistics, sys
stamp, best = sys.argv[1], sys.argv[2]
root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def cell(tag):
    f = glob.glob(f"{root}/r30g-cell-{tag}-{stamp}.log")
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
        if not (b and a): print(f"{label}: INCOMPLETE {bt}/{at}"); return False
        rows.append((b, a))
    d_tps = [100 * (a["tps"] - b["tps"]) / b["tps"] for b, a in rows]
    d_msr = [100 * ((a["ms"] / a["rounds"]) - (b["ms"] / b["rounds"])) / (b["ms"] / b["rounds"]) for b, a in rows]
    med = statistics.median(d_tps); wins = sum(1 for x in d_tps if x > 0)
    ok = med >= need_med and wins >= need_wins and all(x > -2.0 for x in d_msr)
    print(f"{label}: base {statistics.median([b['tps'] for b, _ in rows]):.2f} -> arm "
          f"{statistics.median([a['tps'] for _, a in rows]):.2f} | d_tps " + " ".join(f"{x:+.2f}%" for x in d_tps)
          + f" med {med:+.2f}% wins {wins}/{len(rows)} | ms/round " + " ".join(f"{x:+.2f}%" for x in d_msr)
          + f" | acceptance base {[b['accepted'] for b, _ in rows]} arm {[a['accepted'] for _, a in rows]}"
          + f" -> {'PASS' if ok else 'FAIL'}")
    return ok
ok1 = report("1k", [f"base-1k-{i}" for i in (1, 2, 3, 4, 5)], [f"b{best}-1k-{i}" for i in (1, 2, 3, 4, 5)], 2.0, 4)
ok32 = report("32k", ["base-32k-1", "base-32k-2"], [f"b{best}-32k-1", f"b{best}-32k-2"], 0.0, 0)
print(f"WINDOW VERDICT: {'ADOPT-CANDIDATE (battery next)' if (ok1 and ok32) else 'NOT ADOPTED'} (BELOW {best})")
PY
echo "== done (trap restores 8081) =="
