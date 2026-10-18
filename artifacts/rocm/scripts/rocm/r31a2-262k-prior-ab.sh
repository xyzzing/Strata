#!/bin/bash
# r31a2-262k-prior-ab.sh - does the route prior regress at 262K? (2026-10-18)
# The live-env re-baseline measured 262K decode 45.12 tok/s with acceptance 24/103, while the
# pre-prior 262K cell measured ~56.5 (window C). Cross-window, so this is a paired A/B on the
# same binary and workload: prior 0 vs 0.5, 3 interleaved pairs, everything else live.
# Reading: median paired tok/s <= -3% -> the prior regresses at 262K (a depth trade-off to decide);
# |delta| < 3% -> trajectory noise; >= +3% -> depth win.
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
WT=/home/zacch/projects/strata-r145; BUILD=build-0145; EXE=$WT/$BUILD/strata
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r31a2-262k-prior-$STAMP.log
T262=$ROOT/artifacts/rocm/workloads/qsa-profile-261888.tokens
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
for i in $(seq 1 45); do [ "$(memav)" -ge 66000 ] && [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "MemAvailable $(memav) MiB; VRAM $(vram) MiB"
cell() {  # cell <tag> <prior>
    local tag="$1" prior="$2"
    local OUT=$ROOT/artifacts/rocm/logs/r31a2-cell-$tag-$STAMP.log
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_PRIOR=$prior STRATA_GDN_CHUNKED=1 STRATA_PF_GEMM=1 STRATA_PF_PAD=1 \
        systemd-run --user --scope -p MemoryMax=64G \
        "$EXE" --tokens-file "$T262" --max-new 64 --max-context 262144 \
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
for p in 1 2 3; do
    if [ $((p % 2)) = 1 ]; then cell off-$p 0; cell on-$p 0.5
    else cell on-$p 0.5; cell off-$p 0; fi
done
python3 - "$STAMP" <<'PY'
import glob, re, statistics, sys
stamp = sys.argv[1]; root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def cell(tag):
    f = glob.glob(f"{root}/r31a2-cell-{tag}-{stamp}.log")
    if not f: return None
    t = open(f[0]).read()
    d = re.search(r"decode +(\d+) tokens in ([\d.]+) ms +-> +([\d.]+) tok/s", t)
    r = re.search(r"drafts accepted (\d+) of (\d+)", t)
    if not (d and r): return None
    return dict(tps=float(d.group(3)), acc=int(r.group(1)), off=int(r.group(2)), ms=float(d.group(2)))
rows = []
for p in (1, 2, 3):
    b, a = cell(f"off-{p}"), cell(f"on-{p}")
    if not (b and a): print(f"pair {p}: INCOMPLETE"); sys.exit(4)
    rows.append((b, a))
d_tps = [100 * (a["tps"] - b["tps"]) / b["tps"] for b, a in rows]
print("262K paired: " + " ".join(f"{x:+.2f}%" for x in d_tps) + f" | median {statistics.median(d_tps):+.2f}%")
print("off tps " + " ".join(f"{b['tps']:.2f}" for b, _ in rows) + " | on tps " + " ".join(f"{a['tps']:.2f}" for _, a in rows))
print("acceptance off " + " ".join(f"{b['acc']}/{b['off']}" for b, _ in rows) + " | on " + " ".join(f"{a['acc']}/{a['off']}" for _, a in rows))
med = statistics.median(d_tps)
print("READING:", "prior regresses at 262K" if med <= -3 else ("trajectory noise" if abs(med) < 3 else "prior wins at 262K"))
PY
echo "== done (trap restores 8081) =="
