#!/bin/bash
# r31c-t08-sweeps.sh - item 3: T08's remaining sweeps on the live config (2026-10-18)
#   residency: --vram-reserve-mib 2048 / 4096 vs live 3072 (2 pairs each @1K)
#   CPU pool:  --pool-workers 9 / 13 vs auto (0 = 11 here) (2 pairs each @1K)
#   n-gram locality: STRATA_LOOKAHEAD_STATS=1 diagnostic run (one cell, reported only)
# PRE-DECLARED: a value replaces the live one only if its median paired tok/s is >= +2.0% with
# both pairs up and no acceptance change; otherwise the live value stands (the sweep is recorded).
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
WT=/home/zacch/projects/strata-r145; BUILD=build-0145; EXE=$WT/$BUILD/strata
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r31c-t08-$STAMP.log
T1K=$ROOT/artifacts/rocm/workloads/qsa-profile-1024.tokens
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
LIVE=(STRATA_ROUTE_PRIOR=0.5 STRATA_GDN_CHUNKED=1 STRATA_PF_GEMM=1 STRATA_PF_PAD=1)
cell() {  # cell <tag> <reserve> <workers> [extra env...]
    local tag="$1" reserve="$2" workers="$3"; shift 3
    local OUT=$ROOT/artifacts/rocm/logs/r31c-cell-$tag-$STAMP.log
    local wargs=(); [ "$workers" != "0" ] && wargs=(--pool-workers "$workers")
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 "${LIVE[@]}" "$@" \
        systemd-run --user --scope -p MemoryMax=64G \
        "$EXE" --tokens-file "$T1K" --max-new 256 --max-context 139264 \
        --pack $EXP/packs/iq3_s \
        --native /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        --ple-gguf /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
        --mtp /mnt/LINUXWINSHARE/models/mtp/rt \
        --expert-profile $EXP/data/expert-profile.bin \
        --expert-cache auto --spec 5 --kv int8 --kv-resident 32768 --prefill 8192 \
        --prompt-cache 6 --spec-min-p 0.70 --vram-reserve-mib "$reserve" --lookup-chain 2 "${wargs[@]}" > "$OUT" 2>&1
    local rc=$? d acc
    d=$(grep -oE "decode +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    acc=$(grep -oE "drafts accepted [0-9]+ of [0-9]+" "$OUT" | head -1)
    echo "CELL $tag rc=$rc decode=$d $acc"
    [ "$rc" = "0" ] || echo "  (failed - $OUT)"
}
echo "== residency budget (base live 3072) =="
for p in 1 2; do
    cell base-res-p$p 3072 0
    cell res2048-p$p 2048 0
    cell res4096-p$p 4096 0
done
echo "== CPU pool workers (base auto) =="
for p in 1 2; do
    cell base-w-p$p 3072 0
    cell w9-p$p 3072 9
    cell w13-p$p 3072 13
done
echo "== n-gram locality diagnostic =="
cell lookahead 3072 0 STRATA_LOOKAHEAD_STATS=1
grep -a "strata lookahead stats" "$ROOT/artifacts/rocm/logs/r31c-cell-lookahead-$STAMP.log" | tail -2 | cut -c1-300
echo "== verdict =="
python3 - "$STAMP" <<'PY'
import glob, re, statistics, sys
stamp = sys.argv[1]; root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def cell(tag):
    f = glob.glob(f"{root}/r31c-cell-{tag}-{stamp}.log")
    if not f: return None
    t = open(f[0]).read()
    d = re.search(r"decode +(\d+) tokens in [\d.]+ ms +-> +([\d.]+) tok/s", t)
    r = re.search(r"drafts accepted (\d+) of (\d+)", t)
    return dict(tps=float(d.group(2)) if d else None, acc=int(r.group(1)) if r else None)
def cmp(label, base_tag, arm_tag, n):
    ds = []
    for i in range(1, n + 1):
        b, a = cell(base_tag.format(i)), cell(arm_tag.format(i))
        if not (b and a and b["tps"] and a["tps"]): print(f"{label}: INCOMPLETE"); return
        ds.append(100 * (a["tps"] - b["tps"]) / b["tps"])
    print(f"{label}: " + " ".join(f"{x:+.2f}%" for x in ds) + f" | median {statistics.median(ds):+.2f}%")
cmp("reserve 2048 vs 3072", "base-res-p{}", "res2048-p{}", 2)
cmp("reserve 4096 vs 3072", "base-res-p{}", "res4096-p{}", 2)
cmp("workers 9 vs auto", "base-w-p{}", "w9-p{}", 2)
cmp("workers 13 vs auto", "base-w-p{}", "w13-p{}", 2)
PY
echo "== done (trap restores 8081) =="
