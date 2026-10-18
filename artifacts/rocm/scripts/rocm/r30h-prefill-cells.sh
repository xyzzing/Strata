#!/bin/bash
# r30h-prefill-cells.sh - item 5: the prefill half of the board (2026-10-18)
#
# Three isolated cells on build-0145 (0.1.45), production env, one-shot:
#   P1 exact-scorer three-way @128K  (the #1298 question, and ledger item 4)
#      A current: STRATA_SELECT_WMMA=1 (fp16 gfx110x arm)
#      B exact:   STRATA_SELECT_WMMA=0 STRATA_SCORES_THREAD=1 (four-lanes-per-query; same scores)
#      C ref:     STRATA_SELECT_WMMA=0 (default radix)
#   P2 GDN chunked vs serial @32K  (ledger item 2 / R21; upstream chunked WMMA gfx11, output-changing)
#      A default; B STRATA_GDN_CHUNKED=1
#   P3 PF_GEMM / PF_PAD isolate @128K  (ledger item 3; the #1655 bundle)
#      A default; B STRATA_PF_GEMM=1; C STRATA_PF_GEMM=1 STRATA_PF_PAD=1
# Pre-declared: each candidate must beat its base by >= 3% prefill tok/s in the median of its
# pairs to move on (GDN/PAD are output-changing and would then need the battery). No adoption here.
# Resource gate (2026-10-18 later): the desktop baseline grew after the route-prior adoption, so
# "MemAvailable >= 68000 MiB" was met at 67.2-67.7k in two attempts. The cell cgroup cap is lowered
# 66G -> 64G and the gate to 66,000 MiB, keeping >= 2 GiB headroom above the cap; this is a declared
# resource adaptation, not a measurement tolerance. An OOM cell fails and is recorded.
set -u
WT=/home/zacch/projects/strata-r145
BUILD=build-0145
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r30h-prefill-cells-$STAMP.log
T32K=$ROOT/artifacts/rocm/workloads/qsa-profile-32768.tokens
T128K=$ROOT/artifacts/rocm/workloads/qsa-profile-131072.tokens
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
[ -x "$EXE" ] || { echo "FATAL: binary missing"; exit 9; }
"$EXE" --version
echo "== 1. stop 8081 + drain =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(memav)" -ge 66000 ] && [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "8081: $(systemctl --user is-active strata-8081.service); MemAvailable $(memav) MiB (gate 66000); VRAM $(vram) MiB (gate 2600)"
[ "$(memav)" -ge 66000 ] && [ "$(vram)" -le 2600 ] || { echo "FATAL: drain gates not met"; exit 9; }

cell() {  # cell <tag> <tok> <new> [env...]
    local tag="$1" tok="$2" new="$3"; shift 3
    local OUT=$ROOT/artifacts/rocm/logs/r30h-cell-$tag-$STAMP.log
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_PREFILL_TIMING=1 \
        "$@" \
        systemd-run --user --scope -p MemoryMax=64G \
        "$EXE" --tokens-file "$tok" --max-new "$new" --max-context 139264 \
        --pack $EXP/packs/iq3_s \
        --native /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        --ple-gguf /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
        --mtp /mnt/LINUXWINSHARE/models/mtp/rt \
        --expert-profile $EXP/data/expert-profile.bin \
        --expert-cache auto --spec 5 --kv int8 --kv-resident 32768 --prefill 8192 \
        --prompt-cache 6 --spec-min-p 0.70 --vram-reserve-mib 3072 --lookup-chain 2 > "$OUT" 2>&1
    local rc=$? p sel
    p=$(grep -oE "prefill +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    sel=$(grep -oE "qsa select [0-9]+" "$OUT" | head -1)
    echo "CELL $tag rc=$rc prefill=$p ${sel:+| $sel ms}"
    [ "$rc" = "0" ] || echo "  (failed - $OUT)"
}

echo "== P1. exact-scorer three-way @128K (A current, B exact, C ref) =="
# pair 1: A B C ; pair 2: C B A (order reversal against drift)
cell p1a-1 "$T128K" 64
cell p1b-1 "$T128K" 64 STRATA_SELECT_WMMA=0 STRATA_SCORES_THREAD=1
cell p1c-1 "$T128K" 64 STRATA_SELECT_WMMA=0
cell p1c-2 "$T128K" 64 STRATA_SELECT_WMMA=0
cell p1b-2 "$T128K" 64 STRATA_SELECT_WMMA=0 STRATA_SCORES_THREAD=1
cell p1a-2 "$T128K" 64

echo "== P2. GDN chunked vs serial @32K (A default, B chunked) =="
cell p2a-1 "$T32K" 128
cell p2b-1 "$T32K" 128 STRATA_GDN_CHUNKED=1
cell p2b-2 "$T32K" 128 STRATA_GDN_CHUNKED=1
cell p2a-2 "$T32K" 128

echo "== P3. PF_GEMM / PF_PAD isolate @128K (A default, B gemm, C gemm+pad) =="
cell p3a-1 "$T128K" 64
cell p3b-1 "$T128K" 64 STRATA_PF_GEMM=1
cell p3c-1 "$T128K" 64 STRATA_PF_GEMM=1 STRATA_PF_PAD=1
cell p3c-2 "$T128K" 64 STRATA_PF_GEMM=1 STRATA_PF_PAD=1
cell p3b-2 "$T128K" 64 STRATA_PF_GEMM=1
cell p3a-2 "$T128K" 64

echo "== verdict =="
python3 - "$STAMP" <<'PY'
import glob, re, statistics, sys
stamp = sys.argv[1]; root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def cell(tag):
    f = glob.glob(f"{root}/r30h-cell-{tag}-{stamp}.log")
    if not f: return None
    t = open(f[0]).read()
    p = re.search(r"prefill +\d+ tokens in [\d.]+ ms +-> +([\d.]+) tok/s", t)
    s = re.search(r"qsa select (\d+) ms", t)
    return dict(pf=float(p.group(1)) if p else None, sel=int(s.group(1)) if s else None)
def trio(label, tags_a, tags_b, tags_c=None):
    A = [cell(t) for t in tags_a]; B = [cell(t) for t in tags_b]; C = [cell(t) for t in tags_c] if tags_c else None
    if any(x is None for x in A + B + (C or [])): print(f"{label}: INCOMPLETE"); return
    def med(xs, k): 
        v = [x[k] for x in xs if x[k] is not None]
        return statistics.median(v) if v else None
    pa, pb = med(A, "pf"), med(B, "pf")
    print(f"{label}: base {pa:.1f} -> cand {pb:.1f} prefill tok/s ({100*(pb-pa)/pa:+.2f}%)", end="")
    sa, sb = med(A, "sel"), med(B, "sel")
    if sa and sb: print(f" | select {sa} -> {sb} ms ({100*(sb-sa)/sa:+.1f}%)", end="")
    if C:
        pc = med(C, "pf"); sc = med(C, "sel")
        print(f" | ref {pc:.1f} ({100*(pc-pa)/pa:+.2f}% vs base)", end="")
        if sc: print(f", select {sc} ms", end="")
    print()
trio("P1 exact vs current (B/A)", ["p1a-1", "p1a-2"], ["p1b-1", "p1b-2"], ["p1c-1", "p1c-2"])
trio("P1 current vs ref (A/C)", ["p1c-1", "p1c-2"], ["p1a-1", "p1a-2"])
trio("P2 GDN chunked vs serial (B/A)", ["p2a-1", "p2a-2"], ["p2b-1", "p2b-2"])
trio("P3 PF_GEMM vs default (B/A)", ["p3a-1", "p3a-2"], ["p3b-1", "p3b-2"], ["p3c-1", "p3c-2"])
print("GATE: >= +3% prefill moves a candidate on; GDN/PF_PAD are output-changing (battery next if so)")
PY
echo "== done (trap restores 8081) =="
