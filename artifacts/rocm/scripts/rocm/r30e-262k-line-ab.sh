#!/bin/bash
# r30e-262k-line-ab.sh - item 1: is the 262K decode drop 0.1.43.1 -> 0.1.45 real? (2026-10-18)
#
# The IQ3_S campaign measured 262K decode at 56.5 tok/s on 0.1.45, while the R1292 sweep
# recorded 69.3 on the 0.1.43.1 line (not a like-for-like config). This window runs the two
# engine BINARIES on the same workload, same args, same env, interleaved:
#   A = /home/zacch/projects/strata-r143/build-0143/strata  0.1.43.1-8a15c99c
#   B = /home/zacch/projects/strata-r145/build-0145/strata  0.1.45-8eb3e85d
# 3 pairs, 261,888-token prompt, --max-context 262144, 64 new tokens, production env
# (SELECT_WMMA, HIP_WMMA, SH_STREAM, TAIL_SKIP=7, lookup-chain 2, spec 5).
#
# Pre-declared reading: the question is whether B is materially slower than A at 262K.
#   |median paired delta| < 3%  -> the 69.3-vs-56.5 gap is CONFIG, not the line (close the row)
#   B slower by >= 3%           -> a LINE REGRESSION; the phase split (prefill vs decode, and the
#                                  decode summary's wait/pool/pcie lines) says where
#   B faster by >= 3%           -> the R1292 config was the difference (also closes the row)
# No adoption decision here; this is a diagnosis window.
set -u
EXP=/home/zacch/strata-rocm-experiments/20261003-231142
ROOT=/home/zacch/projects/strata-amd
A=/home/zacch/projects/strata-r143/build-0143/strata
B=/home/zacch/projects/strata-r145/build-0145/strata
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/r30e-262k-line-$STAMP.log
T262=$ROOT/artifacts/rocm/workloads/qsa-profile-261888.tokens
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

echo "== 0. preflight ($STAMP) =="
for e in "$A" "$B"; do [ -x "$e" ] || { echo "FATAL: $e missing"; exit 9; }; "$e" --version; done
[ -f "$T262" ] || { echo "FATAL: 262K workload missing"; exit 9; }

echo "== 1. stop 8081 + drain =="
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(memav)" -ge 68000 ] && [ "$(vram)" -le 2600 ] && break; sleep 4; done
echo "8081: $(systemctl --user is-active strata-8081.service); MemAvailable $(memav) MiB; VRAM $(vram) MiB"
[ "$(memav)" -ge 68000 ] && [ "$(vram)" -le 2600 ] || { echo "FATAL: drain gates not met"; exit 9; }

cell() {  # cell <exe> <tag>
    local exe="$1" tag="$2"
    local OUT=$ROOT/artifacts/rocm/logs/r30e-cell-$tag-$STAMP.log
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_PREFILL_TIMING=1 \
        systemd-run --user --scope -p MemoryMax=66G \
        "$exe" --tokens-file "$T262" --max-new 64 --max-context 262144 \
        --pack $EXP/packs/iq3_s \
        --native /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        --ple-gguf /home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
        --mtp /mnt/LINUXWINSHARE/models/mtp/rt \
        --expert-profile $EXP/data/expert-profile.bin \
        --expert-cache auto --spec 5 --kv int8 --kv-resident 32768 --prefill 8192 \
        --prompt-cache 6 --spec-min-p 0.70 --vram-reserve-mib 3072 --lookup-chain 2 > "$OUT" 2>&1
    local rc=$? p d acc
    p=$(grep -oE "prefill +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    d=$(grep -oE "decode +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1)
    acc=$(grep -oE "drafts accepted [0-9]+ of [0-9]+" "$OUT" | head -1)
    echo "CELL $tag: rc=$rc prefill=$p decode=$d $acc"
    [ "$rc" = "0" ] || echo "  (failed - $OUT)"
}

echo "== 2. 3 interleaved pairs at 261,888 tokens (A=0.1.43.1, B=0.1.45) =="
cell "$A" a1-0143; cell "$B" b1-0145
cell "$B" b2-0145; cell "$A" a2-0143
cell "$A" a3-0143; cell "$B" b3-0145

echo "== 3. verdict =="
python3 - "$STAMP" <<'PY'
import glob, re, statistics, sys
stamp = sys.argv[1]
root = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"
def cell(tag):
    f = glob.glob(f"{root}/r30e-cell-{tag}-{stamp}.log")
    if not f: return None
    t = open(f[0]).read()
    d = re.search(r"decode +(\d+) tokens in ([\d.]+) ms +-> +([\d.]+) tok/s", t)
    r = re.search(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+)", t)
    p = re.search(r"prefill +(\d+) tokens in \d+ chunks, ([\d.]+) ms \(([\d.]+) tok/s\)", t)
    if not (d and r and p): return None
    w = re.search(r"verify window\s+wait for rings ([\d.]+)\s+pool ([\d.]+)", t)
    pc = re.search(r"pcie experts\s+([\d.]+) distinct", t)
    return dict(tps=float(d.group(3)), ms=float(d.group(2)), rounds=int(r.group(1)),
                accepted=int(r.group(3)), offered=int(r.group(4)),
                prefill=float(p.group(3)), wait=float(w.group(1)) if w else None,
                pool=float(w.group(2)) if w else None,
                pcie=float(pc.group(1)) if pc else None)
pairs = [(f"a{i}-0143", f"b{i}-0145") for i in (1, 2, 3)]
rows = []
for a_tag, b_tag in pairs:
    a, b = cell(a_tag), cell(b_tag)
    if not (a and b):
        print(f"INCOMPLETE: {a_tag}/{b_tag}"); sys.exit(4)
    rows.append((a, b))
print(f"{'pair':>4} {'0143 tok/s':>10} {'0145 tok/s':>10} {'d_tps':>8} | {'0143 ms/round':>13} {'0145 ms/round':>13} {'d_msr':>8} | {'0143 pf':>8} {'0145 pf':>8}")
d_tps, d_msr, d_pf = [], [], []
for i, (a, b) in enumerate(rows, 1):
    dt = 100 * (b["tps"] - a["tps"]) / a["tps"]
    dm = 100 * ((b["ms"] / b["rounds"]) - (a["ms"] / a["rounds"])) / (a["ms"] / a["rounds"])
    dp = 100 * (b["prefill"] - a["prefill"]) / a["prefill"]
    d_tps.append(dt); d_msr.append(dm); d_pf.append(dp)
    print(f"{i:>4} {a['tps']:>10.2f} {b['tps']:>10.2f} {dt:>+7.2f}% | {a['ms']/a['rounds']:>13.2f} {b['ms']/b['rounds']:>13.2f} {dm:>+7.2f}% | {a['prefill']:>8.1f} {b['prefill']:>8.1f}")
med_tps, med_msr = statistics.median(d_tps), statistics.median(d_msr)
print(f"median: d_tps {med_tps:+.2f}% | d_ms/round {med_msr:+.2f}% | prefill deltas " + " ".join(f"{x:+.2f}%" for x in d_pf))
a0, b0 = rows[0]
print(f"decode summary pair1: 0143 wait {a0['wait']} pool {a0['pool']} pcie {a0['pcie']} accepted {a0['accepted']}/{a0['offered']}")
print(f"                     0145 wait {b0['wait']} pool {b0['pool']} pcie {b0['pcie']} accepted {b0['accepted']}/{b0['offered']}")
if abs(med_tps) < 3.0:
    verdict = "CONFIG, not the line (|delta| < 3%): the 69.3-vs-56.5 gap is the config difference"
elif med_tps < -3.0:
    verdict = "LINE REGRESSION: 0.1.45 is slower at 262K on the same config"
else:
    verdict = "0.1.45 FASTER at 262K on the same config (the R1292 config was the difference)"
print(f"READING: {verdict}")
PY
echo "== done (trap restores 8081) =="
