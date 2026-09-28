#!/bin/sh
# strata-rocm doctor - read-only diagnostics.
#
# Reports ready / blocked / unverified for every prerequisite the rollout
# depends on. Never installs, never modifies system state, never needs root.
#
# Two facts this machine forces us to keep separate:
#   * a device node can be missing because the driver is absent, or because the
#     sandbox this shell runs in mounts its own minimal /dev. Those are
#     different problems with different fixes, so they are different statuses.
#   * rocm-smi reads sysfs and works even when /dev/kfd is out of reach; it is
#     evidence about the card, not about our ability to run on it.
#
# Usage: strata-rocm doctor [--json] [--manifest-only]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

JSON_OUT=0
MANIFEST_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --json) JSON_OUT=1 ;;
        --manifest-only) MANIFEST_ONLY=1 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $arg" $EX_USAGE ;;
    esac
done

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG="$LOGS_DIR/doctor-$TS.log"
LOG_FILE="$RUN_LOG"
: >"$RUN_LOG" 2>/dev/null || true

CHECKS_FILE=$(mktemp)
ENV_FILE=$(mktemp)
trap 'rm -f "$CHECKS_FILE" "$ENV_FILE"' EXIT INT TERM

emit() {  # emit <id> <status> <detail>
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$CHECKS_FILE"
}

envkv() { # envkv <key> <value>
    printf '%s\t%s\n' "$1" "$2" >>"$ENV_FILE"
}

log "doctor start $(date -u +%FT%TZ)"

# ---- 1. operating system ----------------------------------------------------
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    OS_PRETTY=$(. /etc/os-release && printf '%s' "$PRETTY_NAME")
    OS_ID=$(. /etc/os-release && printf '%s %s' "$ID" "$VERSION_ID")
    emit os ready "$OS_PRETTY ($OS_ID), kernel $(uname -r)"
else
    emit os unverified "/etc/os-release is unreadable"
fi
envkv os "$(uname -s) $(uname -r) $(uname -m)"

# ---- 2. cpu / ram -----------------------------------------------------------
NPROC=$(nproc 2>/dev/null || echo 0)
MEM_MIB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$NPROC" -gt 0 ] && [ "$MEM_MIB" -gt 0 ]; then
    emit cpu_ram ready "${NPROC} threads, ${MEM_MIB} MiB RAM visible to this shell"
else
    emit cpu_ram blocked "could not read nproc/MemTotal"
fi
envkv cpu_threads "$NPROC"
envkv mem_total_mib "$MEM_MIB"
envkv cpu_model "$(awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null | sed 's/^ //')"

# ---- 3. disk ----------------------------------------------------------------
FREE_MIB=$(free_mib "$ROCM_ROOT")
FREE_MIB=${FREE_MIB:-0}
if [ "$FREE_MIB" -ge 20480 ]; then
    emit disk ready "${FREE_MIB} MiB free on $ROCM_ROOT (>= 20 GiB)"
elif [ "$FREE_MIB" -ge 5120 ]; then
    emit disk unverified "${FREE_MIB} MiB free on $ROCM_ROOT: enough to build, not for a full-model download"
else
    emit disk blocked "${FREE_MIB} MiB free on $ROCM_ROOT: too little to build"
fi
envkv disk_free_mib "$FREE_MIB"

# ---- 4. GPU identity (sysfs + rocm-smi: works without /dev access) ----------
GPU_PCI=""
if have lspci; then
    GPU_PCI=$(lspci -nn 2>/dev/null | grep -i 'vga\|3d controller' | grep -i 'amd\|ati' | head -1)
fi
GFX_VER=""
if have rocm-smi; then
    GFX_VER=$(timeout 30 rocm-smi --showproductname 2>/dev/null | sed -n 's/.*GFX Version:[[:space:]]*//p' | tr -d '[:space:]' | head -1)
fi
# `Card Series` is blank on this box, so the PCI description is the identity
# that actually exists. The device id is what distinguishes a 7900 XTX (0x744c)
# from its siblings, so it is reported rather than assumed.
GPU_NAME=$(printf '%s' "$GPU_PCI" | sed -n 's/.*\[AMD\/ATI\] \(.*\) \[[0-9a-f]*:[0-9a-f]*\].*/\1/p')
[ -n "$GPU_NAME" ] || GPU_NAME="AMD GPU"
GPU_DEVID=$(printf '%s' "$GPU_PCI" | grep -o '\[[0-9a-f]\{4\}:[0-9a-f]\{4\}\]' | head -1)
if [ -n "$GPU_PCI" ]; then
    emit gpu_identity ready "$GPU_NAME | $GPU_DEVID | gfx=${GFX_VER:-unknown} | $(printf '%s' "$GPU_PCI" | awk '{print $1}')"
else
    emit gpu_identity unverified "no AMD display device found in lspci"
fi
envkv gpu_pci "$GPU_PCI"
envkv gpu_gfx "$GFX_VER"
envkv gpu_name "$GPU_NAME"

# ---- 5. GPU device nodes (the sandbox-sensitive one) ------------------------
SYSFS_RENDER=""
for d in /sys/class/drm/renderD*; do
    [ -e "$d" ] && SYSFS_RENDER="$d" && break
done
DEV_OK=yes
[ -e /dev/kfd ] || DEV_OK=no
[ -e "$SYSFS_RENDER" ] || true
for d in /dev/dri/renderD*; do [ -e "$d" ] || DEV_OK=no; done

if [ "$DEV_OK" = yes ]; then
    emit gpu_devices ready "/dev/kfd and a render node are present in this namespace"
elif [ -n "$SYSFS_RENDER" ]; then
    emit gpu_devices blocked \
        "device nodes hidden by this shell's sandbox namespace: $SYSFS_RENDER exists but /dev/kfd and /dev/dri are absent. Not a driver problem. Re-run GPU commands outside the sandbox (DSH: danger-full-access); see PORTING.md 'Running GPU work'."
else
    emit gpu_devices blocked "/dev/kfd absent and no DRM render node in sysfs: amdgpu/kfd is not up"
fi
envkv gpu_devices_visible "$DEV_OK"

# ---- 6. kernel driver -------------------------------------------------------
if lsmod 2>/dev/null | grep -q '^amdgpu'; then
    KFD=$(lsmod 2>/dev/null | awk '$1=="amdgpu" {print $3}')
    emit kernel_driver ready "amdgpu loaded (refcount $KFD)"
else
    if [ -n "$SYSFS_RENDER" ]; then
        emit kernel_driver ready "amdgpu present via sysfs (lsmod unreadable in this namespace)"
    else
        emit kernel_driver blocked "amdgpu module is not loaded"
    fi
fi
envkv amdgpu_params "$(tr ' ' '\n' </proc/cmdline 2>/dev/null | grep -i amdgpu | tr '\n' ' ')"

# ---- 7. ROCm / HIP toolchain -----------------------------------------------
if have hipcc; then
    HIP_FULL=$(hipcc --version 2>/dev/null | tr '\n' ' ')
    HIP_VER=$(printf '%s' "$HIP_FULL" | sed -n 's/.*HIP version: *\([^ ]*\).*/\1/p')
    GFX_LIB=""
    for p in /usr/lib64/rocm/gfx1100 /usr/lib64/rocm/amdgcn/bitcode /opt/rocm/amdgcn/bitcode; do
        [ -e "$p" ] && GFX_LIB="$p" && break
    done
    if [ -d /usr/lib64/rocm/gfx1100 ] || ls /usr/lib64/rocm/amdgcn/bitcode/gfx1100* >/dev/null 2>&1; then
        emit rocm_runtime ready "hipcc HIP ${HIP_VER:-unknown}, gfx1100 device code present"
    else
        emit rocm_runtime unverified "hipcc HIP ${HIP_VER:-unknown}, but no gfx1100 device libraries found"
    fi
else
    emit rocm_runtime blocked "hipcc not found on PATH"
fi
envkv hip_version "${HIP_VER:-}"
envkv rocm_lib_dir "$(ls -d /usr/lib64/rocm /opt/rocm 2>/dev/null | head -1)"

# ---- 8. build tools ---------------------------------------------------------
MISSING=""
have cmake || MISSING="$MISSING cmake"
have ninja || MISSING="$MISSING ninja"
have clang++ || MISSING="$MISSING clang++"
have make || MISSING="$MISSING make"
have python3 || MISSING="$MISSING python3"
have perl || MISSING="$MISSING perl"
have git || MISSING="$MISSING git"
if [ -z "$MISSING" ]; then
    CM_VER=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
    emit build_tools ready "cmake $CM_VER, $(clang++ --version 2>/dev/null | head -1), python $(python3 -V 2>&1 | awk '{print $2}')"
else
    emit build_tools blocked "missing:$MISSING"
fi
envkv cmake_version "$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')"
envkv clang_version "$(clang++ --version 2>/dev/null | head -1)"

# ---- 9. HIPIFY --------------------------------------------------------------
if [ -f "$VENDOR_DIR/hipify-perl" ]; then
    emit hipify ready "vendored hipify-perl $(sha256_of "$VENDOR_DIR/hipify-perl" | cut -c1-12)"
elif have hipify-clang; then
    emit hipify ready "system hipify-clang"
elif have hipify-perl; then
    emit hipify ready "system hipify-perl"
else
    emit hipify unverified "no HIPIFY available; ./strata-rocm prepare vendors a pinned hipify-perl"
fi

# ---- 10. network ------------------------------------------------------------
# One probe, its result reused: a second ls-remote doubles the wait for no
# extra information.
NET_UP=0
if have git && timeout 45 git ls-remote https://github.com/Niko1221/Strata HEAD >/dev/null 2>&1; then
    NET_UP=1
    emit network ready "github.com reachable"
else
    emit network unverified "cannot reach github.com (offline work still possible)"
fi
envkv network "$([ "$NET_UP" = 1 ] && echo up || echo down)"

# ---- 11. ports and the service we must not disturb --------------------------
PORT=8081
if have ss; then
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$PORT\$"; then
        emit port_8081 blocked "port $PORT already in use; pick another with STRATA_PORT"
    else
        emit port_8081 ready "port $PORT is free (loopback target)"
    fi
    EXISTING=$(ss -ltn 2>/dev/null | awk '{print $4}' | grep -cE '[:.]8080$' || true)
else
    emit port_8081 unverified "ss not available; cannot check port $PORT"
    EXISTING=0
fi
if [ "${EXISTING:-0}" -gt 0 ]; then
    MODELS=$(timeout 10 curl -fsS http://127.0.0.1:8080/v1/models 2>/dev/null \
             | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
print(",".join(m.get("name","?") for m in d.get("models",[])))' 2>/dev/null)
    emit existing_service ready "server on 8080 must keep running (model: ${MODELS:-unknown}); the port-8081 launch does not disturb it"
else
    emit existing_service ready "no server on 8080; nothing to preserve"
fi
envkv existing_server_8080 "${EXISTING:-0}"

# ---- 12. VRAM (informational) ----------------------------------------------
if have rocm-smi; then
    VRAM=$(timeout 30 rocm-smi --showmeminfo vram 2>/dev/null | awk '/VRAM Total Used Memory/ {u=$NF} /VRAM Total Memory/ {t=$NF} END {if (t!="") printf "%s/%s", u, t}')
    emit vram ready "VRAM used/total bytes: ${VRAM:-unknown} (an existing model server holds most of it; do not free it)"
    envkv vram_used_total "$VRAM"
fi

# ---- 13. repository state ---------------------------------------------------
if [ -d "$STRATA_DIR/.git" ]; then
    REV=$(strata_revision)
    DIRTY=$(strata_dirty)
    COUNT=$(git -C "$STRATA_DIR" rev-list --count HEAD 2>/dev/null || echo "?")
    if [ "$DIRTY" = "true" ]; then
        emit repository unverified "checkout at $REV has uncommitted tracked changes; evidence must name the patch"
    else
        emit repository ready "clean checkout at $REV ($COUNT commits)"
    fi
else
    emit repository blocked "no git checkout at $STRATA_DIR"
fi
envkv strata_revision "$(strata_revision)"
envkv strata_dirty "$(strata_dirty)"

# ---- assemble ---------------------------------------------------------------
BLOCKED=$(awk -F'\t' '$2=="blocked"' "$CHECKS_FILE" | wc -l)
UNVERIFIED=$(awk -F'\t' '$2=="unverified"' "$CHECKS_FILE" | wc -l)
READY=$(awk -F'\t' '$2=="ready"' "$CHECKS_FILE" | wc -l)

python3 - "$CHECKS_FILE" "$ENV_FILE" "$MANIFEST_FILE" "$RESULTS_DIR/doctor-$TS.json" \
         "$TS" "$BLOCKED" "$UNVERIFIED" "$READY" <<'PY'
import json, os, sys
checks_f, env_f, manifest_f, result_f, ts, blocked, unver, ready = sys.argv[1:9]

checks = []
for line in open(checks_f):
    line = line.rstrip("\n")
    if not line:
        continue
    cid, status, detail = line.split("\t", 2)
    checks.append({"id": cid, "status": status, "detail": detail})

env = {}
for line in open(env_f):
    line = line.rstrip("\n")
    if not line:
        continue
    k, v = line.split("\t", 1)
    env[k] = v

overall = "blocked" if int(blocked) else ("unverified" if int(unver) else "ready")
manifest = {"schema": 1, "at": ts, "overall": overall, "environment": env, "checks": checks}
os.makedirs(os.path.dirname(manifest_f), exist_ok=True)
with open(manifest_f, "w") as fh:
    json.dump(manifest, fh, indent=2); fh.write("\n")

result = dict(manifest)
result["counts"] = {"ready": int(ready), "blocked": int(blocked), "unverified": int(unver)}
with open(result_f, "w") as fh:
    json.dump(result, fh, indent=2); fh.write("\n")

PY

if [ "$JSON_OUT" = 1 ]; then
    cat "$RESULTS_DIR/doctor-$TS.json"
    printf '\n'
else
    printf '\nStrata ROCm doctor  (%s)\n' "$TS"
    printf '%s\n' "------------------------------------------------------------"
    awk -F'\t' '{printf "%-18s %-11s %s\n", $1, $2, $3}' "$CHECKS_FILE"
    printf '%s\n' "------------------------------------------------------------"
    printf 'ready %s   blocked %s   unverified %s\n' "$READY" "$BLOCKED" "$UNVERIFIED"
    [ "$BLOCKED" -gt 0 ] && printf 'overall: BLOCKED - see the blocked lines above\n' || printf 'overall: usable (%s unverified)\n' "$UNVERIFIED"
fi

state_set environment "$(python3 -c 'import json,sys;print(json.dumps(json.load(open(sys.argv[1]))["environment"]))' "$MANIFEST_FILE")"
state_set next_action '"resolve blocked checks, then ./strata-rocm prepare"'

[ "$BLOCKED" -gt 0 ] && exit $EX_BLOCKED
exit $EX_OK
