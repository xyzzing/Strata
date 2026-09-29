#!/bin/sh
# strata-rocm prepare - idempotent, unprivileged local setup.
#
# What it does, and nothing else:
#   * creates the rollout's own directories
#   * vendors one pinned copy of HIPIFY's hipify-perl, with its hash recorded
#   * creates a project-local python venv for the reference implementations
#
# What it never does: sudo, dnf, pip --user, editing /etc, touching the
# existing model server, or writing inside the Strata checkout.
#
# Usage: strata-rocm prepare [--offline] [--force]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

OFFLINE=0
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --offline) OFFLINE=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" $EX_USAGE ;;
    esac
done

# HIPIFY's hipify-perl, pinned by commit AND by content hash. The commit pins
# what to fetch; the hash pins what we accept: a re-download that disagrees with
# the known-good bytes fails instead of silently replacing the translator that
# produced every recorded tree.
HIPIFY_COMMIT=5038c162dec23ffe39617d0ac07500d3014f791e
HIPIFY_SHA256=bd4fcfd286bcbd1511c2f8719e585ee417cddebd43927fd2906f75dfef8a8415
HIPIFY_URL="https://raw.githubusercontent.com/ROCm/HIPIFY/$HIPIFY_COMMIT/bin/hipify-perl"
VENDOR_MANIFEST="$VENDOR_DIR/MANIFEST.json"
VENV="${STRATA_ROCM_VENV:-$ROCM_ROOT/.venv-rocm}"

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOGS_DIR/prepare-$TS.log"
: >"$LOG_FILE"

log "prepare: directories"
mkdir -p "$ART_DIR" "$ART_DIR/env" "$ART_DIR/logs" "$ART_DIR/results" "$ART_DIR/smoke" \
         "$BUILD_DIR" "$TESTS_DIR" "$VENDOR_DIR" "$ROCM_ROOT/tests/rocm/ref" || \
    die "could not create rollout directories" $EX_FAIL

state_init >/dev/null

# ---- 1. vendored HIPIFY -----------------------------------------------------
install_hipify() {
    log "prepare: vendoring hipify-perl @ $HIPIFY_COMMIT"
    if [ "$OFFLINE" = 1 ]; then
        warn "offline: cannot vendor hipify-perl"
        return 1
    fi
    tmp="$VENDOR_DIR/.hipify-perl.tmp"
    if ! curl -fsSL --max-time 120 "$HIPIFY_URL" -o "$tmp"; then
        rm -f "$tmp"
        warn "hipify-perl download failed: $HIPIFY_URL"
        return 1
    fi
    if ! head -1 "$tmp" | grep -q 'perl'; then
        rm -f "$tmp"
        warn "downloaded hipify-perl does not look like a perl script"
        return 1
    fi
    GOT_SHA=$(sha256_of "$tmp")
    if [ "$GOT_SHA" != "$HIPIFY_SHA256" ]; then
        rm -f "$tmp"
        warn "downloaded hipify-perl hash mismatch: got $GOT_SHA, want $HIPIFY_SHA256"
        warn "the pinned commit's content changed upstream or the fetch was tampered with; refusing to install"
        return 1
    fi
    mv "$tmp" "$VENDOR_DIR/hipify-perl"
    chmod 755 "$VENDOR_DIR/hipify-perl"
    SIZE=$(wc -c <"$VENDOR_DIR/hipify-perl" | tr -d ' ')
    SHA=$(sha256_of "$VENDOR_DIR/hipify-perl")
    cat >"$VENDOR_MANIFEST" <<EOF
{
  "tool": "hipify-perl",
  "source": "$HIPIFY_URL",
  "upstream": "https://github.com/ROCm/HIPIFY",
  "commit": "$HIPIFY_COMMIT",
  "license": "MIT",
  "bytes": $SIZE,
  "sha256": "$SHA",
  "vendored_at": "$TS",
  "note": "Vendored so the translator is pinned by hash rather than by whatever a distro ships. Never edited by hand; re-run prepare --force to replace it."
}
EOF
    state_budget download_bytes "$SIZE" >/dev/null
    log "prepare: hipify-perl installed ($SIZE bytes, sha256 ${SHA%"${SHA#????????????}"})"
}

if [ -f "$VENDOR_MANIFEST" ] && [ "$FORCE" = 0 ]; then
    CUR=$(sha256_of "$VENDOR_DIR/hipify-perl")
    WANT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$VENDOR_MANIFEST" 2>/dev/null || echo "")
    if [ "$CUR" = "$WANT" ] && [ -n "$WANT" ]; then
        log "prepare: hipify-perl already vendored and hash matches"
    else
        warn "vendored hipify-perl hash mismatch; reinstalling"
        install_hipify || true
    fi
else
    install_hipify || warn "HIPIFY unavailable; hipify step will be blocked"
fi

# ---- 1b. the pinned llama.cpp sources ---------------------------------------
# `src/prefill/{kernels,moe_mmq}.cu` include llama.cpp's ggml-cuda headers
# (common.cuh, mmq.cuh, quantize.cuh) and the CPU expert path includes
# ggml-cpu.h/ggml.h. Upstream does not vendor them: setup.py downloads exactly
# this archive at exactly this commit. Same dependency, same pin, fetched
# unprivileged into a regenerable directory and hashed. MIT licensed.
#
# Those headers are also HIP-aware (vendors/hip.h, GGML_USE_HIP branches), which
# is why reusing them beats rewriting ggml's MMQ path by hand.
LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
LLAMA_URL="https://github.com/ggml-org/llama.cpp/archive/$LLAMA_COMMIT.zip"
LLAMA_DIR="$BUILD_DIR/llama.cpp"
LLAMA_MANIFEST="$ART_DIR/env/llama-cpp.json"

install_llama_cpp() {
    log "prepare: fetching llama.cpp @ $LLAMA_COMMIT"
    if [ "$OFFLINE" = 1 ]; then
        warn "offline: cannot fetch llama.cpp; prefill MMQ and the CPU expert path stay unbuildable"
        return 1
    fi
    tmpzip="$VENDOR_DIR/llama-cpp-$LLAMA_COMMIT.zip"
    if [ ! -s "$tmpzip" ]; then
        if ! curl -fsSL --max-time 600 "$LLAMA_URL" -o "$tmpzip.part"; then
            rm -f "$tmpzip.part"
            warn "llama.cpp download failed: $LLAMA_URL"
            return 1
        fi
        mv "$tmpzip.part" "$tmpzip"
        state_budget download_bytes "$(wc -c <"$tmpzip" | tr -d ' ')" >/dev/null
    fi
    rm -rf "$LLAMA_DIR"
    mkdir -p "$LLAMA_DIR"
    # python's zipfile rather than unzip: stdlib, and it can strip the archive's
    # top-level directory in one pass. Refuses path-traversal entries (zip slip)
    # explicitly: extraction must not be able to write outside $LLAMA_DIR.
    if ! python3 - "$tmpzip" "$LLAMA_DIR" <<'PYEOF'
import os, sys, zipfile
zpath, dest = sys.argv[1], sys.argv[2]
dest = os.path.abspath(dest)
with zipfile.ZipFile(zpath) as z:
    roots = {n.split("/")[0] for n in z.namelist()}
    if len(roots) != 1:
        raise SystemExit("unexpected archive layout: " + ",".join(sorted(roots)))
    root = roots.pop() + "/"
    for m in z.infolist():
        if not m.filename.startswith(root) or m.is_dir():
            continue
        rel = m.filename[len(root):]
        if rel.startswith("/") or ".." in rel.split("/"):
            raise SystemExit("unsafe path in archive: " + m.filename)
        out = os.path.abspath(os.path.join(dest, rel))
        if not out.startswith(dest + os.sep):
            raise SystemExit("path escapes extraction dir: " + m.filename)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(m) as src, open(out, "wb") as fh:
            fh.write(src.read())
PYEOF
    then
        warn "could not extract $tmpzip"
        return 1
    fi
    cat >"$LLAMA_MANIFEST" <<EOF
{
  "component": "llama.cpp (ggml) sources",
  "pinned_by": "Strata/setup.py LLAMA_CPP_COMMIT",
  "commit": "$LLAMA_COMMIT",
  "url": "$LLAMA_URL",
  "archive_bytes": $(wc -c <"$tmpzip" | tr -d ' '),
  "archive_sha256": "$(sha256_of "$tmpzip")",
  "license": "MIT",
  "extracted_to": "build-hip/llama.cpp",
  "used_for": ["src/prefill/kernels.cu", "src/prefill/moe_mmq.cu", "src/kernels/cpu/native_expert.cpp"],
  "note": "Regenerable: delete build-hip/ and re-run prepare. The archive is kept under scripts/rocm/vendor/ so the fetch happens once."
}
EOF
    log "prepare: llama.cpp extracted to $LLAMA_DIR"
}

if [ -d "$LLAMA_DIR/ggml" ] && [ "$FORCE" = 0 ]; then
    log "prepare: llama.cpp sources already present"
else
    install_llama_cpp || warn "pinned llama.cpp sources unavailable; prefill MMQ stays blocked"
fi

# ---- 2. project-local python venv ------------------------------------------
if [ -x "$VENV/bin/python" ] && [ "$FORCE" = 0 ]; then
    log "prepare: venv already present at $VENV"
else
    log "prepare: creating venv at $VENV"
    if python3 -m venv "$VENV" >>"$LOG_FILE" 2>&1; then
        log "prepare: venv created"
    else
        warn "python3 -m venv failed (ensurepip missing?); see $LOG_FILE"
    fi
fi

if [ -x "$VENV/bin/python" ]; then
    if "$VENV/bin/python" -c 'import numpy' 2>/dev/null; then
        log "prepare: numpy already available in the venv"
    elif [ "$OFFLINE" = 1 ]; then
        warn "offline: numpy not installed; reference tests using numpy will be skipped"
    else
        log "prepare: installing numpy into the venv (unprivileged, project-local)"
        if "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check numpy >>"$LOG_FILE" 2>&1; then
            log "prepare: numpy installed"
            # the venv's on-disk size is not a download size; recorded separately
            # so the 2 GB download budget means what it says
            state_budget venv_bytes "$(du -sb "$VENV" 2>/dev/null | awk '{print $1}')" >/dev/null
        else
            warn "numpy install failed; see $LOG_FILE"
        fi
    fi
    "$VENV/bin/python" -c 'import sys;print("venv python", sys.version.split()[0])' 2>/dev/null | while read -r l; do log "prepare: $l"; done
fi

# ---- 3. record what prepare produced ---------------------------------------
HIPIFY_SHA=$(sha256_of "$VENDOR_DIR/hipify-perl")
VENV_PY=$([ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import sys;print(sys.version.split()[0])' || echo "absent")
python3 - "$RESULTS_DIR/prepare.json" "$TS" "$HIPIFY_SHA" "$VENV_PY" <<'PY'
import json, sys
out, ts, sha, py = sys.argv[1:5]
json.dump({"name": "prepare", "at": ts,
           "hipify_perl_sha256": sha,
           "venv_python": py,
           "system_packages_touched": False}, open(out, "w"), indent=2)
PY

state_set next_action '"./strata-rocm doctor (outside the sandbox) then ./strata-rocm build"'
printf '\nprepare: done. hipify-perl sha256 %s, venv python %s\n' "${HIPIFY_SHA%"${HIPIFY_SHA#????????????}"}" "$VENV_PY"
