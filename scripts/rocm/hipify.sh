#!/bin/sh
# strata-rocm hipify - mechanical CUDA -> HIP translation, reproducibly.
#
# Method: take a pristine copy of the pinned revision (git archive, so
# untracked junk and our own files can never leak in), run the vendored,
# hash-pinned hipify-perl over every CUDA-dependent file *in place*, and keep
# hipify-perl's own .prehip backups as the before-image. The checkout under
# Strata/ is never written to.
#
# "Generated, not forked": the HIP tree is a build artifact, regenerated from
# upstream sources on demand and never hand-edited. Diffs between the CUDA and
# HIP trees are therefore always attributable to HIPIFY or to a documented
# patch in scripts/rocm/patches/, never to drift.
#
# Usage: strata-rocm hipify [--force] [--report]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

FORCE=0
REPORT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --report) REPORT=1; shift ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" $EX_USAGE ;;
    esac
done

HIPIFY="$VENDOR_DIR/hipify-perl"
OUT="$BUILD_DIR/hipify"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOGS_DIR/hipify-$TS.log"

[ -f "$HIPIFY" ] || die "vendored hipify-perl missing; run './strata-rocm prepare'" $EX_BLOCKED
have perl || die "perl missing; hipify-perl needs it" $EX_BLOCKED
[ -d "$STRATA_DIR/.git" ] || die "no checkout at $STRATA_DIR" $EX_BLOCKED

REV=$(strata_revision)
# The stamp covers the patch set too: editing a patch must regenerate the tree,
# otherwise a stale tree silently contradicts the recorded patches.
# Covers both the whole-file patches and the rule table: editing either must
# regenerate the tree, or a stale tree silently contradicts the recorded patches.
PATCH_HASH=$(cat "$SCRIPTS_DIR"/patches/*.patch "$SCRIPTS_DIR"/patches/*.py \
             "$SCRIPTS_DIR"/patches/*.h 2>/dev/null | sha256sum 2>/dev/null | cut -c1-16)
PATCH_HASH=${PATCH_HASH:-none}
STAMP="$OUT/.hipify-revision"
FRESH=0
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$REV $PATCH_HASH" ] && [ "$FORCE" = 0 ]; then
    log "hipify: tree already generated for $REV with this patch set (use --force to regenerate)"
else
    log "hipify: generating from $REV"
    rm -rf "$OUT"
    mkdir -p "$OUT"
    git -C "$STRATA_DIR" archive --format=tar HEAD | tar -x -C "$OUT" || \
        die "could not export $REV" $EX_FAIL
    printf '%s' "$REV $PATCH_HASH" >"$STAMP"
    FRESH=1

    # Which files does HIPIFY touch? Anything that names a CUDA API, a CUDA
    # runtime header, or a device-code qualifier. Files that merely live next to
    # them are left byte-identical, which is why a plain copy is the right
    # before-image for the report.
    FILES=$(grep -rlE 'cuda[A-Z_]|cuda_runtime|cuda\.h|__device__|__global__|__host__|__constant__|__shared__|cublas|cuBLAS|nvtx|nvml|__nv_|cuda_fp16|cuda_bf16|mma\.sync|ldmatrix|cooperative_groups|cub::' \
            --include='*.cu' --include='*.cuh' --include='*.cpp' --include='*.h' --include='*.hpp' \
            "$OUT" 2>/dev/null | sed "s|^$OUT/||" | sort)
    N=$(printf '%s\n' "$FILES" | grep -c . || true)
    log "hipify: $N CUDA-dependent files"
    : >"$LOG_FILE"
    printf '%s\n' "$FILES" | while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        printf '%s\n' "$rel" >>"$LOG_FILE"
    done
fi

# ---- translation ------------------------------------------------------------
# idempotent: a .prehip file means this file was already translated
todo=$(find "$OUT" \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) \
       -not -name '*.prehip' 2>/dev/null)
printf '%s\n' "$todo" | while IFS= read -r f; do
    [ -n "$f" ] || continue
    [ -f "$f.prehip" ] && continue
    before=$(sha256_of "$f")
    if perl "$HIPIFY" -quiet-warnings -inplace "$f" >>"$LOG_FILE" 2>&1; then
        :
    else
        # hipify-perl exits nonzero on unsupported constructs; it still leaves
        # a translated file behind. Recorded, not fatal - the report names them.
        printf 'HIPIFY_NONZERO %s\n' "$f" >>"$LOG_FILE"
    fi
    after=$(sha256_of "$f")
    if [ "$before" != "$after" ]; then
        printf 'CHANGED %s\n' "${f#$OUT/}" >>"$LOG_FILE"
    fi
done

# ---- declared textual patches ----------------------------------------------
# Applied after hipify-perl, to the generated tree only. See
# scripts/rocm/patches/hipify_patches.py for the rules and their justification.
if [ -f "$SCRIPTS_DIR/patches/hipify_patches.py" ]; then
    log "hipify: applying declared patches"
    python3 "$SCRIPTS_DIR/patches/hipify_patches.py" "$OUT" \
        --json "$RESULTS_DIR/hipify-patches.json" >>"$LOG_FILE" 2>&1 || \
        warn "patch step failed; see $LOG_FILE"
fi

# ---- declared file patches ---------------------------------------------------
# Whole-block rewrites that are too large to express as a regex rule (the PTX
# replacement in native_qsa_score.cu, for instance). A unified diff, applied to
# the generated tree, recorded with its hash - so "which patch produced this"
# is always answerable.
PATCHED="[]"
if [ "$FRESH" != 1 ]; then
    # The translate step is self-guarding (a .prehip means "already done"); a
    # file patch is not - applying it twice writes .rej files and leaves a tree
    # that still compiles, which is the worst possible outcome. Patches therefore
    # only ever run against a tree that was exported in this same run, which is
    # also what makes editing a patch take effect: the stamp covers the patch set.
    log "hipify: tree reused; file patches not re-applied"
elif ls "$SCRIPTS_DIR"/patches/*.patch >/dev/null 2>&1; then
    log "hipify: applying file patches"
    : >"$ART_DIR/results/patches-applied.jsonl"
    for pf in "$SCRIPTS_DIR"/patches/*.patch; do
        pname=$(basename "$pf")
        psha=$(sha256_of "$pf")
        if patch -p1 -d "$OUT" --forward --silent <"$pf" >>"$LOG_FILE" 2>&1; then
            log "hipify: applied $pname"
            printf '{"patch":"%s","sha256":"%s","status":"applied"}\n' "$pname" "$psha" \
                >>"$ART_DIR/results/patches-applied.jsonl"
        else
            warn "hipify: $pname did not apply cleanly (see $LOG_FILE)"
            printf '{"patch":"%s","sha256":"%s","status":"failed"}\n' "$pname" "$psha" \
                >>"$ART_DIR/results/patches-applied.jsonl"
        fi
    done
    PATCHED=$(python3 -c '
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(json.dumps(rows))' "$ART_DIR/results/patches-applied.jsonl")
fi

# ---- reject guard -----------------------------------------------------------
# A rejected hunk leaves the file at its pre-patch content, which usually still
# compiles. Compiling is not evidence of anything here, so this fails loudly
# instead of publishing a manifest that misdescribes the tree.
REJECTS=$(find "$OUT" -name '*.rej' 2>/dev/null)
if [ -n "$REJECTS" ]; then
    warn "hipify: rejected hunks present; the tree does NOT match the recorded patches:"
    printf '%s\n' "$REJECTS" | sed 's/^/    /' >&2
    state_record "hipify" "{\"status\":\"rejected_hunks\",\"rejects\":\"$(printf '%s' "$REJECTS" | tr '\n' ' ')\"}"
    state_gate S0-source-audit failed "$ART_DIR/results/hipify.json"
    exit $EX_FAIL
fi

# ---- manifest + report ------------------------------------------------------
python3 - "$OUT" "$RESULTS_DIR/hipify.json" "$TS" "$REV" "$STRATA_DIR" <<'PY'
import hashlib, json, os, re, sys


def sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

out, result_path, ts, rev, strata = sys.argv[1:6]
patches_path = os.path.join(os.path.dirname(result_path), "hipify-patches.json")
patches = json.load(open(patches_path)) if os.path.exists(patches_path) else {}
patches_path2 = os.path.join(os.path.dirname(result_path), "patches-applied.jsonl")
file_patches = [json.loads(l) for l in open(patches_path2)] if os.path.exists(patches_path2) else []

# Constructs HIPIFY cannot translate. Each needs a hand-written HIP path; they
# are listed rather than silently compiled, because "it built" must not be
# confused with "the path is correct".
UNSUPPORTED = [
    (r"\basm\s*\(|asm\s+volatile", "inline PTX / device assembly", "S3"),
    (r"mma\.sync|ldmatrix|wgmma|cp\.async", "NVIDIA tensor-core PTX", "S3"),
    (r"\bcub::", "CUB (no HIP equivalent)", "S2"),
    (r"cooperative_groups", "cooperative groups", "S2"),
    (r"cudaGraph", "CUDA graph capture", "S5"),
    (r"cudaStreamAddCallback|cudaLaunchHostFunc", "stream callbacks", "S5"),
    (r"cudaDeviceSetLimit|cudaFuncSetAttribute.*MaxDynamicSharedMemorySize", "shared-mem opt-in", "S2"),
    (r"cudaMemPool|cudaMallocAsync", "stream-ordered allocator", "S2"),
    (r"__nv_|nvml|nvtx", "NVIDIA-only runtime", "S2"),
    (r"cuda_fp16\.h|cuda_bf16\.h", "CUDA fp16/bf16 headers", "S2"),
]

files = []
for root, dirs, names in os.walk(out):
    dirs[:] = [d for d in dirs if d != ".git"]
    for n in names:
        if n.endswith(".prehip") or n == ".hipify-revision":
            continue   # generated marker, not source
        p = os.path.join(root, n)
        rel = os.path.relpath(p, out)
        pre = p + ".prehip"
        after = hashlib.sha256(open(p, "rb").read()).hexdigest()
        before = hashlib.sha256(open(pre, "rb").read()).hexdigest() if os.path.exists(pre) else None
        files.append({"path": rel, "sha256": after,
                      "cuda_sha256": before,
                      "hipified": before is not None and before != after,
                      "translated": before is not None})

findings = {}
for f in files:
    if not f["translated"]:
        continue
    try:
        text = open(os.path.join(out, f["path"]), encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    # Strip line comments first: after the portable-path patch the only mention of
    # mma.sync left in the tree is the comment explaining why it is gone, and a
    # scanner that counts its own explanation is a scanner nobody trusts.
    code = "\n".join(l.split("//", 1)[0] for l in text.splitlines())
    for pattern, label, stage in UNSUPPORTED:
        hits = len(re.findall(pattern, code))
        if hits:
            e = findings.setdefault(label, {"label": label, "stage": stage, "hits": 0,
                                            "files": []})
            e["hits"] += hits
            if f["path"] not in e["files"]:
                e["files"].append(f["path"])

summary = {
    "generated_at": ts,
    "revision": rev,
    "checkout": strata,
    "hipify_perl_sha256": sha256_file(os.path.join(os.environ.get("VENDOR_DIR", ""), "hipify-perl")),
    "files_total": len(files),
    "files_translated": sum(1 for f in files if f["translated"]),
    "files_changed": sum(1 for f in files if f["hipified"]),
    "unsupported": sorted(findings.values(), key=lambda e: -e["hits"]),
    "patches": patches,
    "file_patches": file_patches,
    "files": files,
}
os.makedirs(os.path.dirname(result_path), exist_ok=True)
with open(result_path, "w") as fh:
    json.dump(summary, fh, indent=2)
    fh.write("\n")

md = os.path.join(os.environ.get("ROCM_ROOT", "."), "HIPIFY_REPORT.md")
report_text = []
report_text.append("# HIPIFY report\n\n")
report_text.append(f"Generated {ts} from `{rev}` with hipify-perl "
                   f"`{(summary['hipify_perl_sha256'] or '?')[:12]}`.\n\n")
report_text.append(f"- files in the generated tree: {summary['files_total']}\n")
report_text.append(f"- files hipify-perl rewrote: {summary['files_changed']}\n")
report_text.append("- tree: `build-hip/hipify/` (generated, never hand-edited; "
                   "`.prehip` files are hipify-perl's before-image)\n\n")
report_text.append("## Declared textual patches applied after HIPIFY\n\n")
if not patches.get("rules"):
    report_text.append("None applied.\n")
else:
    report_text.append("| patch | hits | what it means |\n| --- | ---: | --- |\n")
    for r in patches["rules"]:
        report_text.append(f"| `{r['name']}` | {r['hits']} | {r['note']} |\n")
    report_text.append(f"\n{patches.get('files_changed', 0)} files touched. Rules live in "
                       "`scripts/rocm/patches/hipify_patches.py`; the forced-include shim for "
                       "NVIDIA-only intrinsics is `scripts/rocm/patches/hip_nv_intrinsics.h`.\n")
report_text.append("\n### File patches applied after HIPIFY\n\n")
if not file_patches:
    report_text.append("None.\n")
else:
    report_text.append("| patch | sha256 | status |\n| --- | --- | --- |\n")
    for e in file_patches:
        report_text.append(f"| `{e['patch']}` | `{e['sha256'][:12]}` | {e['status']} |\n")
report_text.append("\n## Constructs HIPIFY cannot translate\n\n")
if not summary["unsupported"]:
    report_text.append("None found.\n")
else:
    report_text.append("| construct | stage | hits | files |\n| --- | --- | ---: | --- |\n")
    for e in summary["unsupported"]:
        fl = "<br>".join(f"`{p}`" for p in e["files"][:6])
        if len(e["files"]) > 6:
            fl += f"<br>... and {len(e['files']) - 6} more"
        report_text.append(f"| {e['label']} | {e['stage']} | {e['hits']} | {fl} |\n")
report_text.append("\nA row here is an unsupported or unverified path, not a build error: "
                   "it needs a HIP implementation and an independent test before any claim.\n")
# Two copies, one content: the root copy is what a reader opens first, the
# artifacts copy is what an auditor reading only artifacts/ gets. They used to
# drift (artifacts held a 17-byte stub) because only one was written.
for md in (os.path.join(os.environ.get("ROCM_ROOT", "."), "HIPIFY_REPORT.md"),
           os.path.join(os.path.dirname(os.environ.get("ART_DIR", ".")),
                        "HIPIFY_REPORT.md")):
    try:
        os.makedirs(os.path.dirname(md), exist_ok=True)
        with open(md, "w") as fh:
            fh.write("".join(report_text))
    except OSError as exc:
        sys.stderr.write(f"warning: could not write {md}: {exc}\n")
print(json.dumps({"files_total": summary["files_total"],
                  "files_changed": summary["files_changed"],
                  "unsupported_categories": len(summary["unsupported"]),
                  "unsupported_hits": sum(e["hits"] for e in summary["unsupported"])}))
PY

state_set hipify_revision "\"$REV\""
state_record "hipify" "{\"revision\":\"$REV\",\"tree\":\"$OUT\"}"
printf '\nhipify: tree at %s\n' "$OUT"
[ "$REPORT" = 1 ] && sed -n '1,60p' "$ROCM_ROOT/HIPIFY_REPORT.md"
exit $EX_OK
