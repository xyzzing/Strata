#!/bin/sh
# strata-rocm report - one redacted Markdown report plus a machine-readable file.
#
# Redaction: the absolute home path is replaced with ~, and anything that looks
# like a token or key is dropped. No secrets, no credentials, no weights.
#
# Usage: strata-rocm report [--out FILE]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

OUT="$ROCM_ROOT/REPORT.md"
if [ "${1:-}" = "--out" ]; then
    [ $# -ge 2 ] || die "--out needs a file argument" $EX_USAGE
    OUT=$2
fi

TS=$(date -u +%Y%m%dT%H%M%SZ)

python3 - "$ROCM_ROOT" "$ART_DIR" "$STATE_FILE" "$OUT" "$TS" <<'PY'
import json, os, re, subprocess, sys
from datetime import datetime, timezone

root, art, state_path, out_path, ts = sys.argv[1:6]
results = os.path.join(art, "results")

def load(p, d=None):
    # A missing file and an unreadable one must not look the same: "never ran"
    # and "the evidence is there but corrupt" are different problems.
    try:
        with open(p) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return d if d is not None else {}
    except Exception as exc:
        return {"__load_error__": f"results file unreadable: "
                                  f"{os.path.basename(p)} ({exc})"}

def redact(text):
    text = text.replace(root, "~")
    text = re.sub(r"(?i)(token|api[_-]?key|password|secret)\s*[:=]\s*\S+", r"\1: <redacted>", text)
    return text

state = load(state_path)
build = load(os.path.join(results, "build.json"))
hipify = load(os.path.join(results, "hipify.json"))
kernels = load(os.path.join(results, "test-kernels.json"))
smoke = load(os.path.join(results, "smoke-run.json"))
doctor = load(os.path.join(art, "env", "manifest.json"))
patches = load(os.path.join(results, "hipify-patches.json"))

rev = state.get("current_revision", "unknown")
gates = state.get("gates", {})
if state.get("__load_error__"):
    state_err = state["__load_error__"]
    rev = "unknown"
    gates = {}
else:
    state_err = None
passed = [g for g, v in gates.items() if v.get("status") in ("passed", "passed_with_skips")]
pending = [g for g, v in gates.items() if v.get("status") not in ("passed", "passed_with_skips")]

lines = []
w = lines.append

w("# Strata ROCm status report")
w("")
w(f"Generated {ts} · revision `{rev}` · Fedora / ROCm HIP backend for `gfx1100`")
w("")
if state_err:
    w(f"**{state_err}** - gates and budgets below are missing, not clean.")
    w("")
w("**Scope of this report.** It states what has been executed on this machine and what has "
  "not. A build succeeding is not a port; a kernel agreeing with its reference is not a "
  "forward pass. Unsupported and unverified items are listed rather than omitted.")
w("")

w("## 1. Environment actually used")
w("")
env = doctor.get("environment", {})
for k in ("os", "cpu_model", "cpu_threads", "mem_total_mib", "gpu_name", "gpu_gfx",
          "hip_version", "cmake_version", "disk_free_mib"):
    if k in env:
        w(f"- `{k}`: {redact(str(env[k]))}")
w(f"- checkout: `{rev}`, tracked changes: "
  f"{'yes' if state.get('dirty_tree') else 'none'}")
w("")

w("## 2. Gates")
w("")
w("| gate | status | evidence |")
w("| --- | --- | --- |")
for g in gates:
    v = gates[g]
    ev = redact(str(v.get("evidence") or ""))
    w(f"| `{g}` | {v.get('status')} | {ev} |")
w("")

w("## 3. What was run")
w("")
if build:
    w(f"- **build**: {build.get('status')}, {build.get('executables')} executables, "
      f"{build.get('compiler_errors')} compiler errors, {build.get('jobs')} jobs, "
      f"timeout {build.get('timeout_s')}s")
if hipify:
    w(f"- **HIPIFY**: {hipify.get('files_changed')} of {hipify.get('files_total')} files "
      f"rewritten from `{hipify.get('revision')}`, translator sha256 "
      f"`{str(hipify.get('hipify_perl_sha256'))[:12]}`")
if patches.get("rules"):
    hits = ", ".join(f"{r['name']} x{r['hits']}" for r in patches["rules"] if r["hits"])
    w(f"- **declared patches after HIPIFY**: {hits or 'none'}")
if smoke:
    w(f"- **GPU smoke test**: {smoke.get('result', {}).get('status', '?')} "
      f"(device, arch, integer exactness and float agreement recorded in `results/`)")
w("")

w("## 4. Numerical evidence")
w("")
if "__load_error__" in kernels:
    w(f"**{kernels['__load_error__']}** - the suite result exists but cannot be "
      f"rendered; do not read this as a pass or as a fresh run.")
    w("")
elif kernels:
    w(f"Independent parity tests executed on the card: **{kernels.get('passed')} passed, "
      f"{kernels.get('failed')} failed, {kernels.get('skipped')} skipped**.")
    w("")
    w("| test | result | seconds | note |")
    w("| --- | --- | ---: | --- |")
    for t in kernels.get("tests", []):
        note = redact(t.get("detail", ""))[:160]
        w(f"| `{t['name']}` | {t['status']} | {t['seconds']} | {note} |")
    w("")
    w(f"Suite: {kernels.get('coverage', {}).get('suite')}")
    w("")
    w(f"> {kernels.get('coverage', {}).get('caveat')}")
    w("")
    if kernels.get("coverage", {}).get("not_run"):
        w("### Not verified")
        w("")
        for n in kernels["coverage"]["not_run"]:
            w(f"- {redact(n)}")
        w("")
else:
    w("No kernel test results recorded.")
    w("")

w("## 5. Unsupported and excluded")
w("")
if hipify.get("unsupported"):
    w("| construct | stage | hits | files |")
    w("| --- | --- | ---: | --- |")
    for e in hipify["unsupported"]:
        fl = ", ".join(f"`{p}`" for p in e["files"][:4])
        w(f"| {e['label']} | {e['stage']} | {e['hits']} | {fl} |")
else:
    w("None recorded.")
w("")
if kernels.get("coverage", {}).get("unsupported_sources"):
    for s in kernels["coverage"]["unsupported_sources"]:
        w(f"- excluded from the build: `{redact(s)}`")
w("")

w("## 6. Blockers and next action")
w("")
if state.get("blocker"):
    w(f"- blocker: {redact(json.dumps(state['blocker']))}")
w(f"- next action: {redact(str(state.get('next_action', '-')))}")
w("- pending gates: " + ", ".join(f"`{g}`" for g in pending))
w("")

w("## 7. Resource use")
w("")
b = state.get("budgets", {})
repairs = b.get("repairs", {})
repairs_used = (sum(int(e.get("count", 0)) for e in repairs.values())
                if isinstance(repairs, dict) else 0)
w(f"- wall clock recorded: {b.get('wall_clock_s', 0)} s")
w(f"- downloads: {b.get('download_bytes', 0)} bytes "
  f"(limit for unapproved downloads: {state.get('limits', {}).get('unapproved_download_bytes')} bytes)")
w(f"- repair attempts used: {repairs_used} of "
  f"{state.get('limits', {}).get('repair_attempts_per_failure_signature')} per failure signature")
w("- model weights downloaded: **none**")
w("")

text = redact("\n".join(lines))
with open(out_path, "w") as fh:
    fh.write(text)

json.dump({
    "at": ts, "revision": rev, "gates": gates,
    "tests": {k: kernels.get(k) for k in ("passed", "failed", "skipped")},
    "unsupported": hipify.get("unsupported", []),
    "budgets": state.get("budgets", {}),
    "blocker": state.get("blocker"),
    "next_action": state.get("next_action"),
    "report_markdown": os.path.relpath(out_path, root),
}, open(os.path.join(results, "report.json"), "w"), indent=2)
print(out_path)
PY

printf 'report written: %s\n' "$OUT"
printf 'machine-readable: %s\n' "$RESULTS_DIR/report.json"
