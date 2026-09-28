#!/bin/sh
# strata-rocm status - where the port actually is, from recorded state.
#
# Reads artifacts/rocm/state.json and the latest result files. Prints nothing it
# cannot point at evidence for.
#
# Usage: strata-rocm status [--json]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

JSON=0
[ "${1:-}" = "--json" ] && JSON=1

python3 - "$STATE_FILE" "$RESULTS_DIR" "$STRATA_DIR" "$JSON" <<'PY'
import json, os, sys

state_path, results, strata, as_json = sys.argv[1:5]

def load(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default

st = load(state_path, {})
if not st:
    print("no state recorded yet: run ./strata-rocm doctor")
    raise SystemExit(1)

def git(*a):
    import subprocess
    try:
        return subprocess.run(["git", "-C", strata, *a], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""

rev = git("rev-parse", "HEAD") or st.get("current_revision", "unknown")
tracked_dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
untracked = git("status", "--porcelain", "--untracked-files=all").splitlines()

tests = st.get("tests", {})
kernels = load(os.path.join(results, "test-kernels.json"), {})
smoke = load(os.path.join(results, "smoke-run.json"), {})
build = load(os.path.join(results, "build.json"), {})
hipify = load(os.path.join(results, "hipify.json"), {})
doctor = load(os.path.join(results, "doctor-latest.json")) or load(
    os.path.join(os.path.dirname(results), "env", "manifest.json"), {})

gates = st.get("gates", {})

if as_json == "1":
    print(json.dumps({
        "revision": rev, "tracked_dirty": tracked_dirty, "untracked": untracked,
        "stage": st.get("stage"), "gates": gates, "tests": tests,
        "blocker": st.get("blocker"), "next_action": st.get("next_action"),
        "budgets": st.get("budgets"),
    }, indent=2))
    raise SystemExit(0)

def row(k, v):
    print(f"  {k:<22} {v}")

print(f"Strata ROCm port - status")
print(f"{'-'*66}")
row("revision", rev)
# Parens matter: without them the `+` binds into the ternary's true branch and
# the untracked count silently disappears.
row("checkout", ("tracked changes: yes" if tracked_dirty else "tracked changes: none")
    + (f" | {len(untracked)} untracked file(s)" if untracked else ""))
row("stage", st.get("stage", "?"))
row("last state update", st.get("updated", "?"))

if build:
    row("last build", f"{build.get('status')} ({build.get('executables')} executables, "
                      f"{build.get('compiler_errors')} compiler errors)")
if hipify:
    row("hipify", f"{hipify.get('files_changed')}/{hipify.get('files_total')} files rewritten, "
                  f"{len(hipify.get('unsupported', []))} unsupported construct classes")
if smoke:
    row("gpu smoke", smoke.get("result", {}).get("status", smoke.get("status", "?")))
if kernels:
    row("kernel tests", f"{kernels.get('passed')} passed, {kernels.get('failed')} failed, "
                        f"{kernels.get('skipped')} skipped")
row("budgets", f"wall {st.get('budgets',{}).get('wall_clock_s',0)}s, "
               f"downloads {st.get('budgets',{}).get('download_bytes',0)} bytes")

print(f"\n  gates")
for g in gates:
    v = gates[g]
    mark = {"passed": "PASS", "failed": "FAIL", "blocked": "BLOCK",
            "passed_with_skips": "PASS*", "partial": "PART", "pending": "...."}.get(
                v.get("status"), "?")
    ev = v.get("evidence") or ""
    print(f"    {mark:<6} {g:<28} {ev}")
print("    (* passed with declared, explicitly-reported gaps)")

if kernels.get("coverage", {}).get("not_run"):
    print("\n  not verified")
    for n in kernels["coverage"]["not_run"]:
        print(f"    - {n[:150]}")
if kernels.get("coverage", {}).get("unsupported_sources"):
    for n in kernels["coverage"]["unsupported_sources"]:
        print(f"    - excluded from the build: {n}")

if st.get("blocker"):
    b = st["blocker"]
    print(f"\n  BLOCKER: {b.get('detail') or b.get('kind')}")
print(f"\n  next: {st.get('next_action','-')}")
PY
