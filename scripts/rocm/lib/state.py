#!/usr/bin/env python3
"""Resumable rollout state and machine-readable results.

One file, one job: keep artifacts/rocm/state.json honest across sessions so a
later agent can tell what was already proven, against which revision, and what
comes next. No third-party imports.

    state.py init
    state.py get <dotted.key>
    state.py set <dotted.key> <json-value>
    state.py gate <gate-id> <status> [evidence-path]
    state.py record <name> <json-object>     # results/<name>.json + history.jsonl
    state.py budget <name> <delta>
    state.py attempt <failure-signature> <outcome>
    state.py show
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.environ.get("ROCM_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
ART = os.environ.get("ART_DIR") or os.path.join(ROOT, "artifacts", "rocm")
STATE = os.environ.get("STATE_FILE") or os.path.join(ART, "state.json")
RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(ART, "results")
HISTORY = os.path.join(RESULTS, "history.jsonl")
STRATA = os.environ.get("STRATA_DIR") or os.path.join(ROOT, "Strata")

# Initial automation limits, section 7 of the rollout plan. Recorded in state so
# a later session can see what was actually enforced, not what was intended.
LIMITS = {
    "repair_attempts_per_failure_signature": 3,
    "execution_slice_minutes": 60,
    "compile_parallelism": 4,
    "small_gpu_test_timeout_s": 120,
    "build_timeout_s": 1800,
    "unapproved_download_bytes": 2 * 1024**3,
}

# Acceptance gates, in order. `pending` is the only honest starting value.
GATES = [
    "S0-hardware-hip-smoke",
    "S0-source-audit",
    "S1-harness-selfcheck",
    "S1-cpu-reference",
    "S2-kernel-suite-gfx1100",
    "S2.6-iq-fixtures",
    "S2.7-core-runtime",
    "S3.2b-mmq-types",
    "S3-attention-prefill",
    "S3.3-attention",
    "S3.3-gdn-recurrence",
    "S3.3-gdn-preprocess",
    "S3.3-moe-combine",
    "S3.3-gr-postops",
    "S3.3-qsa-ops",
    "S3.3-qsa-indexer",
    "S3.3-rope",
    "S3.3-mmvq-q5k",
    "S3.3-forward-ops",
    "S3-feasibility",
    "S4-first-inference",
    "S5-concurrency",
    "S6-performance",
    "S7-release",
]


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sh(cmd, cwd=None):
    try:
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except Exception:
        return ""


def environment():
    """Cheap, read-only snapshot. doctor.sh writes the full manifest."""
    return {
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "python": sys.version.split()[0],
        "rocm_root": ROOT,
    }


def default_state():
    rev = sh(["git", "-C", STRATA, "rev-parse", "HEAD"]) or "unknown"
    dirty = bool(sh(["git", "-C", STRATA, "status", "--porcelain", "--untracked-files=no"]))
    return {
        "schema": 1,
        "created": now(),
        "updated": now(),
        "stage": "0",
        "baseline_revision": rev,
        "current_revision": rev,
        "dirty_tree": dirty,
        "limits": LIMITS,
        "budgets": {
            "wall_clock_s": 0,
            "download_bytes": 0,
            "repairs": {},
        },
        "gates": {g: {"status": "pending", "evidence": None, "at": None} for g in GATES},
        "tests": {"passed": 0, "failed": 0, "skipped": 0, "last": None},
        "unverified": [],
        "blocker": None,
        "next_action": "run ./strata-rocm doctor",
        "environment": environment(),
        "history": [],
    }


def load():
    try:
        with open(STATE) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default_state()
    except json.JSONDecodeError:
        sys.stderr.write("state.json is corrupt; leaving it in place, not overwriting\n")
        raise SystemExit(1)


def save(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    st["updated"] = now()
    tmp = STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, STATE)


def get_path(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def set_path(obj, dotted, value):
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def main(argv):
    if len(argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    cmd = argv[1]

    if cmd == "init":
        st = load()
        save(st)
        json.dump(st, sys.stdout, indent=2)
        print()
        return 0

    if cmd == "show":
        json.dump(load(), sys.stdout, indent=2)
        print()
        return 0

    if cmd == "get":
        val = get_path(load(), argv[2])
        if val is None:
            return 1
        print(val if isinstance(val, str) else json.dumps(val))
        return 0

    if cmd == "set":
        st = load()
        try:
            value = json.loads(argv[3])
        except json.JSONDecodeError:
            value = argv[3]
        set_path(st, argv[2], value)
        save(st)
        return 0

    if cmd == "gate":
        gate, status, evidence = argv[2], argv[3], (argv[4] if len(argv) > 4 else None)
        if gate not in GATES:
            sys.stderr.write(
                f"warning: '{gate}' is not a declared gate in state.py GATES; "
                f"recording it anyway, but the GATES table has drifted from this "
                f"caller - reconcile them\n")
        st = load()
        entry = st["gates"].setdefault(gate, {})
        entry.update({"status": status, "evidence": evidence, "at": now()})
        st["history"].append({"at": now(), "event": "gate", "gate": gate, "status": status})
        save(st)
        print(f"{gate}: {status}")
        return 0

    if cmd == "record":
        name = argv[2]
        payload = argv[3]
        data = json.loads(payload) if payload.lstrip().startswith(("{", "[")) else {"value": payload}
        record = {
            "name": name,
            "at": now(),
            "revision": sh(["git", "-C", STRATA, "rev-parse", "HEAD"]) or "unknown",
            "dirty_tree": bool(
                sh(["git", "-C", STRATA, "status", "--porcelain", "--untracked-files=no"])
            ),
            "result": data,
        }
        os.makedirs(RESULTS, exist_ok=True)
        # namespaced: scripts own results/<thing>.json for their own output, and
        # a summary record must never overwrite it (it did once - hipify.json and
        # build.json were lost, and status then reported None for both).
        with open(os.path.join(RESULTS, "record-" + name + ".json"), "w") as fh:
            json.dump(record, fh, indent=2)
            fh.write("\n")
        with open(HISTORY, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(os.path.join(RESULTS, "record-" + name + ".json"))
        return 0

    if cmd == "budget":
        st = load()
        name, delta = argv[2], float(argv[3])
        st["budgets"][name] = st["budgets"].get(name, 0) + delta
        save(st)
        print(f"{name}: {st['budgets'][name]}")
        return 0

    if cmd == "attempt":
        signature, outcome = argv[2], argv[3]
        st = load()
        reps = st["budgets"].setdefault("repairs", {})
        entry = reps.setdefault(signature, {"count": 0, "outcomes": []})
        entry["count"] += 1
        entry["outcomes"].append({"at": now(), "outcome": outcome})
        limit = st["limits"]["repair_attempts_per_failure_signature"]
        over = entry["count"] >= limit
        if over:
            st["blocker"] = {
                "signature": signature,
                "attempts": entry["count"],
                "limit": limit,
                "at": now(),
                "detail": "repair attempt limit reached for this failure signature",
            }
        save(st)
        print(f"{signature}: attempt {entry['count']}/{limit}" + (" LIMIT REACHED" if over else ""))
        return EXIT_LIMIT if over else 0

    sys.stderr.write(f"unknown command: {cmd}\n")
    return 2


EXIT_LIMIT = 9

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except BrokenPipeError:
        sys.exit(0)
