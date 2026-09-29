#!/usr/bin/env python3
"""Run the ported test binaries and classify the outcome honestly.

ctest is not used here on purpose. It reports pass/fail/timeout but not *why* a
test stopped, and this rollout has to distinguish three outcomes that must never
be conflated:

  passed   the test ran and agreed with its reference
  failed   the test ran and disagreed, or crashed
  skipped  the test could not run because an artifact it needs does not exist
           here (an unpublished fixture, the not-yet-downloaded model)

A "skipped" is only granted when the test's own output matches a declared
missing-artifact signature. A test that is declared unavailable and then fails
for a *different* reason is reported as failed, so declaring a test unavailable
cannot hide a regression.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="build directory holding the test binaries")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unavailable", default="", help="JSON file of declared unavailable tests")
    ap.add_argument("--args", default="", help="JSON file of per-test argv overrides")
    ap.add_argument("--logs", default="")
    ap.add_argument("--filter", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    unavailable = {}
    if args.unavailable and os.path.exists(args.unavailable):
        doc = json.load(open(args.unavailable))
        # the file documents itself with a "_comment" key, so the real table may
        # sit under "tests"
        unavailable = doc.get("tests", doc)

    # Named probe binaries are listed explicitly: they are real self-tests (the
    # device probe asserts arena alignment, poisoning and over-allocation
    # refusal), and a suffix guess would silently drop one that gets renamed.
    PROBES = {"strata-device"}
    # Default argv is --selftest, but a test that takes a positional argument must
    # say so: passing --selftest to native_expert_parity made it treat the flag as
    # a filename and abort.
    argv_override = {}
    if args.args and os.path.exists(args.args):
        argv_override = json.load(open(args.args)).get("tests", {})

    binaries = sorted(
        p for p in glob.glob(os.path.join(args.dir, "*"))
        if os.path.isfile(p) and os.access(p, os.X_OK)
        and (p.endswith("_parity") or p.endswith("_test")
             or os.path.basename(p) in PROBES)
    )
    if args.filter:
        binaries = [b for b in binaries if args.filter in os.path.basename(b)]

    if args.list:
        for b in binaries:
            print(os.path.basename(b))
        return 0

    if args.logs:
        os.makedirs(args.logs, exist_ok=True)

    results = []
    for path in binaries:
        name = os.path.basename(path)
        started = time.time()
        status, detail, log_path = "passed", "", ""
        try:
            # ${ROCM_ROOT} expands here, so a test's arguments can name an
            # absolute path without the config file hard-coding one user's home.
            argv = [a.replace("${ROCM_ROOT}", os.environ.get("ROCM_ROOT", ""))
                    for a in argv_override.get(name, {}).get("argv", ["--selftest"])]
            proc = subprocess.run(
                [path] + list(argv), capture_output=True, text=True,
                timeout=args.timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            code = proc.returncode
            if code != 0:
                status = "failed"
                decl = unavailable.get(name)
                matched = None
                if decl:
                    try:
                        matched = re.search(decl["signature"], out)
                    except re.error as exc:
                        # A broken declaration must not abort the suite and must
                        # never earn a skip: the test is reported failed with
                        # the reason named.
                        detail = (f"invalid skip signature in unavailable.json "
                                  f"for {name}: {exc}")
                if matched:
                    status = "skipped"
                    detail = decl["reason"]
                elif not detail:
                    detail = next(
                        (l.strip() for l in reversed(out.splitlines()) if l.strip()),
                        f"exit {code}",
                    )[:300]
        except subprocess.TimeoutExpired:
            out, code = "", None
            status = "failed"
            detail = f"TIMEOUT after {args.timeout}s (not a numerical failure)"
        except OSError as exc:
            out, code = "", None
            status = "failed"
            detail = f"could not execute: {exc}"

        if args.logs:
            log_path = os.path.join(args.logs, f"{name}.log")
            with open(log_path, "w") as fh:
                fh.write(out)
        results.append({
            "name": name, "status": status, "seconds": round(time.time() - started, 2),
            "exit": code, "detail": detail, "log": log_path,
        })

    passed = [r for r in results if r["status"] == "passed"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    summary = {
        "passed": len(passed), "failed": len(failed), "skipped": len(skipped),
        "status": "failed" if failed else ("incomplete" if skipped else "passed"),
        "tests": results,
    }
    json.dump(summary, open(args.out, "w"), indent=2)

    for r in results:
        mark = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}[r["status"]]
        extra = f"  {r['detail'][:100]}" if r["detail"] else ""
        print(f"  {mark}  {r['name']:<26} {r['seconds']:6.2f}s{extra}")

    # Nonzero whenever anything did not pass, with distinct codes: a skipped run
    # is not a pass and must not look like one.
    if failed:
        return 4
    if skipped:
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
