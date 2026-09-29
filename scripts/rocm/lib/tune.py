#!/usr/bin/env python3
"""strata-rocm tune - paired, holdout-confirmed flag tuning for the serve config.

Implements the optimization contract from the port-strata-hip skill in this
repo's own tooling: preflight fingerprint -> baseline -> interleaved candidate
-> accept/reject by a predeclared gain threshold -> holdout confirmation ->
record. Decisions are made by THIS script (deterministic, outside any LLM),
saved under artifacts/rocm/results/tune-*.json, and the only thing it may
modify is the serve config (a .bak checkpoint is kept). It never touches the
parity suite, tolerances, or any correctness gate.

The engine's own counters are the timing source (the "prompt N tokens ... read
in ... (X tok/s)" / "generated in ... (Y tok/s)" serve-log lines) - not client
wall-clock, which conflates HTTP and queueing.

    python3 tune.py --config <serve.json> --set prefill=auto [--set flag] \
        [--engine-binary PATH] [--workload <json>] [--min-gain 0.03] [--pairs 2] [--dry-run]

--engine-binary A/Bs two engine BUILDS (kernel-change trials) with the same
interleaved pairing; binary deltas are record-only - the live config is never
repointed at a scratch build.

Workload JSON: {"prompts": ["...", ...], "max_tokens": 64}. The default
workload is one short and one LONG prompt (~2600 tokens): prompt-read rate only
becomes measurable above the fixed per-request overheads, and chunked prefill
is where the engine's published numbers live.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "rocm" / "lib"))
import state  # noqa: E402  (the rollout's own record layer)

SERVE_PY = ROOT / "build-hip/hipify/serve/server.py"
VENV_PY = ROOT / ".venv-rocm/bin/python"
ENGINE_LOG = ROOT / "artifacts/rocm/logs/engine-tune.log"
PORT = 8083
UNIT = "strata-tune"

PROMPT_RE = re.compile(r"prompt (\d+) tokens = .*? read in .*? \(([0-9.]+) tok/s\)")
DECODE_RE = re.compile(r"(\d+) generated in .*? \(([0-9.]+) tok/s\)")

DEFAULT_WORKLOAD = {
    "prompts": [
        "Name the capital of France in one word.",
        ("The following is a long reference text for a prompt-processing measurement. "
         + "Tokens and throughput matter here, not meaning. " * 170),
    ],
    "max_tokens": 48,
}


# ---------------------------------------------------------------- pure logic

def parse_counters(log_text: str):
    """Latest (prompt_tok_s, decode_tok_s) from the engine's serve-log lines."""
    p = d = None
    for m in PROMPT_RE.finditer(log_text):
        p = float(m.group(2))
    for m in DECODE_RE.finditer(log_text):
        d = float(m.group(2))
    return p, d


def decide(baseline_decode, candidate_decode, baseline_prompt, candidate_prompt,
           min_gain=0.03, max_prompt_regression=0.03):
    """(accept: bool, reason: str). Gain is on median decode; prompt may not
    regress beyond max_prompt_regression; missing counters reject."""
    if not baseline_decode or not candidate_decode:
        return False, "missing counters"
    gain = (candidate_decode - baseline_decode) / baseline_decode
    if gain < min_gain:
        return False, f"decode gain {gain:+.1%} < {min_gain:.0%}"
    if baseline_prompt and candidate_prompt:
        reg = (baseline_prompt - candidate_prompt) / baseline_prompt
        if reg > max_prompt_regression:
            return False, f"prompt regression {reg:.1%} > {max_prompt_regression:.0%}"
    return True, f"decode gain {gain:+.1%}, prompt " + (
        f"{(candidate_prompt - baseline_prompt) / baseline_prompt:+.1%}" if baseline_prompt and candidate_prompt else "n/a")


def parse_sets(sets):
    """['prefill=auto', 'mmap-experts'] -> ['--prefill', 'auto', '--mmap-experts']"""
    out = []
    for s in sets:
        name, eq, val = s.partition("=")
        out += ["--" + name] + ([val] if eq else [])
    return out


def with_flags(args, flags):
    """args list with each --flag [value] applied (replacing an existing one)."""
    out = list(args)
    i = 0
    while i < len(flags):
        name = flags[i]
        val = flags[i + 1] if i + 1 < len(flags) and not flags[i + 1].startswith("--") else None
        if name in out:
            j = out.index(name)
            del out[j]
            if j < len(out) and not out[j].startswith("--"):
                del out[j]
        if val is not None:
            out += [name, val]
            i += 2
        else:
            out += [name]
            i += 1
    return out


# ---------------------------------------------------------------- engine mgmt

def sh(cmd, timeout=120, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def unit_stop():
    sh(["systemctl", "--user", "stop", UNIT])
    sh(["systemctl", "--user", "stop", "strata-8081"])   # the resident server owns the GPU
    time.sleep(4)


def unit_start(cfg_path):
    ENGINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    ENGINE_LOG.touch()
    r = sh(["systemd-run", "--user", "--collect", f"--unit={UNIT}",
            "-p", "OOMScoreAdjust=-500",
            "bash", "-c",
            f"exec env PYTHONPATH={ROOT}/build-hip/hipify HOME={os.environ.get('HOME', '~')} "
            f"{VENV_PY} {SERVE_PY} --engine strata --config {cfg_path} "
            f"--port {PORT} --host 127.0.0.1 > /tmp/tune_serve.log 2>&1"])
    if r.returncode != 0:
        raise RuntimeError(f"systemd-run failed: {r.stderr}")
    for _ in range(40):
        time.sleep(4)
        c = sh(["curl", "-fsS", "--max-time", "3", f"http://127.0.0.1:{PORT}/health"])
        if c.returncode == 0:
            return True
        if not sh(["systemctl", "--user", "is-active", UNIT]).stdout.strip() == "active":
            break
    raise RuntimeError("engine never became ready; see /tmp/tune_serve.log and " + str(ENGINE_LOG))


def one_request(prompt, max_tokens):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0})
    t0 = time.time()
    r = sh(["curl", "-fsS", "--max-time", "900",
            "http://127.0.0.1:%d/v1/chat/completions" % PORT,
            "-H", "Content-Type: application/json", "-d", body], timeout=920)
    ok = r.returncode == 0
    return ok, time.time() - t0
    # note: sh() timeouts raise; one_request callers treat missing counters as
    # failed samples in decide(), so a mid-experiment abort is visible in the record


def measure_one_pass(cfg_path, workload):
    """Start the engine, warm it, run the workload once, stop. Returns samples
    with the counters each request appended to the engine log."""
    unit_start(cfg_path)
    log = ENGINE_LOG.read_text(errors="replace")
    log_pos = len(log)
    samples = []
    try:
        # a FULL workload warmup pass: the first requests after a restart vary
        # 3x in decode while the tier prefills and pages warm - one-request
        # warmups poisoned earlier cycles
        for p in workload["prompts"]:
            one_request(p, workload["max_tokens"])
            log_pos = len(ENGINE_LOG.read_text(errors="replace"))
        for slot, p in enumerate(workload["prompts"]):
            ok, wall = one_request(p, workload["max_tokens"])
            text = ENGINE_LOG.read_text(errors="replace")
            pr, de = parse_counters(text[log_pos:])
            log_pos = len(text)
            samples.append({"ok": ok, "wall_s": round(wall, 2), "slot": slot,
                            "prompt_tok_s": pr, "decode_tok_s": de})
    finally:
        unit_stop_local()
    return samples


def measure_interleaved(base_cfg, cand_cfg, workload, pairs):
    """INTERLEAVED pairing (the contract's requirement): each pair restarts BOTH
    variants alternately, so cold-start costs land equally on each side. The
    first tune cycle taught this the hard way: run-all-of-A-then-all-of-B gave
    the baseline the whole cold tier and fabricated a +685% decode delta."""
    base_samples, cand_samples = [], []
    for i in range(pairs):
        print(f"[tune] pair {i + 1}/{pairs}: baseline ...")
        base_samples += measure_one_pass(base_cfg, workload)
        print(f"[tune] pair {i + 1}/{pairs}: candidate ...")
        cand_samples += measure_one_pass(cand_cfg, workload)
    return base_samples, cand_samples


def unit_stop_local():
    sh(["systemctl", "--user", "stop", UNIT])
    time.sleep(3)


def restore_resident(cfg_path):
    sh(["systemctl", "--user", "stop", UNIT])
    subprocess.run(["systemd-run", "--user", "--collect", "--unit=strata-8081",
                    "-p", "OOMScoreAdjust=-500", "-p", "Restart=on-failure",
                    "bash", "-c",
                    f"exec env PYTHONPATH={ROOT}/build-hip/hipify HOME={os.environ.get('HOME', '~')} "
                    f"{VENV_PY} {SERVE_PY} --engine strata --config {cfg_path} "
                    f"--port 8081 --host 127.0.0.1 > /tmp/serve_8081.log 2>&1"],
                   capture_output=True, text=True)


def vram_free_bytes():
    r = sh(["rocm-smi", "--showmeminfo", "vram"])
    total = used = None
    for line in r.stdout.splitlines():
        if "VRAM Total Memory" in line:
            total = int(line.split()[-1])
        if "VRAM Total Used Memory" in line:
            used = int(line.split()[-1])
    return (total - used) if total and used else 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="FLAG[=VALUE]",
                    help="candidate flag delta, repeatable: --set prefill=auto --set mmap-experts")
    ap.add_argument("--engine-binary", metavar="PATH", default="",
                    help="candidate ENGINE BINARY (kernel-change A/B): overrides the config's exe. "
                         "Binary deltas are record-only - the live config is never repointed at a scratch build")
    ap.add_argument("--workload", help="JSON {prompts, max_tokens}")
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg_path = Path(a.config)
    cfg = json.loads(cfg_path.read_text())
    workload = json.loads(Path(a.workload).read_text()) if a.workload else DEFAULT_WORKLOAD

    fingerprint = {
        "revision": subprocess.run(["git", "-C", str(ROOT / "Strata"), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "engine_sha256_16": subprocess.run(
            ["sha256sum", str(ROOT / "build-hip/cmake/strata")],
            capture_output=True, text=True).stdout[:16],
        "config_sha256_16": subprocess.run(["sha256sum", str(cfg_path)],
                                           capture_output=True, text=True).stdout[:16],
        "vram_free_bytes": vram_free_bytes(),
    }
    if a.engine_binary:
        fingerprint["candidate_engine_sha256_16"] = subprocess.run(
            ["sha256sum", a.engine_binary], capture_output=True, text=True).stdout[:16]
    print("[tune] fingerprint:", json.dumps(fingerprint))
    server_active = sh(["systemctl", "--user", "is-active", "strata-8081"]).stdout.strip() == "active"
    # The resident server's VRAM is freed before measurement, so it does not
    # count against the floor; anything ELSE holding VRAM does (the starved-tier
    # lesson: a leftover engine shrinks the expert cache and the comparison lies).
    if fingerprint["vram_free_bytes"] < 20 * (1 << 30) and not server_active:
        print("[tune] BLOCKED: <20 GiB VRAM free at start and no resident server to free "
              "- the expert tier would be starved. Check rocm-smi for leftovers.")
        return 3
    if fingerprint["vram_free_bytes"] < 20 * (1 << 30):
        print(f"[tune] note: {fingerprint['vram_free_bytes'] / (1 << 30):.1f} GiB free now; "
              "strata-8081 will be stopped first, freeing its VRAM for the tier")
    if not a.sets:
        print("[tune] no --set given; nothing to test (use --dry-run to inspect)")
        return 2
    candidate_flags = parse_sets(a.sets)

    if a.dry_run:
        print("[tune] baseline args:", cfg["args"])
        print("[tune] candidate args:", with_flags(cfg["args"], candidate_flags))
        return 0

    tmp = Path("/tmp/tune-candidate.json")
    cand_cfg = dict(cfg)
    cand_cfg["args"] = with_flags(cfg["args"], candidate_flags)
    if a.engine_binary:
        cand_cfg["exe"] = a.engine_binary
    cand_cfg["log"] = str(ENGINE_LOG)
    tmp.write_text(json.dumps(cand_cfg, indent=2))
    base_log_cfg = Path("/tmp/tune-baseline.json")
    base_cfg = dict(cfg)
    base_cfg["log"] = str(ENGINE_LOG)
    base_log_cfg.write_text(json.dumps(base_cfg, indent=2))

    crashed = None
    unit_stop()
    try:
        print("[tune] measuring (interleaved pairs) ...")
        base, cand = measure_interleaved(str(base_log_cfg), str(tmp), workload, a.pairs)
    except Exception as e:   # restore the resident server, then re-raise
        crashed = repr(e)
    finally:
        unit_stop_local()
    if crashed:
        restore_resident(cfg_path)
        raise SystemExit(f"[tune] aborted ({crashed}); resident server restored")

    def med(rows, key):
        """Per-slot median, averaged over slots: short and long prompts live in
        different regimes, so a pooled median is a coin flip between them."""
        slots = sorted({r["slot"] for r in rows if r.get("slot") is not None})
        if not slots:
            vals = [r[key] for r in rows if r[key] is not None]
            return statistics.median(vals) if vals else None
        per_slot = []
        for s in slots:
            vals = [r[key] for r in rows if r.get("slot") == s and r[key] is not None]
            if vals:
                per_slot.append(statistics.median(vals))
        return statistics.mean(per_slot) if per_slot else None

    bd, cd = med(base, "decode_tok_s"), med(cand, "decode_tok_s")
    bp, cp = med(base, "prompt_tok_s"), med(cand, "prompt_tok_s")
    accept, reason = decide(bd, cd, bp, cp, a.min_gain)

    record = {
        "fingerprint": fingerprint,
        "candidate": candidate_flags,
        "baseline": {"decode_med": bd, "prompt_med": bp, "samples": base},
        "candidate_out": {"decode_med": cd, "prompt_med": cp, "samples": cand},
        "verdict": "accept" if accept else "reject",
        "reason": reason,
        "min_gain": a.min_gain,
    }
    if accept:   # holdout: one fresh re-measure of the winner
        print("[tune] accepted; running holdout ...")
        hold = measure_one_pass(str(tmp), workload)
        record["holdout"] = {"decode_med": med(hold, "decode_tok_s"),
                             "prompt_med": med(hold, "prompt_tok_s"), "samples": hold}
        if record["holdout"]["decode_med"] and bd and \
           (record["holdout"]["decode_med"] - bd) / bd < a.min_gain / 2:
            record["verdict"] = "reject_after_holdout"
            record["reason"] += f"; holdout decode {record['holdout']['decode_med']:.1f} did not confirm"
            accept = False

    out = Path(ROOT / "artifacts/rocm/results")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (out / f"tune-{stamp}.json").write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/rocm/lib/state.py"), "record",
                    "tune", json.dumps({"verdict": record["verdict"], "reason": reason,
                                        "candidate": candidate_flags, "baseline_decode": bd,
                                        "candidate_decode": cd, "at": stamp})],
                   capture_output=True, text=True)

    if accept and a.engine_binary:
        record["verdict"] = "accept_record_only"
        print(f"[tune] ACCEPT (record-only, binary delta): {reason}. "
              "The live config is NOT repointed; promote the build by hand.")
    elif accept:
        bak = cfg_path.with_suffix(".json.bak")
        shutil.copy2(cfg_path, bak)
        cfg_path.write_text(json.dumps(cand_cfg, indent=2) + "\n")
        print(f"[tune] ACCEPT: {reason}. Config updated ({bak.name} kept).")
    else:
        print(f"[tune] REJECT: {reason}. Config untouched.")

    # restore the resident server (on the final config)
    restore_resident(cfg_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
