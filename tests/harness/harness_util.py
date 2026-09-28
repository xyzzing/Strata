"""Shared helpers for the harness self-tests (tests/harness).

These tests exercise the rollout's own tooling (scripts/rocm) against throwaway
ART_DIRs, never against artifacts/rocm. They test the wrapper code, not the
port; the GPU parity suite lives in tests/rocm and runs via ./strata-rocm test.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "rocm"
COMMON_SH = SCRIPTS / "lib" / "common.sh"
VENV_PY = REPO / ".venv-rocm" / "bin" / "python"
GGUF_PY = REPO / "build-hip" / "llama.cpp" / "gguf-py"
IQ_FIXTURES = REPO / "artifacts" / "rocm" / "iq_fixture"


def _find_python():
    """A python3 that actually behaves like one when subprocessed.

    Under some launch wrappers sys.executable is the wrapper binary, not the
    interpreter, so [sys.executable, script.py] launches a GUI instead of the
    test. sys.base_prefix is not rewritten; a behavioral check backs it up.
    """
    candidates = [Path(sys.base_prefix) / "bin" / "python3",
                  Path("/usr/bin/python3")]
    for cand in candidates:
        if cand.is_file():
            r = subprocess.run([str(cand), "-c", "print('interpreter-ok')"],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip() == "interpreter-ok":
                return str(cand)
    return sys.executable


PYTHON = _find_python()


class isolated_art:
    """An isolated ART_DIR/BUILD_DIR so a test never touches real state."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="strata-harness-")
        return Path(self.path)

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


def _base_env(art):
    env = dict(os.environ)
    env["ROCM_ROOT"] = str(REPO)
    env["ART_DIR"] = art
    env["BUILD_DIR"] = os.path.join(art, "build")
    # common.sh exports STATE_FILE/RESULTS_DIR/LOGS_DIR pointing at the REAL
    # artifacts dir; state.py prefers STATE_FILE over deriving it from ART_DIR,
    # so without these overrides a test would read and write the live record.
    env["STATE_FILE"] = os.path.join(art, "state.json")
    env["RESULTS_DIR"] = os.path.join(art, "results")
    env["LOGS_DIR"] = os.path.join(art, "logs")
    return env


def run_sh_snippet(snippet, timeout=60):
    """Run a POSIX sh snippet with common.sh sourced and an isolated ART_DIR.

    $0 is not meaningful under sh -c, so ROCM_ROOT is pinned via the environment
    (common.sh honors a preset value).
    """
    art = tempfile.mkdtemp(prefix="strata-harness-")
    env = _base_env(art)
    source = '. "$ROCM_ROOT/scripts/rocm/lib/common.sh"'
    try:
        return subprocess.run(
            ["sh", "-c", f"{source} && {snippet}"], capture_output=True,
            text=True, env=env, timeout=timeout,
        )
    finally:
        shutil.rmtree(art, ignore_errors=True)


def run_script(script, *args, env_extra=None, timeout=120):
    """Run one rollout script with an isolated ART_DIR; return the process."""
    art = tempfile.mkdtemp(prefix="strata-harness-")
    env = _base_env(art)
    if env_extra:
        env.update(env_extra)
    try:
        return subprocess.run(
            ["sh", str(script), *[str(a) for a in args]],
            capture_output=True, text=True, env=env, timeout=timeout,
        )
    finally:
        shutil.rmtree(art, ignore_errors=True)
