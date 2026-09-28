"""state.py: the resumable-record contract.

Regression locks: corrupt state is refused, not overwritten; missing keys exit
1. New behavior under test: an unknown gate id must warn loudly instead of
silently creating a phantom gate (S3.4 was once added this way without anyone
noticing the GATES table had drifted).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness_util import PYTHON, REPO, _base_env

STATE_PY = REPO / "scripts" / "rocm" / "lib" / "state.py"


def run_state(*args, art=None, env_extra=None):
    # _base_env, not dict(os.environ): under ./strata-rocm selftest the wrapper
    # exports STATE_FILE pointing at the live record, and state.py trusts it.
    env = _base_env(art)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [PYTHON, str(STATE_PY), *args],
        capture_output=True, text=True, env=env, timeout=60,
    )


class StateGateTest(unittest.TestCase):
    def setUp(self):
        self.art = Path(tempfile.mkdtemp(prefix="strata-state-"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.art, ignore_errors=True)

    @property
    def state_file(self):
        return self.art / "state.json"

    def test_known_gate_records_silently(self):
        r = run_state("gate", "S2-kernel-suite-gfx1100", "passed", art=self.art)
        self.assertEqual(r.returncode, 0)
        self.assertNotIn("not a declared gate", r.stderr)
        doc = json.loads(self.state_file.read_text())
        self.assertEqual(doc["gates"]["S2-kernel-suite-gfx1100"]["status"], "passed")

    def test_unknown_gate_warns_but_still_records(self):
        r = run_state("gate", "totally-made-up-gate", "passed", art=self.art)
        self.assertEqual(r.returncode, 0)
        self.assertIn("not a declared gate", r.stderr)
        self.assertIn("totally-made-up-gate", r.stderr)
        doc = json.loads(self.state_file.read_text())
        self.assertEqual(doc["gates"]["totally-made-up-gate"]["status"], "passed")


class StateRegressionLocks(unittest.TestCase):
    def setUp(self):
        self.art = Path(tempfile.mkdtemp(prefix="strata-state-"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.art, ignore_errors=True)

    @property
    def state_file(self):
        return self.art / "state.json"

    def test_corrupt_state_is_refused_not_overwritten(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        garbage = '{"stage": "3", "trunc'
        self.state_file.write_text(garbage)
        r = run_state("get", "stage", art=self.art)
        self.assertEqual(r.returncode, 1)
        self.assertIn("corrupt", r.stderr)
        self.assertEqual(self.state_file.read_text(), garbage)

    def test_get_missing_key_exits_1_with_no_output(self):
        run_state("init", art=self.art)
        r = run_state("get", "no.such.key", art=self.art)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")

    def test_record_is_namespaced_per_caller(self):
        run_state("record", "build", '{"status":"ok"}', art=self.art)
        run_state("record", "hipify", '{"status":"ok"}', art=self.art)
        self.assertTrue((self.art / "results" / "record-build.json").exists())
        self.assertTrue((self.art / "results" / "record-hipify.json").exists())


if __name__ == "__main__":
    unittest.main()
