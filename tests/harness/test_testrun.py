"""testrun.py classification contract: passed / failed / skipped, never conflated.

The fake binaries here double as the S1.6 harness self-check: the injected
defect (fail_parity) must classify as failed, and a declared-unavailable test
that fails for a different reason must NOT earn the skip.
"""

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness_util import PYTHON, REPO

TESTRUN = REPO / "scripts" / "rocm" / "lib" / "testrun.py"

PASS = "#!/bin/sh\nexit 0\n"
FAIL = '#!/bin/sh\necho "value mismatch: got 1 want 2" >&2\nexit 1\n'
SKIP_OK = '#!/bin/sh\necho "required PLE table is missing or incompatible" >&2\nexit 2\n'
SKIP_DECLARED_OTHER = '#!/bin/sh\necho "completely unexpected crash" >&2\nexit 2\n'
USAGE = "#!/bin/sh\necho 'usage: native_expert_parity <shard1.gguf>' >&2\nexit 2\n"


def make_bin(dirpath, name, body):
    p = dirpath / name
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestrunClassificationTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="strata-testrun-"))
        self.out = self.dir / "out.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_testrun(self, timeout=10):
        cmd = [PYTHON, str(TESTRUN),
               "--dir", str(self.dir), "--out", str(self.out),
               "--timeout", str(timeout)]
        decl = self.dir / "unavailable.json"
        if decl.exists():
            cmd += ["--unavailable", str(decl)]
        return subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout * len(os.listdir(self.dir)) + 30,
        )

    def status_of(self, name):
        doc = json.loads(self.out.read_text())
        return next(t["status"] for t in doc["tests"] if t["name"] == name)

    def write_unavailable(self, table):
        path = self.dir / "unavailable.json"
        path.write_text(json.dumps({"tests": table}))
        return path

    def test_pass_exits_zero_and_counts_passed(self):
        make_bin(self.dir, "pass_parity", PASS)
        r = self.run_testrun()
        self.assertEqual(r.returncode, 0)
        doc = json.loads(self.out.read_text())
        self.assertEqual(doc["passed"], 1)
        self.assertEqual(doc["failed"], 0)
        self.assertEqual(doc["skipped"], 0)

    def test_injected_defect_classifies_as_failed_with_rc4(self):
        make_bin(self.dir, "fail_parity", FAIL)
        r = self.run_testrun()
        self.assertEqual(r.returncode, 4)
        self.assertEqual(self.status_of("fail_parity"), "failed")

    def test_declared_signature_earns_skip_with_rc5(self):
        make_bin(self.dir, "ple_parity", SKIP_OK)
        self.write_unavailable({
            "ple_parity": {"signature": "required PLE table is missing",
                           "reason": "PLE table not published upstream"}})
        r = self.run_testrun()
        self.assertEqual(r.returncode, 5)
        self.assertEqual(self.status_of("ple_parity"), "skipped")

    def test_declared_but_different_failure_is_failed_not_skipped(self):
        make_bin(self.dir, "skipfail_parity", SKIP_DECLARED_OTHER)
        self.write_unavailable({
            "skipfail_parity": {"signature": "required PLE table is missing",
                                "reason": "PLE table not published upstream"}})
        r = self.run_testrun()
        self.assertEqual(r.returncode, 4)
        self.assertEqual(self.status_of("skipfail_parity"), "failed")

    def test_usage_signature_earns_skip(self):
        make_bin(self.dir, "native_expert_parity", USAGE)
        self.write_unavailable({
            "native_expert_parity": {"signature": r"usage: native_expert_parity <shard1\.gguf>",
                                     "reason": "needs a GGUF shard (missing input)"}})
        r = self.run_testrun()
        self.assertEqual(r.returncode, 5)
        self.assertEqual(self.status_of("native_expert_parity"), "skipped")

    def test_timeout_is_failed_and_named_as_timeout(self):
        make_bin(self.dir, "hang_parity", "#!/bin/sh\nsleep 30\n")
        r = self.run_testrun(timeout=1)
        self.assertEqual(r.returncode, 4)
        self.assertEqual(self.status_of("hang_parity"), "failed")
        doc = json.loads(self.out.read_text())
        detail = next(t["detail"] for t in doc["tests"] if t["name"] == "hang_parity")
        self.assertIn("TIMEOUT", detail)

    def test_invalid_skip_regex_fails_that_test_without_killing_the_suite(self):
        make_bin(self.dir, "badregex_parity", SKIP_OK)
        make_bin(self.dir, "pass_parity", PASS)
        self.write_unavailable({
            "badregex_parity": {"signature": "(unclosed",
                                "reason": "never matches"}})
        r = self.run_testrun()
        # The suite must complete and classify the healthy test; the test whose
        # declaration is broken is reported failed, not skipped, not a crash.
        self.assertEqual(r.returncode, 4)
        doc = json.loads(self.out.read_text())
        self.assertEqual(self.status_of("pass_parity"), "passed")
        bad = next(t for t in doc["tests"] if t["name"] == "badregex_parity")
        self.assertEqual(bad["status"], "failed")
        self.assertIn("invalid skip signature", bad["detail"])


if __name__ == "__main__":
    unittest.main()
