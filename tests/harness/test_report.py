"""report.sh rendering contract.

The report is the redacted public face of the record, so it must never render a
limit as if it were usage (the repair-budget bug found in the audit), and a
corrupt results file must be called out as unreadable instead of quietly
reading as "never ran".
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from harness_util import REPO, _base_env

REPORT_SH = REPO / "scripts" / "rocm" / "report.sh"


def craft_state(art, repairs=None):
    (art / "state.json").write_text(json.dumps({
        "schema": 1,
        "current_revision": "b89c989a7155e984e544ddd90d1038dda7da9e3d",
        "dirty_tree": False,
        "stage": "3",
        "limits": {"repair_attempts_per_failure_signature": 3},
        "budgets": {"wall_clock_s": 0, "download_bytes": 40551103,
                    "repairs": repairs or {}},
        "gates": {"S2-kernel-suite-gfx1100":
                  {"status": "passed_with_skips", "evidence": "x", "at": "t"}},
        "tests": {"passed": 1, "failed": 0, "skipped": 0, "last": "kernels"},
        "blocker": None,
        "next_action": "test",
        "environment": {},
    }))


def craft_kernels(art):
    (art / "results").mkdir(parents=True, exist_ok=True)
    (art / "results" / "test-kernels.json").write_text(json.dumps({
        "passed": 1, "failed": 0, "skipped": 0,
        "tests": [{"name": "rope_parity", "status": "passed",
                   "seconds": 1.0, "detail": ""}],
        "coverage": {"suite": "s", "caveat": "c", "not_run": [],
                     "unsupported_sources": []},
    }))


def craft_minimal_results(art):
    """The other files report.sh loads; empty is fine, it must not crash."""
    r = art / "results"
    r.mkdir(parents=True, exist_ok=True)
    for name in ("build.json", "hipify.json", "smoke-run.json",
                 "hipify-patches.json"):
        (r / name).write_text("{}")


def run_report(art):
    out = Path(art) / "REPORT.md"
    env = _base_env(art)
    env["PATH"] = "/usr/bin:/bin:/usr/local/bin"
    env["HOME"] = str(Path(art))
    r = subprocess.run(
        ["sh", str(REPORT_SH), "--out", str(out)],
        capture_output=True, text=True, env=env,
        timeout=120,
    )
    return r, out


class ReportRepairBudgetTest(unittest.TestCase):
    def setUp(self):
        self.art = Path(tempfile.mkdtemp(prefix="strata-report-"))
        craft_minimal_results(self.art)

    def tearDown(self):
        shutil.rmtree(self.art, ignore_errors=True)

    def test_zero_repairs_renders_used_of_limit_not_the_limit_alone(self):
        craft_state(self.art, repairs={})
        craft_kernels(self.art)
        r, out = run_report(self.art)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = out.read_text()
        self.assertIn("repair attempts used: 0 of 3", text)
        self.assertNotIn("repair attempts per failure signature: 3", text)

    def test_used_repairs_are_counted_across_signatures(self):
        craft_state(self.art, repairs={
            "sig-a": {"count": 2, "outcomes": []},
            "sig-b": {"count": 1, "outcomes": []}})
        craft_kernels(self.art)
        r, out = run_report(self.art)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("repair attempts used: 3 of 3", out.read_text())


class ReportCorruptResultsTest(unittest.TestCase):
    def setUp(self):
        self.art = Path(tempfile.mkdtemp(prefix="strata-report-"))
        craft_minimal_results(self.art)

    def tearDown(self):
        shutil.rmtree(self.art, ignore_errors=True)

    def test_corrupt_kernels_file_is_named_unreadable_not_never_ran(self):
        craft_state(self.art)
        craft_kernels(self.art)
        (self.art / "results" / "test-kernels.json").write_text('{"passed": tr')
        r, out = run_report(self.art)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = out.read_text()
        self.assertIn("unreadable", text)
        self.assertNotIn("No kernel test results recorded", text)

    def test_missing_kernels_file_still_reads_as_never_ran(self):
        craft_state(self.art)
        r, out = run_report(self.art)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("No kernel test results recorded", out.read_text())

    def test_corrupt_state_does_not_crash_the_report(self):
        craft_minimal_results(self.art)
        (self.art / "state.json").write_text("{oops")
        craft_kernels(self.art)
        r, out = run_report(self.art)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("state.json", out.read_text())


if __name__ == "__main__":
    unittest.main()
