"""worst_rc(): the severity ordering that lets `test --stage all` aggregate.

Contract (low to high): 0 ok, 3 blocked, 5 skip, 124 timeout, 1 failure,
4 test failure. A test failure outranks everything because it is the suite's
actual verdict; a definitive failure outranks a timeout because "what do I do
next" must not be hidden behind an ambiguous hang.
"""

import unittest

from harness_util import run_sh_snippet


def worst_rc(*codes):
    r = run_sh_snippet(f"worst_rc {' '.join(str(c) for c in codes)}")
    return r.stdout.strip(), r


class WorstRcSeverityTest(unittest.TestCase):
    def test_all_ok_is_zero(self):
        out, _ = worst_rc(0, 0, 0)
        self.assertEqual(out, "0")

    def test_single_ok(self):
        out, _ = worst_rc(0)
        self.assertEqual(out, "0")

    def test_skip_outranks_ok(self):
        out, _ = worst_rc(0, 5)
        self.assertEqual(out, "5")

    def test_skip_outranks_blocked(self):
        out, _ = worst_rc(5, 3)
        self.assertEqual(out, "5")

    def test_timeout_outranks_blocked(self):
        out, _ = worst_rc(3, 124)
        self.assertEqual(out, "124")

    def test_failure_outranks_timeout(self):
        out, _ = worst_rc(124, 1)
        self.assertEqual(out, "1")

    def test_test_failure_outranks_generic_failure(self):
        out, _ = worst_rc(1, 4)
        self.assertEqual(out, "4")

    def test_test_failure_outranks_timeout(self):
        out, _ = worst_rc(4, 124)
        self.assertEqual(out, "4")

    def test_test_failure_outranks_skip_and_blocked(self):
        out, _ = worst_rc(3, 5, 4)
        self.assertEqual(out, "4")

    def test_test_failure_outranks_ok(self):
        out, _ = worst_rc(4, 0)
        self.assertEqual(out, "4")

    def test_unknown_nonzero_ranks_as_failure_but_keeps_its_code(self):
        out, r = worst_rc(0, 7)
        self.assertEqual(out, "7")
        self.assertEqual(r.returncode, 0)

    def test_unknown_nonzero_loses_to_test_failure(self):
        out, _ = worst_rc(7, 4)
        self.assertEqual(out, "4")


class CommonShRegressionLocks(unittest.TestCase):
    def test_sourcing_common_sh_in_pure_posix_sh_succeeds(self):
        r = run_sh_snippet("have sh && echo ok")
        self.assertEqual(r.returncode, 0)
        self.assertIn("ok", r.stdout)


if __name__ == "__main__":
    unittest.main()
