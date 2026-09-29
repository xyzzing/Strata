"""Unit tests for the tune controller's pure logic (no GPU, no engine)."""

import unittest

import sys
sys.path.insert(0, "~/projects/strata-amd/scripts/rocm/lib")

from tune import parse_counters, decide, with_flags, parse_sets


class ParseCountersTest(unittest.TestCase):
    def test_extracts_latest_prompt_and_decode(self):
        log = (
            'strata serve: prompt 61 tokens = 0 reused + 61 read in 1293 ms (47.2 tok/s), 36 generated in 928 ms (38.8 tok/s)\n'
            'strata serve: prompt 3200 tokens = 0 reused + 3200 read in 6100 ms (524.6 tok/s), 48 generated in 1100 ms (43.6 tok/s)\n'
        )
        p, d = parse_counters(log)
        self.assertEqual(p, 524.6)
        self.assertEqual(d, 43.6)

    def test_missing_lines_give_none(self):
        self.assertEqual(parse_counters("unrelated text"), (None, None))

    def test_reuse_lines_still_parse(self):
        log = 'strata serve: prompt 100 tokens = 59 reused + 41 read in 636 ms (64.5 tok/s), 64 generated in 1404 ms (45.6 tok/s)'
        p, d = parse_counters(log)
        self.assertEqual((p, d), (64.5, 45.6))


class DecideTest(unittest.TestCase):
    def test_gain_above_threshold_accepts(self):
        ok, why = decide(40.0, 44.0, 50.0, 50.0, min_gain=0.03)
        self.assertTrue(ok)
        self.assertIn("+10.0%", why)

    def test_gain_below_threshold_rejects(self):
        ok, why = decide(40.0, 41.0, 50.0, 50.0, min_gain=0.03)
        self.assertFalse(ok)
        self.assertIn("< 3%", why)

    def test_prompt_regression_rejects(self):
        ok, why = decide(40.0, 50.0, 100.0, 50.0, min_gain=0.03)
        self.assertFalse(ok)
        self.assertIn("regression", why)

    def test_missing_counters_reject(self):
        ok, why = decide(None, 44.0, 50.0, 50.0)
        self.assertFalse(ok)


class WithFlagsTest(unittest.TestCase):
    def test_replaces_existing_value(self):
        out = with_flags(["--prefill", "1024", "--spec", "2"], ["--prefill", "auto"])
        self.assertEqual(out, ["--spec", "2", "--prefill", "auto"])

    def test_appends_new_flag(self):
        out = with_flags(["--spec", "2"], ["--kv-resident", "20480"])
        self.assertEqual(out, ["--spec", "2", "--kv-resident", "20480"])

    def test_bare_flag_without_value(self):
        out = with_flags(["--spec", "2"], ["--mmap-experts"])
        self.assertEqual(out, ["--spec", "2", "--mmap-experts"])

    def test_replaces_bare_with_valued(self):
        out = with_flags(["--mmap-experts"], ["--mmap-experts", "x"])
        self.assertEqual(out, ["--mmap-experts", "x"])


if __name__ == "__main__":
    unittest.main()


class ParseSetsTest(unittest.TestCase):
    def test_valued_and_bare(self):
        self.assertEqual(parse_sets(["prefill=auto", "mmap-experts"]),
                         ["--prefill", "auto", "--mmap-experts"])

    def test_numeric_value(self):
        self.assertEqual(parse_sets(["kv-resident=20480"]),
                         ["--kv-resident", "20480"])
