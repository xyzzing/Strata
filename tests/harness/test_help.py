"""--help must print only the comment header, never a line of live code.

Every script extracts its usage text with sed over its own source; the ranges
had drifted one line past the header, so `-h` output ended with a raw
`. "…/lib/common.sh"` line. The assertion is on the leak signature, not on
exact text, so re-flowed headers stay green.
"""

import unittest

from harness_util import REPO, SCRIPTS, run_script

# (script, flag). Only the scripts that implement a help flag; status, report
# and fixtures have no -h and would otherwise run their main body.
CASES = [
    (REPO / "strata-rocm", "--help"),
    (SCRIPTS / "doctor.sh", "-h"),
    (SCRIPTS / "prepare.sh", "-h"),
    (SCRIPTS / "hipify.sh", "-h"),
    (SCRIPTS / "build.sh", "-h"),
    (SCRIPTS / "test.sh", "-h"),
    (SCRIPTS / "bench.sh", "-h"),
    (SCRIPTS / "run.sh", "-h"),
    (SCRIPTS / "smoke.sh", "-h"),
]


class HelpHasNoCodeLeakTest(unittest.TestCase):
    def test_help_prints_header_only(self):
        for script, flag in CASES:
            with self.subTest(script=script.name):
                r = run_script(script, flag)
                self.assertEqual(r.returncode, 0, f"{script.name}: {r.stderr}")
                self.assertTrue(r.stdout.strip(), f"{script.name}: empty help")
                self.assertNotIn("CDPATH=", r.stdout,
                                 f"{script.name}: help leaks a source line")
                self.assertNotIn("set -u", r.stdout,
                                 f"{script.name}: help leaks a source line")
                self.assertNotIn("sed -n", r.stdout,
                                 f"{script.name}: help leaks a source line")


if __name__ == "__main__":
    unittest.main()
