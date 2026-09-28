"""run.sh must refuse the forbidden port explicitly, before any other check.

Port 8080 belongs to the user's llama.cpp server. A free-looking 8080 (say,
after the user's own server is down for maintenance) must still be refused:
this rollout never binds it, by rule.
"""

import unittest

from harness_util import SCRIPTS, run_script


class RunPortRefusalTest(unittest.TestCase):
    def test_port_8080_is_refused_explicitly(self):
        r = run_script(SCRIPTS / "run.sh", "--port", "8080")
        self.assertNotEqual(r.returncode, 0)
        combined = r.stdout + r.stderr
        self.assertIn("refusing", combined.lower())
        self.assertIn("8080", combined)

    def test_other_ports_are_not_caught_by_the_refusal(self):
        # 8081 (the normal target) must get past the refusal and fail later on
        # GPU visibility inside the sandbox - but never with the refusal text.
        r = run_script(SCRIPTS / "run.sh", "--port", "8081")
        combined = r.stdout + r.stderr
        self.assertNotIn("refusing", combined.lower())


if __name__ == "__main__":
    unittest.main()
