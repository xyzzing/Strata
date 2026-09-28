"""ggufpy_reference.py --json: machine-readable fixture provenance.

fixtures.sh must be able to record which decoder produced the .f32 reference
(gguf-py vs ggml's C code) into results/, so a later "iq_parity passed" can be
reconciled against reference independence. Runs against copies of the real
generated fixtures; skipped, loudly, when the fixtures or the venv are absent.
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from harness_util import GGUF_PY, IQ_FIXTURES, VENV_PY

SCRIPT = Path(__file__).resolve().parents[1] / ".." / "scripts" / "rocm" / "ggufpy_reference.py"
SCRIPT = (Path(__file__).resolve().parents[2]
          / "scripts" / "rocm" / "ggufpy_reference.py")

NEEDS = [VENV_PY.exists(), GGUF_PY.exists(), (IQ_FIXTURES / "Q2_0.bin").exists(),
         (IQ_FIXTURES / "Q2_0.f32").exists()]


@unittest.skipUnless(all(NEEDS),
                     "needs the project venv, gguf-py and generated IQ fixtures "
                     "(run ./strata-rocm prepare && ./strata-rocm fixtures)")
class GgufpyReferenceJsonTest(unittest.TestCase):
    def setUp(self):
        self.art = Path(tempfile.mkdtemp(prefix="strata-ggufpy-"))
        shutil.copy(IQ_FIXTURES / "Q2_0.bin", self.art / "Q2_0.bin")
        shutil.copy(IQ_FIXTURES / "Q2_0.f32", self.art / "Q2_0.f32")

    def tearDown(self):
        shutil.rmtree(self.art, ignore_errors=True)

    def run_ref(self, *extra):
        env = {"PATH": "/usr/bin:/bin",
               "PYTHONPATH": str(GGUF_PY),
               "HOME": str(self.art)}
        return subprocess.run(
            [str(VENV_PY), str(SCRIPT), str(self.art), *extra],
            capture_output=True, text=True, env=env, timeout=120,
        )

    def test_json_flag_writes_provenance(self):
        r = self.run_ref("--json", str(self.art / "prov.json"))
        self.assertEqual(r.returncode, 0, r.stderr)
        prov = json.loads((self.art / "prov.json").read_text())
        self.assertIn("Q2_0", prov["provenance"])
        self.assertIn(prov["provenance"]["Q2_0"], ("gguf-py", "ggml-c"))
        # Q2_0 was verified bit-exact in the audit; if both decoders exist they
        # must agree, and the summary says which one produced the file.
        self.assertTrue(prov["replaced"] or prov["kept"],
                        "Q2_0 must be either replaced or explicitly kept")

    def test_without_json_no_file_is_written_and_exit_is_zero(self):
        r = self.run_ref()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.art / "prov.json").exists())


if __name__ == "__main__":
    unittest.main()
