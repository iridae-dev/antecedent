import os
from pathlib import Path
import subprocess
import tempfile
import unittest

GATE = Path(__file__).with_name("gate_promotion.sh")

class PromotionOverrideGuard(unittest.TestCase):
    def test_ordinary_entry_rejects_overrides_and_selftest_allows_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "python3"
            fake.write_text("#!/bin/sh\necho MOCK\nexit 0\n")
            fake.chmod(0o755)
            env = dict(os.environ, PATH=tmp + os.pathsep + os.environ["PATH"])
            names = ("PROMOTION_TRANSPORT_STAGES", "PROMOTION_SUPPORT_LICENSED",
                     "PROMOTION_SUPPORT_CLOSED", "PROMOTION_EXTRA_SOURCES",
                     "PROMOTION_DISABLE_RULES", "PROMOTION_SYNTHETIC_EVIDENCE",
                     "PROMOTION_EVIDENCE_CACHE", "PROMOTION_STATIC_ONLY")
            for name in names:
                with self.subTest(name=name):
                    controlled = {k: v for k, v in env.items() if k not in names}
                    controlled[name] = "override"
                    ordinary = subprocess.run(["bash", str(GATE)], env=controlled,
                                              capture_output=True, text=True)
                    self.assertEqual(ordinary.returncode, 2)
                    self.assertIn(name, ordinary.stderr)
                    self.assertNotIn("MOCK", ordinary.stdout)
                    explicit = subprocess.run(["bash", str(GATE), "--self-test"],
                                              env=controlled, capture_output=True, text=True)
                    self.assertEqual(explicit.returncode, 0)
                    self.assertIn("MOCK", explicit.stdout)
