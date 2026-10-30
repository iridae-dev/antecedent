"""Fail-closed controls for cyclic evidence metrics and identities."""
import math
import unittest

from gate_cyclic_systems import finite_at_most, valid_digest


class CyclicEvidenceChecks(unittest.TestCase):
    def test_digest_is_exact_lowercase_hex(self):
        self.assertTrue(valid_digest("a" * 64))
        for value in ("g" * 64, "A" * 64, "a" * 63, None):
            self.assertFalse(valid_digest(value))

    def test_nonfinite_metrics_never_pass(self):
        self.assertTrue(finite_at_most(1e-8, 1e-7))
        for value, limit in ((math.nan, 1.0), (math.inf, 1.0), (0.0, math.nan), (-1.0, 1.0)):
            self.assertFalse(finite_at_most(value, limit))


if __name__ == "__main__":
    unittest.main()
