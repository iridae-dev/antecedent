import unittest
from calibration_recipe_receipts import original_gate_passed

class OriginalGateResult(unittest.TestCase):
    def test_success_does_not_mask_a_failed_test_or_suite(self):
        ok = b"test result: ok. 1 passed; 0 failed; 0 ignored\n"
        self.assertTrue(original_gate_passed(ok, exactly_one=True))
        self.assertFalse(original_gate_passed(ok + b"test x ... FAILED\n"))
        self.assertFalse(original_gate_passed(ok + b"test result: FAILED. 1 passed; 1 failed;\n"))
        self.assertFalse(original_gate_passed(b"test result: ok. 0 passed; 0 failed;\n"))
