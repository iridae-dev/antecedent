"""Fail-closed controls for cyclic evidence metrics and identities."""
import math
import unittest
from unittest.mock import patch

import gate_cyclic_systems as gate

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


class CyclicSelectorChecks(unittest.TestCase):
    def test_every_selector_requires_a_selected_runnable_test_in_its_crate(self):
        checks = [("crate-a", "first"), ("crate-a", "second"), ("crate-b", None)]
        match = {"ignored": False, "filter-match": {"status": "matches"}}
        listing = {"rust-suites": {
            "crate-a": {"testcases": {"tests::first": match, "tests::second": match}},
            "crate-b": {"testcases": {"tests::anything": match}},
        }}
        gate.validate_rust_listing(listing, checks)
        for second in (None, {"ignored": True, "filter-match": {"status": "matches"}},
                       {"ignored": False, "filter-match": {"status": "mismatch"}}):
            with self.subTest(second=second):
                cases = {"tests::first": match}
                if second is not None:
                    cases["tests::second"] = second
                broken = {"rust-suites": {"crate-a": {"testcases": cases},
                                          "crate-b": {"testcases": {"tests::second": match}}}}
                with self.assertRaisesRegex(ValueError, "crate-a / second"):
                    gate.validate_rust_listing(broken, checks)

    def test_success_runs_the_same_combined_selection_once(self):
        from types import SimpleNamespace
        listing = '{"rust-suites": {"crate": {"testcases": {"tests::first": {"ignored": false, "filter-match": {"status": "matches"}}, "tests::second": {"ignored": false, "filter-match": {"status": "matches"}}}}}}'
        with patch.object(gate, "RUST_CHECKS", [("crate", "first"), ("crate", "second")]), \
             patch.object(gate.subprocess, "run", return_value=SimpleNamespace(stdout=listing)) as list_run, \
             patch.object(gate, "run") as run:
            gate.run_rust_checks()
            list_command = list_run.call_args.args[0]
            run_command = run.call_args.args[0]
            self.assertEqual(list_command[list_command.index("--lib"):list_command.index("--lib") + 3],
                             run_command[run_command.index("--lib"):run_command.index("--lib") + 3])
            run.assert_called_once()

    def test_listing_failure_prevents_combined_run(self):
        from types import SimpleNamespace
        listing = '{"rust-suites": {}}'
        with patch.object(gate, "RUST_CHECKS", [("crate", "missing")]), \
             patch.object(gate.subprocess, "run", return_value=SimpleNamespace(stdout=listing)), \
             patch.object(gate, "run") as run:
            with self.assertRaises(ValueError):
                gate.run_rust_checks()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
