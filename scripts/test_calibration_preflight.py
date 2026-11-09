import json
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_decision as decision
import calibration_preflight as preflight
import calibration_runner as runner


class FormalMeasurementPreflight(unittest.TestCase):
    def test_missing_malformed_and_foreign_context_fail(self):
        config = decision.config_for_manifest({"groups": [{"label": "x"}]})
        for text in (None, "{", json.dumps({}), decision.canonical_json({**config, "family_digest": "f" * 64})):
            with self.subTest(text=text), self.assertRaises(ValueError):
                preflight.validate_inherited(text, config)
        preflight.validate_inherited(decision.canonical_json(config), config)

    def test_a_formal_run_without_the_frozen_context_measures_nothing(self):
        task = runner.Task(runner.GroupSpec(1, "s: t", "antecedent", "s", (), ("t",), True, False), None)
        config = decision.config_for_manifest({"groups": [{"label": "x"}]})
        with (
            patch("calibration_family.current_config", return_value=config),
            patch("nix_evidence_build.build_committed_targets") as build,
            self.assertRaises(ValueError),
        ):
            runner.execute(Path("."), [task], env={}, log_dir=Path("unused"), jobs=1,
                           first_nsim=400, recheck_nsim=2000)
        build.assert_not_called()

    def test_a_smoke_run_needs_no_decision_context(self):
        task = runner.Task(runner.GroupSpec(1, "s: t", "antecedent", "s", (), ("t",), True, False), None)
        with (
            patch("calibration_family.current_config") as config,
            patch("nix_evidence_source._clean_head", side_effect=RuntimeError("stop before building")),
            self.assertRaisesRegex(RuntimeError, "stop before building"),
        ):
            runner.execute(Path("."), [task], env={}, log_dir=Path("unused"), jobs=1,
                           first_nsim=8, recheck_nsim=9, smoke=True)
        config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
