import json
import tempfile
import unittest
from pathlib import Path

import calibration_decision as decision
from calibration_decision_logs import audit_logs


class DecisionLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sha = "a" * 40
        self.roster = [{"index": 1, "label": "suite: test", "grid": True}]
        self.config = decision.config_for_manifest(
            {"groups": self.roster, "validation_domain": decision.LEGACY_DOMAIN}
        )

    def write(self, name="suite__test.p0.log", planned=400, covered=380, record=True, skip=0):
        log = self.root / name
        value = decision.evaluate(self.config, planned, planned, covered, 0.95, False)
        row = {
            "test": "test",
            "name": "tally",
            "nominal": 0.95,
            "record_id": "record" if record else None,
            "grid_point": 0,
            "measured_at": self.sha,
            "decision_protocol": value,
        }
        lines = ["calibration-decision " + json.dumps(row)]
        if record:
            lines += [
                "calibration-record "
                + json.dumps(
                    {
                        "id": "record",
                        "nominal": 0.95,
                        "measured_at": self.sha,
                        "grid_point": 0,
                        "replicates": planned,
                        "observed": covered / planned,
                        "decision_protocol": value,
                    }
                )
            ]
        lines += [
            "calibration-measured-at " + self.sha,
            "calibration-decision-context " + json.dumps(self.config),
        ]
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.with_suffix(".tallies.jsonl").write_text(
            json.dumps(
                {
                    "key": "test::tally@0.950000",
                    "covered": covered,
                    "scored": planned - skip,
                    "skipped": skip,
                    "map_end": planned,
                    "decision_context": {
                        "config": self.config,
                        "measured_at": self.sha,
                        "grid_point": 0,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return log

    def audit(self, logs):
        return audit_logs(logs, self.sha, expected_config=self.config, roster=self.roster)

    def test_followup_receipt_cannot_collect_another_construction_or_recordless_cell(self):
        import calibration_family as family

        self.roster = json.loads(
            family.read_source(
                family.ROOT, family.MANIFEST_PATH, family.allocations.ORIGINAL_SOURCE
            )
        )["groups"]
        self.config = decision.config_for_manifest(
            {"groups": self.roster, "validation_domain": decision.PAG_CONFIDENCE_SET_DOMAIN}
        )
        label, construction = family.followup_allocation(self.config)
        stem = label.translate(str.maketrans(" /:", "___"))
        log = self.write(name=stem + ".p0.log")
        with self.assertRaisesRegex(ValueError, "unallocated"):
            self.audit([log])
        lines = []
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.startswith("calibration-record "):
                row = json.loads(line.partition(" ")[2])
                row.update(
                    {
                        k: construction[k]
                        for k in ("functional", "interval_method", "se_kind", "nominal")
                    }
                )
                row["id"] = construction["record_id"]
                line = "calibration-record " + json.dumps(row)
            elif line.startswith("calibration-decision "):
                row = json.loads(line.partition(" ")[2])
                row["record_id"] = construction["record_id"]
                line = "calibration-decision " + json.dumps(row)
            lines.append(line)
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertTrue(self.audit([log]))
        log = self.write(name=stem + ".p0.log", record=False)
        with self.assertRaisesRegex(ValueError, "absent construction"):
            self.audit([log])

    def test_raw_record_binding_preserves_valid_cross_language_roundoff(self):
        log = self.write(planned=20000, covered=19000)
        lines = log.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.startswith(("calibration-decision ", "calibration-record ")):
                prefix, _, payload = line.partition(" ")
                row = json.loads(payload)
                row["decision_protocol"]["p_value"] *= 1 - 1e-12
                lines[index] = prefix + " " + json.dumps(row)
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertIn(log.name, self.audit([log]))
        record = json.loads(lines[1].partition(" ")[2])
        record["decision_protocol"]["p_value"] *= 1 - 1e-12
        lines[1] = "calibration-record " + json.dumps(record)
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not bound"):
            self.audit([log])

    def test_all_looks_and_recordless_decisions_audited(self):
        first = self.write(record=False, skip=2)
        second = self.write("suite__test.p0.recheck.log", planned=2000, covered=1900, record=False)
        self.assertEqual(len(self.audit([first, second])), 2)

    def test_incompatible_intermediate_is_not_discarded(self):
        first = self.write(covered=300)
        second = self.write("suite__test.p0.recheck.log", planned=2000, covered=1900)
        rows = self.audit([first, second])
        self.assertFalse(rows[first.name][0]["decision_protocol"]["compatible"])
        self.assertTrue(rows[second.name][0]["decision_protocol"]["compatible"])

    def test_raw_record_binding_and_complete_attempt_state_required(self):
        for mutation in ("sidecar", "record", "source", "group"):
            with self.subTest(mutation=mutation):
                log = self.write()
                if mutation == "sidecar":
                    log.with_suffix(".tallies.jsonl").unlink()
                elif mutation == "record":
                    log.write_text(
                        log.read_text(encoding="utf-8").replace(
                            '"record_id": "record"', '"record_id": null'
                        ),
                        encoding="utf-8",
                    )
                elif mutation == "source":
                    log.write_text(
                        log.read_text(encoding="utf-8").replace(self.sha, "b" * 40),
                        encoding="utf-8",
                    )
                else:
                    new = log.with_name("unknown.p0.log")
                    log.rename(new)
                    log = new
                with self.assertRaises(ValueError):
                    self.audit([log])

    def test_same_look_different_counts_and_fresh_confirmation_rejected(self):
        first = self.write()
        conflicting = self.write("suite__test.p0.recheck.log", covered=381)
        with self.assertRaisesRegex(ValueError, "conflicting repeat"):
            self.audit([first, conflicting])
        confirmation = self.write("suite__test.p0.confirm.log")
        with self.assertRaisesRegex(ValueError, "confirmation"):
            self.audit([confirmation])

    def test_recordless_recheck_successor_and_final_compatibility_required(self):
        log = self.write(record=False, covered=300)
        with self.assertRaisesRegex(ValueError, "final recordless"):
            self.audit([log])
        log.write_text(
            log.read_text(encoding="utf-8")
            + "calibration-recheck reason\ncalibration-recheck-test test\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "successor"):
            self.audit([log])
        successor = self.write(
            "suite__test.p0.recheck.log", planned=2000, covered=1900, record=False
        )
        self.audit([log, successor])

    def test_qualified_test_recheck_request_cannot_disappear(self):
        log = self.write(record=False)
        log.write_text(
            log.read_text(encoding="utf-8").replace(
                '"test": "test"', '"test": "module::tests::test"'
            )
            + "calibration-recheck reason\ncalibration-recheck-test module::tests::test\n",
            encoding="utf-8",
        )
        side = log.with_suffix(".tallies.jsonl")
        side.write_text(
            side.read_text(encoding="utf-8").replace("test::tally@", "module::tests::test::tally@"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "successor"):
            self.audit([log])

    def test_named_boundary_is_separate_from_coverage_family(self):
        nominal = self.write()
        boundary = self.root / "suite__test.p1.log"
        boundary.write_text(
            "calibration-record "
            + json.dumps({"id": "boundary", "role": "named_boundary"})
            + "\ncalibration-measured-at "
            + self.sha
            + "\ncalibration-decision-context "
            + json.dumps(self.config)
            + "\n",
            encoding="utf-8",
        )
        self.audit([nominal, boundary])
        nominal.write_text(
            nominal.read_text(encoding="utf-8")
            + "calibration-record "
            + json.dumps({"id": "boundary", "role": "named_boundary"})
            + "\n",
            encoding="utf-8",
        )
        self.audit([nominal])

    def test_domain_context_and_grid_owner_are_not_interchangeable(self):
        log = self.write()
        side = log.with_suffix(".tallies.jsonl")
        state = json.loads(side.read_text(encoding="utf-8"))
        state["decision_context"]["config"]["validation_domain"] = "another_domain"
        side.write_text(json.dumps(state) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "context"):
            self.audit([log])

    def test_nongrid_receipt_uses_the_actual_common_base_grid_point(self):
        from calibration_decision_logs import validate_ledger

        self.roster[0]["grid"] = False
        log = self.write("suite__test.log", record=False, planned=2000, covered=1900)
        for path in (log, log.with_suffix(".tallies.jsonl")):
            path.write_text(
                path.read_text(encoding="utf-8").replace('"grid_point": 0', '"grid_point": 1'),
                encoding="utf-8",
            )
        audited = self.audit([log])
        row = {
            "group": "suite__test",
            "calibration_sha": self.sha,
            "decision_protocol": self.config,
            "coverage_decisions": [
                {"log": name, **item, "record_id": ""}
                for name, items in audited.items()
                for item in items
            ],
        }
        validate_ledger(row, self.config, self.roster)
        self.assertEqual(row["coverage_decisions"][0]["grid_point"], 1)
        row["coverage_decisions"][0]["grid_point"] = 0
        with self.assertRaisesRegex(ValueError, "owner"):
            validate_ledger(row, self.config, self.roster)

    def test_durable_ledger_retains_all_counts_and_checks_final_decision(self):
        from calibration_decision_logs import validate_ledger

        first = self.write(record=False)
        successor = self.write(
            "suite__test.p0.recheck.log", planned=2000, covered=1900, record=False
        )
        audited = self.audit([first, successor])
        row = {
            "group": "suite__test.p0",
            "calibration_sha": self.sha,
            "decision_protocol": self.config,
            "coverage_decisions": [
                {"log": name, **item, "record_id": ""}
                for name, items in audited.items()
                for item in items
            ],
        }
        validate_ledger(row, self.config, self.roster)
        row["coverage_decisions"][0]["measured_at"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "owner"):
            validate_ledger(row, self.config, self.roster)

    def test_noncoverage_group_preserves_domain_without_inventing_a_coverage_claim(
        self,
    ):
        from calibration_decision_logs import validate_ledger

        log = self.root / "suite__test.p0.log"
        log.write_text(
            "calibration-measured-at "
            + self.sha
            + "\ncalibration-decision-context "
            + json.dumps(self.config)
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(self.audit([log]), {log.name: []})
        validate_ledger(
            {
                "group": "suite__test.p0",
                "calibration_sha": self.sha,
                "decision_protocol": self.config,
                "coverage_decisions": [],
            },
            self.config,
            self.roster,
        )

    def test_legacy_logs_stay_legacy(self):
        log = self.root / "legacy.log"
        log.write_text("test result: ok. 1 passed; 0 failed;\n", encoding="utf-8")
        self.assertEqual(audit_logs([log], self.sha), {})


if __name__ == "__main__":
    unittest.main()
