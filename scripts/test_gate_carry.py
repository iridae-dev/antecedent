"""A pass/fail calibration gate pass is carried while nothing it executed changed."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_groups as groups
import gate_carry as carry

SHA = "a" * 40
FILES = [
    "crates/antecedent-discovery/src/structure_mcmc.rs",
    "crates/antecedent-core/src/ids.rs",
]


def row(name="g", files=FILES, passed=1, sha=SHA):
    out = {"group": name, "passed": passed, "calibration_sha": sha}
    if files is not None:
        out["executed_files"] = files
    return out


class StaleInputs(unittest.TestCase):
    def stale(self, *changed):
        return carry.stale_inputs(FILES, SHA, frozenset(changed))

    def test_unrelated_changes_do_not_stale_a_pass(self):
        self.assertEqual(
            self.stale(
                "crates/antecedent-discovery/src/ci_screened_posterior.rs",
                "crates/antecedent-io-core/data/coverage_records.json",
                "parity/coverage_records.toml",
                "docs/graph_uncertainty.md",
            ),
            [],
        )

    def test_an_executed_file_stales_it(self):
        self.assertEqual(self.stale(FILES[0]), [FILES[0]])

    def test_workspace_and_executed_crate_build_inputs_stale_it(self):
        for path in (
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config.toml",
            "crates/antecedent-core/Cargo.toml",
            "crates/antecedent-core/build.rs",
        ):
            self.assertEqual(self.stale(path), [path], path)
        self.assertEqual(self.stale("crates/antecedent-io-core/Cargo.toml"), [])

    def test_executed_files_come_from_workspace_sources_only(self):
        coverage = {
            "/antecedent-source/crates/a/src/x.rs": [[1, 1]],
            "/antecedent-source/crates/a/src/empty.rs": [],
            "/rustc/abc/library/std/src/lib.rs": [[1, 1]],
        }
        self.assertEqual(carry.executed_files(coverage), ["crates/a/src/x.rs"])


class GroupCarried(unittest.TestCase):
    def test_carried_only_when_every_point_passed_and_is_bound(self):
        none = {SHA: frozenset()}
        self.assertTrue(carry.group_carried([row("a.p0"), row("a.p1")], none)[0])
        self.assertFalse(carry.group_carried([row("a.p0"), None], none)[0])
        self.assertFalse(carry.group_carried([row("a.p0", passed=0)], none)[0])
        why = carry.group_carried([row("a.p0", files=None)], none)[1]
        self.assertIn("commit only", why)
        stale = carry.group_carried([row("a")], {SHA: frozenset({FILES[1]})})
        self.assertFalse(stale[0])
        self.assertIn(FILES[1], stale[1])


    def test_review_carries_only_the_original_receipts_and_reviewed_inputs(self):
        receipt = [row("g.p0"), row("g.p1")]
        definition = {"label": "g", "tests": ["original"]}
        sha = "b" * 40
        fixture = "conformance/science/fixture.json"
        reviews = {("g", carry.receipt_digest(definition, receipt)): (sha, frozenset({fixture}))}
        changes = {SHA: frozenset(FILES), sha: frozenset({"docs/notes.md"})}
        with (
            patch.object(carry, "gate_definitions", return_value={"g": definition}),
            patch.object(carry, "reviewed_receipts", return_value=reviews),
        ):
            self.assertTrue(carry.group_carried(receipt, changes)[0])
            for path in (FILES[0], "Cargo.lock", fixture, "scripts/calibration_family_manifest.json"):
                self.assertFalse(carry.group_carried(receipt, changes | {sha: frozenset({path})})[0])
            self.assertFalse(carry.group_carried(receipt[:1], changes)[0])
            self.assertFalse(carry.group_carried([receipt[0], row("g.p1", passed=2)], changes)[0])
            definition["tests"] = ["different"]
            self.assertFalse(carry.group_carried(receipt, changes)[0])


class SelectSkipsCarriedGates(unittest.TestCase):
    def test_a_carried_recordless_group_is_not_planned(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "gates.toml"
            ledger.write_text(
                f'[[gate]]\ngroup = "crate__t"\npassed = 1\ncalibration_sha = "{SHA}"\nexecuted_files = ["x/y.rs"]\n',
                encoding="utf-8",
            )
            unbound = groups.Group(2, "crate: other", False, False)
            carried = groups.Group(3, "crate: t", False, False)
            suite = groups.Group(1, "suite_calibration: s", False, False)
            for group in [unbound, carried]:
                group.evidence = "gate"
            with (
                patch.object(groups, "GATE_REGISTRY", ledger),
                patch.object(carry, "changed_since", return_value=frozenset()),
            ):
                items = groups.group_items([suite, unbound, carried], [])
        self.assertEqual([i["groups"] for i in items], [[suite.label], [unbound.label]])


class Collector(unittest.TestCase):
    def test_ledger_row_round_trips_executed_files(self):
        import collect_coverage_records as collector

        rows = {"g": {**row("g"), "decision_protocol": None}}
        del rows["g"]["decision_protocol"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gates.toml"
            with patch.object(collector, "GATES", path):
                collector.write_gate_ledger(rows)
            self.assertEqual(carry.gate_rows(path)["g"]["executed_files"], FILES)


if __name__ == "__main__":
    unittest.main()
