"""A unit is supported only while every scientific check it cites has a current passing record."""
import tomllib
import unittest
from unittest.mock import patch

import gate_support
import generate_support_matrix_docs as generator


class FakeLedger(gate_support.Ledger):
    """Records current for exactly the (test, assertion) pairs given; everything else is owed."""

    def __init__(self, current=()):
        self.current = set(current)

    def owed(self, test, name):
        return None if (test, name) in self.current else f"fake::{test}::{name}"

    def counts(self):
        return len(self.current), 0


def first_cell():
    cells = tomllib.loads((gate_support.ROOT / gate_support.CELLS).read_text(encoding="utf-8"))["cell"]
    return cells[0]


class OwedRecordsMakeUnitsNotSupported(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prov = gate_support.Provenance()
        cls.cell = first_cell()
        cls.cited = (cls.cell["evidence_test"], cls.cell["evidence_assertion"])

    def test_an_owed_record_makes_the_cell_not_supported_with_its_selector(self):
        unit = gate_support.check_cell(self.cell, self.prov, FakeLedger())
        self.assertFalse(unit.supported)
        self.assertTrue(unit.owed_only)
        self.assertEqual(unit.problems, [f"owed: fake::{self.cited[0]}::{self.cited[1]}"])
        self.assertEqual(gate_support.cause(unit.problems[0]), gate_support.OWED)

    def test_a_current_record_keeps_the_cell_supported(self):
        unit = gate_support.check_cell(self.cell, self.prov, FakeLedger([self.cited]))
        self.assertTrue(unit.supported, unit.problems)
        self.assertEqual(unit.owed, [])

    def test_an_owed_record_does_not_hide_another_missing_check(self):
        broken = {**self.cell, "evidence_assertion": "no_such_test_exists"}
        unit = gate_support.check_cell(broken, self.prov, FakeLedger())
        self.assertFalse(unit.supported)
        self.assertFalse(unit.owed_only)

    def test_a_graphless_row_follows_its_record(self):
        rows = tomllib.loads((gate_support.ROOT / gate_support.GRAPHLESS).read_text(encoding="utf-8"))["support"]
        manifest = gate_support._load(gate_support.MANIFEST)
        row = rows[0]
        path, name = (row.get("independent_check") or row["known_truth_test"]).split("::", 1)
        owed = gate_support.check_graphless(row, self.prov, manifest, FakeLedger())
        current = gate_support.check_graphless(row, self.prov, manifest, FakeLedger([(path, name)]))
        self.assertTrue(owed.owed_only)
        self.assertTrue(current.supported, current.problems)

    def test_a_capability_needs_every_independent_fixture_current(self):
        records = tomllib.loads((gate_support.ROOT / gate_support.RECORDS).read_text(encoding="utf-8"))["record"]
        rec = next(
            r for r in records
            if r.get("status") == "supported" and not r.get("matrix_units") and not r.get("graphless_units")
            and r.get("support_unit") is not False
            and len(gate_support.independent_fixtures(r)[0]) >= 1
        )
        good = gate_support.independent_fixtures(rec)[0]
        pairs = [(f["evidence_test"], f["evidence_assertion"]) for f in good]
        self.assertTrue(gate_support.check_capability(rec, self.prov, FakeLedger(pairs)).supported)
        if len(set(pairs)) > 1:
            partial = gate_support.check_capability(rec, self.prov, FakeLedger(pairs[:1]))
            self.assertTrue(partial.owed_only)

    def test_summary_splits_owed_from_other_reasons(self):
        owed = gate_support.check_cell(self.cell, self.prov, FakeLedger())
        broken = gate_support.check_cell({**self.cell, "evidence_assertion": "nope"}, self.prov, FakeLedger())
        ok = gate_support.check_cell(self.cell, self.prov, FakeLedger([self.cited]))
        summary = gate_support.summarize(
            {"matrix": [owed, broken, ok], "graphless": [], "capabilities": []}
        )
        self.assertEqual((summary["total"], summary["not_supported"]), (1, 2))
        self.assertEqual((summary["not_supported_owed"], summary["not_supported_other"]), (1, 1))


class EmbeddedStatusFollowsTheLedger(unittest.TestCase):
    def test_derived_status_is_pure_function_of_the_unit(self):
        prov = gate_support.Provenance()
        cell = first_cell()
        cited = (cell["evidence_test"], cell["evidence_assertion"])
        owed = gate_support.derived_status(gate_support.check_cell(cell, prov, FakeLedger()))
        current = gate_support.derived_status(gate_support.check_cell(cell, prov, FakeLedger([cited])))
        self.assertEqual(owed["status"], "not_supported")
        self.assertTrue(owed["missing_evidence"][0].startswith("owed: "))
        self.assertEqual(current, {"status": "supported", "missing_evidence": []})
        self.assertEqual(set(owed), set(gate_support.DERIVED_STATUS_FIELDS))

    def test_snapshot_emits_the_derived_status_and_keeps_evidence_separate(self):
        cell = dict(query="AverageEffect", graph_class="Dag", structure="explicit",
                    inference="Frequentist", validation="none", calibration=[], estimators=["explicit"])
        ident = gate_support.cell_ident(cell)
        owed = {"status": "not_supported", "missing_evidence": ["owed: x::y"]}
        with (
            patch.object(generator, "load", return_value={"record": []}),
            patch.object(generator.external_evidence, "routes", return_value={}),
        ):
            with_status = generator.support_snapshot([], [], [cell], {ident: owed})["supported"][0]
            no_ledger = generator.support_snapshot([], [], [cell])["supported"][0]
        self.assertEqual(with_status["status"], "not_supported")
        self.assertEqual(with_status["missing_evidence"], ["owed: x::y"])
        stripped = {k: v for k, v in with_status.items() if k not in gate_support.DERIVED_STATUS_FIELDS}
        self.assertEqual(stripped, {k: v for k, v in no_ledger.items() if k not in gate_support.DERIVED_STATUS_FIELDS})


class EvidenceOnlyMode(unittest.TestCase):
    def test_evidence_only_ignores_ledger_currency(self):
        class Boom(gate_support.Ledger):
            def __init__(self):
                pass

            def owed(self, test, name):
                raise AssertionError("the ledger must not be read")

        with patch.object(gate_support, "Ledger", Boom):
            result = gate_support.evaluate(evidence_only=True)
        summary = gate_support.summarize(result)
        self.assertEqual(summary["not_supported_owed"], 0)
        self.assertEqual(result["records_current_owed"], (0, 0))



class CitedChecksAreRows(unittest.TestCase):
    def test_every_rust_check_the_rule_cites_is_a_scientific_row(self):
        """A cited check with no scientific row could never earn its record."""
        import scientific_records

        rows = {(r["evidence_test"], r["evidence_assertion"])
                for r in scientific_records.all_scientific_rows(gate_support.ROOT)}
        cited = {(r["evidence_test"], r["evidence_assertion"]) for r in gate_support.cited_checks()
                 if r["evidence_test"].endswith(".rs")}
        self.assertTrue(cited)
        self.assertEqual(cited - rows, set())


if __name__ == "__main__":
    unittest.main()
