#!/usr/bin/env python3
"""A source-only certificate cannot stand in for an unperformed numerical replay."""

import dataclasses
import unittest
from unittest.mock import patch

import _temporal_capture_transfer as transfer
import calibration_facets as facets


class SourceOnlyCertificate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surface = facets.load_surface()
        cls.refs = facets.references(cls.surface)
        cls.repo = facets.Repo()
        cls.records = facets.load_records()
        cls.waiver = facets.Waiver(
            id="bounded-source-only-test",
            from_=transfer.ORIGINAL_SOURCE,
            to=transfer.CORRECTION_SOURCE,
            reviewed_by="unit-test review",
            justification="Exact metadata-only source correction; zero numerical replay.",
            paths=[transfer.PATH],
            replay=[],
            reviewed_equivalence=[
                facets.ReviewedEquivalence(
                    transfer.PATH,
                    cls.repo.blob(transfer.ORIGINAL_SOURCE, transfer.PATH),
                    cls.repo.blob(transfer.CORRECTION_SOURCE, transfer.PATH),
                    "rust_temporal_construction_metadata",
                    "Exact temporal descriptor and ordinary regression only.",
                    "Pinned complete blobs and full compiled-input manifest.",
                )
            ],
            source_only=True,
        )

    def check(self, waiver, require_outcome=False):
        return facets.validate_waivers(
            self.surface, [waiver], self.records, self.repo, self.refs,
            require_outcome=require_outcome,
        )

    def test_explicit_mode_round_trips_and_proves_actual_immutable_source_pair(self):
        parsed, errors = facets.parse_waivers(facets.render_waivers([self.waiver]))
        self.assertFalse(errors)
        self.assertEqual(parsed, [self.waiver])
        checked = self.check(parsed[0])
        self.assertFalse(checked.problems)
        self.assertEqual(checked.valid, parsed)
        self.assertTrue(self.check(parsed[0], require_outcome=True).problems)

    def test_default_mode_still_requires_a_real_replay_request(self):
        text = facets.render_waivers([self.waiver]).replace("source_only = true\n", "")
        parsed, errors = facets.parse_waivers(text)
        self.assertFalse(parsed)
        self.assertTrue(any("needs at least one" in e for e in errors))

    def test_mode_is_boolean_and_cannot_claim_numerical_replay(self):
        text = facets.render_waivers([self.waiver])
        parsed, errors = facets.parse_waivers(text.replace("source_only = true", 'source_only = "true"'))
        self.assertFalse(parsed)
        self.assertTrue(errors)
        mixed = dataclasses.replace(
            self.waiver, replay=[facets.ReplayRecord("arbitrary-record", [transfer.PATH])]
        )
        parsed, errors = facets.parse_waivers(facets.render_waivers([mixed]))
        self.assertFalse(parsed)
        self.assertTrue(any("cannot claim numeric replay" in e for e in errors))

    def test_no_other_source_scope_certificate_or_gate_inheritance_qualifies(self):
        certificate = self.waiver.reviewed_equivalence[0]
        mutations = [
            {"from_": "c30e10304f4bd85a01ad81dd133a360ddc95c2f1"},
            {"paths": [transfer.PATH, "crates/antecedent-estimate/src/response/plugin.rs"]},
            {"reviewed_equivalence": []},
            {"reviewed_equivalence": [dataclasses.replace(certificate, kind="rust_comments")]},
            {"recordless_gates": True},
            {"gate_ledger_commit": transfer.CORRECTION_SOURCE},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                checked = self.check(dataclasses.replace(self.waiver, **mutation))
                self.assertTrue(any("source_only requires" in e for e in checked.problems))
                self.assertFalse(checked.valid)

    def test_whole_compiled_input_proof_is_mandatory(self):
        with patch.object(transfer, "verify_input_transfer", side_effect=ValueError("extra Cargo input")):
            checked = self.check(self.waiver)
        self.assertFalse(checked.valid)
        self.assertTrue(any("compiled-input proof failed" in e for e in checked.problems))


if __name__ == "__main__":
    unittest.main()
