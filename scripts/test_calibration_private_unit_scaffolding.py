#!/usr/bin/env python3
"""Bounded private-unit isolation, independently checked against immutable sources."""

import hashlib
import json
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import calibration_facets as facets

ROOT = Path(__file__).resolve().parents[1]
UNIT = "crates/antecedent-estimate/src/response/tests.rs"
PARENT = "crates/antecedent-estimate/src/response/mod.rs"


def source(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True)


class PrivateUnitScaffolding(unittest.TestCase):
    def test_reviewed_streams_and_test_only_parents_match_immutable_blobs(self):
        proof = json.loads(
            (ROOT / "parity/calibration_unit_scaffolding_proofs.json").read_text(encoding="utf-8")
        )
        self.assertEqual(proof["path"], UNIT)
        self.assertEqual(proof["parent"], PARENT)
        pattern = facets._rust_equivalence.tokens("#[cfg(test)] mod tests;")
        seen = set()
        for row in proof["sources"]:
            body = source(row["source"], UNIT)
            parent = source(row["source"], PARENT)
            code = facets._rust_equivalence.tokens(body)
            parent_code = facets._rust_equivalence.tokens(parent)
            self.assertIn(
                pattern,
                [parent_code[i : i + len(pattern)] for i in range(len(parent_code))],
            )
            digest = hashlib.sha256("\0".join(code).encode()).hexdigest()
            self.assertEqual(digest, row["literal_preserving_token_sha256"])
            seen.add(digest)
            for field, path in (("unit_file_blob", UNIT), ("parent_blob", PARENT)):
                actual = subprocess.check_output(
                    ["git", "rev-parse", f"{row['source']}:{path}"], cwd=ROOT, text=True
                ).strip()
                self.assertEqual(actual, row[field])
            for forbidden in (
                "pub",
                "macro_export",
                "include",
                "include_str",
                "include_bytes",
                "ctor",
                "link_section",
                "export_name",
                "no_mangle",
            ):
                self.assertNotIn(forbidden, code)
            self.assertEqual(facets.scaffolding_view(UNIT, body), "")
        self.assertEqual(seen, facets._BOUNDED_PRIVATE_UNIT_TOKENS[UNIT])
        self.assertIn(UNIT, facets.load_surface().testmods)

    def test_new_exports_macros_includes_initializers_and_producers_stay_visible(self):
        body = source("70f19b82", UNIT)
        for addition in (
            "pub fn escape() {}",
            "pub(crate) fn escape() {}",
            "#[macro_export] macro_rules! escape { () => { 1 }; }",
            'include!("outside.rs");',
            'const TEXT: &str = include_str!("outside.rs");',
            "#[ctor::ctor] fn startup() {}",
            "#[test] #[ignore] fn calibration_producer() {}",
        ):
            changed = body + "\n" + addition
            self.assertEqual(facets.scaffolding_view(UNIT, changed), changed)
        changed_literal = body.replace("0.0", "0.000001", 1)
        self.assertNotEqual(changed_literal, body)
        self.assertEqual(facets.scaffolding_view(UNIT, changed_literal), changed_literal)

    def test_missing_test_only_parent_does_not_ignore_even_a_reviewed_stream(self):
        body = source("70f19b82", UNIT)
        with patch.object(facets, "declares_out_of_line_test_module", return_value=False):
            self.assertEqual(facets.scaffolding_view(UNIT, body), body)
        self.assertNotEqual(facets.scaffolding_view(UNIT + ".other", body), "")

    def test_actual_measurement_producer_cannot_be_declared_scaffolding(self):
        surface = facets.load_surface()
        records = [{"test": UNIT + "::identity_shift_has_the_exact_empirical_mean_influence"}]
        with (
            patch.object(facets, "load_records", return_value=records),
            patch.object(
                facets, "references", return_value=SimpleNamespace(reexports=[], by_facet={})
            ),
        ):
            errors = facets.check(surface)
        self.assertTrue(any(UNIT in error and "measurement" in error for error in errors), errors)
        for row in facets.load_records():
            for field in ("test", "dgp"):
                self.assertNotEqual(str(row.get(field, "")).rsplit("::", 1)[0], UNIT)


if __name__ == "__main__":
    unittest.main()
