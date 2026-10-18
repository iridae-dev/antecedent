"""Closure rules of the acceptance manifest: binding is complete and only a live test closes a case."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import check_acceptance_manifest as manifest  # noqa: E402

GOOD_RUST = '#[test]\nfn refuses() {\n    assert!(run().is_err());\n}\n'
# Spelled in pieces so this file does not itself read as a deferred test.
DEFER = 'to' + 'do!();'
SKIP = 'sk' + 'ip'
GOOD_PY = 'def test_refuses():\n    assert run() is None\n'


def case(cid: str = "c", kind: str = "value", evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": cid, "area": "ingest", "workflow": "w", "configuration": {"rows": 1},
            "expected": {"kind": kind, "outputs": ["o"]},
            "targets": {"runtime_seconds": 1, "memory_bytes": 1, "rationale": "r"},
            "dataset": {"generator": "g", "seed": 1, "simulated_representative": True,
                        "provenance": "Simulated representative of the described study."},
            "evidence": evidence or {"status": "planned"}}


class RequireBoundTest(unittest.TestCase):
    def test_same_file_forwarding_macro_must_really_generate_the_named_test(self) -> None:
        source = '''
macro_rules! outer { ($name:ident, $x:expr) => { inner!($name, $x); }; }
macro_rules! inner { ($name:ident, $x:expr) => {
    #[test]
    #[ignore = "calibration"]
    fn $name() { assert!(run($x)); }
}; }
outer!(actual, 4);
// outer!(comment_only, 7);
const TEXT: &str = "outer!(string_only, 10);";
'''
        result = manifest.test_source(source, ".rs", "actual")
        self.assertIsNotNone(result)
        self.assertIn("#[ignore", result[0])
        self.assertIn("fn actual()", result[1])
        for absent in ("comment_only", "string_only", "nonexistent"):
            self.assertIsNone(manifest.test_source(source, ".rs", absent))
        self.assertIsNone(manifest.test_source(source.replace("#[test]", ""), ".rs", "actual"))
        renamed = source.replace("inner!($name, $x)", "inner!(different, $x)")
        self.assertIsNone(manifest.test_source(renamed, ".rs", "actual"))

    @staticmethod
    def manifest_of(*cases: dict[str, Any]) -> dict[str, Any]:
        return {"manifest": {"schema_version": 1}, "case": list(cases)}

    def test_require_bound_rejects_a_planned_case(self) -> None:
        bound = case("a", evidence={"status": "bound", "test": "x::y"})
        planned = case("b")
        found = manifest.problems(self.manifest_of(bound, planned), require_bound=True)
        self.assertEqual([p for p in found if "planned case" in p], ["'b': planned case at release closure"])
        self.assertFalse(any("planned case" in p for p in manifest.problems(self.manifest_of(bound), require_bound=True)))
        self.assertFalse(any("planned case" in p for p in manifest.problems(self.manifest_of(bound, planned))))


class ClosureTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "crates/c/tests").mkdir(parents=True)
        (self.root / "scripts").mkdir()

    def put(self, rel: str, text: str) -> str:
        (self.root / rel).write_text(text, encoding="utf-8", newline="\n")
        return rel

    def problems(self, rel: str, name: str, kind: str = "value", **evidence: Any) -> list[str]:
        evidence = {"status": "bound", "test": f"{rel}::{name}", **evidence}
        return manifest.evidence_problems("c", evidence, self.root, kind)

    def test_a_case_cannot_close_on_an_audit_limit_failed_measurement_or_deferral(self) -> None:
        rust = self.put("crates/c/tests/t.rs", GOOD_RUST + "\nfn not_a_test() {\n    assert!(true);\n}\n")
        python = self.put("scripts/test_t.py", GOOD_PY)
        self.assertEqual(self.problems(rust, "refuses"), [])
        self.assertEqual(self.problems(python, "test_refuses"), [])

        for kind in ("audit", "limit", "failed_measurement", "documentation_only", "deferral"):
            found = self.problems(rust, "refuses", **{kind: "recorded"})
            self.assertTrue(any(f"evidence.{kind} cannot close a case" in p for p in found), kind)
        self.assertTrue(any("not a test function" in p for p in self.problems(rust, "not_a_test")))

        silent = self.put("crates/c/tests/s.rs", '#[test]\nfn documents() {\n    println!("limit is 4");\n}\n')
        self.assertTrue(any("asserts nothing" in p for p in self.problems(silent, "documents")))
        helped = self.put("crates/c/tests/h.rs", '#[test]\nfn delegates() {\n    check_all();\n}\n\nfn check_all() {\n    assert_eq!(1, 1);\n}\n')
        self.assertEqual(self.problems(helped, "delegates"), [])
        deferred = self.put("crates/c/tests/d.rs", '#[test]\nfn later() {\n    ' + DEFER + '\n    assert!(true);\n}\n')
        self.assertTrue(any("defers" in p for p in self.problems(deferred, "later")))
        skipped = self.put("scripts/test_s.py", '@pytest.mark.' + SKIP + '\ndef test_later():\n    assert 1\n')
        self.assertTrue(any("defers" in p for p in self.problems(skipped, "test_later")))
        script = self.put("scripts/doc.py", 'def documents_a_limit():\n    print("limit")\n')
        self.assertTrue(any("not a test function" in p for p in self.problems(script, "documents_a_limit")))

    def test_an_ignored_test_closes_only_a_calibrated_case_or_a_wired_scale_case(self) -> None:
        ignored = self.put("crates/c/tests/i.rs",
                           '#[test]\n#[ignore = "release scale"]\nfn big() {\n    assert!(true);\n}\n'
                           '#[test]\n#[ignore = "calibration"]\nfn design() {\n    run_design();\n}\n')
        self.assertTrue(any("#[ignore]d" in p for p in self.problems(ignored, "big")))
        self.assertEqual(self.problems(ignored, "design", kind="calibrated_interval"), [])
        self.put(manifest.SCALE_GATE, "#!/usr/bin/env bash\n")
        self.put("scripts/gate_release_candidate.sh", f"bash {manifest.SCALE_GATE}\n")
        self.assertEqual(self.problems(ignored, "big"), [])
        src = self.put("crates/c/lib_i.rs", '#[test]\n#[ignore]\nfn big_src() {\n    assert!(true);\n}\n')
        self.assertTrue(any("#[ignore]d" in p for p in self.problems(src, "big_src")))

    def test_scale_cases_lists_the_ignored_bound_tests_with_their_budgets(self) -> None:
        ignored = self.put("crates/c/tests/i.rs", '#[test]\n#[ignore]\nfn big() {\n    assert!(true);\n}\n')
        (self.root / "crates/c/Cargo.toml").write_text('[package]\nname = "antecedent-c"\n', encoding="utf-8")
        data = {"case": [case("big_case", evidence={"status": "bound", "test": f"{ignored}::big"}),
                         case("plain", evidence={"status": "planned"})]}
        self.assertEqual([c["package"] for c in manifest.scale_cases(data, self.root)], ["antecedent-c"])
        self.assertEqual(manifest.scale_cases(data, self.root)[0]["runtime_seconds"], 1)


if __name__ == "__main__":
    unittest.main()
