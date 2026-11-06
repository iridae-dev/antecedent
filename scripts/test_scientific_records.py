#!/usr/bin/env python3
"""Scientific record fingerprints: what a change invalidates and what it leaves current."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import scientific_records as sr  # noqa: E402

LOCK = """version = 4

[[package]]
name = "antecedent-leaf"
version = "1.0.0"

[[package]]
name = "antecedent-mid"
version = "1.0.0"
dependencies = ["antecedent-leaf"]

[[package]]
name = "serde"
version = "1.0.0"
"""


def package(root: Path, name: str, deps=(), dev=()):
    return {
        "name": name,
        "manifest_path": str(root / "crates" / name / "Cargo.toml"),
        "dependencies": [{"name": d, "kind": None} for d in deps]
        + [{"name": d, "kind": "dev"} for d in dev]
        + [{"name": "serde", "kind": None}],
    }


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.metadata = {
            "packages": [
                package(self.root, "leaf"),
                package(self.root, "mid", deps=["leaf"]),
                package(self.root, "top", deps=["mid"]),
                package(self.root, "other"),
            ]
        }
        for name in ("leaf", "mid", "top", "other"):
            self.write(f"crates/{name}/Cargo.toml", f'[package]\nname = "{name}"\nversion = "1.0.0"\n')
            self.write(f"crates/{name}/src/lib.rs", f"pub fn {name}() -> u8 {{ 1 }}\n")
            self.write(f"crates/{name}/tests/t.rs", f"#[test]\nfn check_{name}() {{}}\n")
        self.write("Cargo.toml", '[workspace]\nmembers = ["crates/*"]\n[workspace.package]\nversion = "1.0.0"\n')
        self.write("Cargo.lock", LOCK)
        self.write("rust-toolchain.toml", '[toolchain]\nchannel = "1.90"\n')
        self.write("conformance/a/expected.json", '{"x": 1}')
        self.write("conformance/b/expected.json", '{"x": 2}')

    def write(self, rel: str, text: str):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def row(self, crate: str, fixtures=()):
        return {
            "evidence_test": f"crates/{crate}/tests/t.rs",
            "evidence_assertion": f"check_{crate}",
            "roles": ["software", "scientific"],
            "fixtures": list(fixtures),
        }

    def fps(self, features: str = ""):
        fp = sr.Fingerprinter(self.root, self.metadata, features)
        return {
            c: fp.fingerprint(self.row(c, ["conformance/a"] if c == "mid" else ["conformance/b"] if c == "top" else []))
            for c in ("leaf", "mid", "top", "other")
        }


class FingerprintTests(Fixture):
    def changed(self, before, after):
        return {k for k in before if before[k] != after[k]}

    def test_unchanged_inputs_are_stable(self):
        self.assertEqual(self.fps(), self.fps())

    def test_comment_and_blank_edits_do_not_invalidate(self):
        before = self.fps()
        self.write("crates/leaf/src/lib.rs", "// note\n\npub fn leaf() -> u8 { 1 } // trailing\n")
        self.assertEqual(before, self.fps())

    def test_literal_edit_invalidates(self):
        self.write("crates/leaf/src/lib.rs", 'pub fn leaf() -> &\'static str { "a" }\n')
        before = self.fps()
        self.write("crates/leaf/src/lib.rs", 'pub fn leaf() -> &\'static str { "b" }\n')
        self.assertTrue(self.changed(before, self.fps()))

    def test_leaf_source_change_owes_only_dependents(self):
        before = self.fps()
        self.write("crates/leaf/src/lib.rs", "pub fn leaf() -> u8 { 2 }\n")
        self.assertEqual(self.changed(before, self.fps()), {"leaf", "mid", "top"})

    def test_middle_change_does_not_owe_dependency(self):
        before = self.fps()
        self.write("crates/mid/src/lib.rs", "pub fn mid() -> u8 { 2 }\n")
        self.assertEqual(self.changed(before, self.fps()), {"mid", "top"})

    def test_fixture_change_owes_only_citers(self):
        before = self.fps()
        self.write("conformance/a/expected.json", '{"x": 99}')
        self.assertEqual(self.changed(before, self.fps()), {"mid"})

    def test_test_file_change_owes_only_that_test(self):
        before = self.fps()
        self.write("crates/top/tests/t.rs", "#[test]\nfn check_top() { assert!(true); }\n")
        self.assertEqual(self.changed(before, self.fps()), {"top"})

    def test_lock_version_only_bump_stays_current(self):
        before = self.fps()
        self.write("Cargo.lock", LOCK.replace('version = "1.0.0"\n\n[[package]]\nname = "antecedent-mid"',
                                              'version = "1.1.0"\n\n[[package]]\nname = "antecedent-mid"'))
        self.write("Cargo.toml", '[workspace]\nmembers = ["crates/*"]\n[workspace.package]\nversion = "1.1.0"\n')
        self.assertEqual(before, self.fps())

    def test_external_dependency_change_invalidates_all(self):
        before = self.fps()
        self.write("Cargo.lock", LOCK.replace('name = "serde"\nversion = "1.0.0"', 'name = "serde"\nversion = "1.0.1"'))
        self.assertEqual(self.changed(before, self.fps()), set(before))

    def test_lock_tracks_which_external_version_a_dependency_edge_uses(self):
        lock = '''version = 4
[[package]]
name = "consumer"
version = "1.0.0"
dependencies = ["dep 1.0.0"]
[[package]]
name = "dep"
version = "1.0.0"
[[package]]
name = "dep"
version = "2.0.0"
'''
        self.write("Cargo.lock", lock)
        before = self.fps()
        self.write("Cargo.lock", lock.replace('"dep 1.0.0"', '"dep 2.0.0"'))
        self.assertEqual(self.changed(before, self.fps()), set(before))

    def test_feature_change_invalidates_all(self):
        self.assertEqual(self.changed(self.fps(), self.fps("--features x")), {"leaf", "mid", "top", "other"})

    def test_toolchain_change_invalidates_all(self):
        before = self.fps()
        self.write("rust-toolchain.toml", '[toolchain]\nchannel = "1.91"\n')
        self.assertEqual(self.changed(before, self.fps()), set(before))

    def test_dev_dependency_counts_for_the_test_crate_only(self):
        self.metadata["packages"][3] = package(self.root, "other", dev=["leaf"])
        self.metadata["packages"][2] = package(self.root, "top", deps=["other"])
        before = self.fps()
        self.write("crates/leaf/src/lib.rs", "pub fn leaf() -> u8 { 3 }\n")
        after = self.fps()
        self.assertIn("other", self.changed(before, after))
        self.assertNotIn("top", self.changed(before, after))


class RecordTests(Fixture):
    def test_withdraw_only_selected_old_passes_before_a_rerun(self):
        rows = [self.row("leaf"), self.row("top")]
        owed, _ = sr.classify(self.root, rows)
        sr.record_passes(self.root, [{**o, "fingerprint": "f", "dependency_manifest": "d" * 64} for o in owed])
        sr.withdraw_records(self.root, owed[:1])
        self.assertNotIn(owed[0]["selector"], sr.load_records(self.root))
        self.assertIn(owed[1]["selector"], sr.load_records(self.root))

    def test_a_pass_is_current_only_through_a_matching_manifest(self):
        rows = [self.row("leaf"), self.row("top")]
        owed, current = sr.classify(self.root, rows)
        self.assertEqual((len(owed), len(current)), (2, 0))
        self.assertTrue(all(o["manifest_reasons"] == ["no_manifest"] for o in owed))
        sr.record_passes(self.root, [{**owed[0], "fingerprint": "f", "dependency_manifest": "a" * 64}])
        owed, current = sr.classify(self.root, rows)
        self.assertEqual((len(owed), len(current)), (2, 0))
        self.assertTrue(any(r.startswith("manifest_owed") for r in owed[0]["manifest_reasons"]))
        text = (self.root / sr.RECORDS).read_text(encoding="utf-8")
        self.assertIn("dependency_manifest", text)
        self.assertNotIn("executed", text)

    def test_a_new_pass_is_saved_beside_records_without_a_manifest(self):
        rows = [self.row("leaf"), self.row("top")]
        sr.write_records(self.root, {"leaf::t::check_leaf": {
            "roles": ["scientific"], "commit": "c", "fingerprint": "f", "recorded_at": "t"}})
        owed, _ = sr.classify(self.root, rows)
        sr.record_passes(self.root, [{**owed[1], "fingerprint": "f", "dependency_manifest": "d" * 64}])
        records = sr.load_records(self.root)
        self.assertNotIn("dependency_manifest", records["leaf::t::check_leaf"])
        self.assertEqual(records["top::t::check_top"]["dependency_manifest"], "d" * 64)
        owed, _ = sr.classify(self.root, rows)
        self.assertIn("leaf::t::check_leaf", [o["selector"] for o in owed])

    def test_native_closure_covers_python_rows(self):
        self.metadata["packages"].append({
            "name": "native", "manifest_path": str(self.root / "python/Cargo.toml"),
            "dependencies": [{"name": "leaf", "kind": None}],
        })
        self.write("python/Cargo.toml", '[package]\nname = "native"\nversion = "1.0.0"\n')
        self.write("python/src/lib.rs", "pub fn n() {}\n")
        self.write("python/antecedent/__init__.py", "x = 1\n")
        self.write("python/tests/test_a.py", "def test_a(): pass\n")
        row = {"evidence_test": "python/tests/test_a.py", "evidence_assertion": "test_a", "roles": ["scientific"], "fixtures": []}
        fp1 = sr.Fingerprinter(self.root, self.metadata, "").fingerprint(row)
        self.write("crates/leaf/src/lib.rs", "pub fn leaf() -> u8 { 9 }\n")
        fp2 = sr.Fingerprinter(self.root, self.metadata, "").fingerprint(row)
        self.write("python/antecedent/__init__.py", "x = 2\n")
        fp3 = sr.Fingerprinter(self.root, self.metadata, "").fingerprint(row)
        self.write("crates/other/src/lib.rs", "pub fn other() -> u8 { 9 }\n")
        fp4 = sr.Fingerprinter(self.root, self.metadata, "").fingerprint(row)
        self.assertEqual(len({fp1, fp2, fp3}), 3)
        self.assertEqual(fp3, fp4)
        self.write("python/pyproject.toml", '[tool.pytest.ini_options]\naddopts = "-x"\n')
        self.assertNotEqual(fp4, sr.Fingerprinter(self.root, self.metadata, "").fingerprint(row))
        self.assertEqual(sr.locate(self.root, row)["selector"], "tests/test_a.py::test_a")


class ExecutionCaptureTests(Fixture):
    def fake_export(self, *rels, covered=1):
        return {
            "data": [
                {
                    "files": [
                        {"filename": str(self.root / r), "summary": {"lines": {"covered": covered}}}
                        for r in rels
                    ]
                }
            ]
        }

    def test_line_coverage_keeps_lines_where_a_covered_region_starts(self):
        export = {"data": [{"files": [{"filename": str(self.root / "crates/top/src/lib.rs"),
                                       "segments": [[1, 1, 3, True, True, False], [2, 1, 0, True, True, False],
                                                    [4, 1, 5, False, False, False]]}]}]}
        coverage = sr.line_coverage(export)
        self.assertEqual(coverage[str((self.root / "crates/top/src/lib.rs").resolve())], {1})

    def test_the_target_is_built_once_and_each_case_runs_its_binary_exactly(self):
        info = sr.locate(self.root, self.row("top"))
        build = sr.build_command(self.root, info["crate"], tuple(info["target"]))
        self.assertEqual(build[:3], ["cargo", "test", "--no-run"])
        self.assertIn("--test", build)
        self.assertIn("-C instrument-coverage", sr.capture_env(self.root)["RUSTFLAGS"])
        seen = []
        self.run_capture(["crates/top/src/lib.rs"], seen=seen)
        self.assertEqual(seen[0][:3], ["/bin/top-test", "--exact", "check_top"])

    def test_the_built_executable_is_read_from_cargo_json(self):
        import json

        lines = [
            json.dumps({"reason": "compiler-artifact", "profile": {"test": False}, "executable": None}),
            json.dumps({"reason": "compiler-artifact", "profile": {"test": True}, "executable": "/t/top-abc"}),
            json.dumps({"reason": "build-finished", "success": True}),
        ]
        self.assertEqual(sr.built_executable("\n".join(lines)), "/t/top-abc")

    def run_capture(self, rels, passed=True, seen=None):
        import json
        import subprocess
        from unittest import mock

        def fake(cmd, **kwargs):
            if seen is not None:
                seen.append(cmd)
            if cmd[0] == "/bin/top-test":
                text = "test result: ok. 1 passed; 0 failed" if passed else "test result: FAILED"
                return subprocess.CompletedProcess(cmd, 0 if passed else 101, text, "")
            if cmd[1] == "export":
                return subprocess.CompletedProcess(cmd, 0, json.dumps(self.fake_export(*rels)), "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch.object(sr, "llvm_tool", lambda name: name):
            return sr.capture_one(
                self.root, "/bin/top-test", sr.locate(self.root, self.row("top")), {}, fake
            )

    def test_a_failing_test_is_not_captured(self):
        self.assertIsNone(self.run_capture(["crates/top/src/lib.rs"], passed=False))


class OneScientificEntryPoint(unittest.TestCase):
    """Scientific assertions run in one place; the lanes and gates do not duplicate it."""

    root = Path(__file__).resolve().parents[1]

    def read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def test_the_runner_batches_rust_and_python_and_uses_the_optimized_profile(self):
        runner = self.read("scripts/run_evidence_rows.py")
        self.assertEqual(runner.count('"nextest"'), 1, "all Rust rows run in one nextest invocation")
        self.assertIn("--ignore-default-filter", runner)
        records = self.read("scripts/scientific_records.py")
        self.assertIn("CARGO_PROFILE_TEST_OPT_LEVEL", records)
        self.assertIn("CARGO_PROFILE_TEST_DEBUG", records)
        self.assertNotIn("GateSession", records)

    def test_workspace_features_match_the_engineering_lane(self):
        from _repo import WORKSPACE_TEST_FEATURES

        runner = self.read("scripts/test.sh")
        line = next(ln for ln in runner.splitlines() if ln.startswith("workspace=("))
        declared = line.split("--features ", 1)[1].rstrip(")").split(",")
        self.assertEqual(sorted(declared), sorted(WORKSPACE_TEST_FEATURES))

    def test_evidence_audit_has_no_second_runner(self):
        self.assertNotIn("run-scientific", self.read("scripts/evidence_audit.py"))
        self.assertIn("scientific_records.py\" run", self.read("scripts/test.sh"))

    def test_no_registry_row_selects_tests_by_substring(self):
        for path in sorted((self.root / "parity").rglob("*.toml")):
            self.assertNotIn("composition_filter", path.read_text(encoding="utf-8"), path.as_posix())

    def test_ledger_wrappers_only_resolve_their_rows(self):
        for name in ("capability_records", "checked_execution", "composition", "transport", "gcm_conformance"):
            text = self.read(f"scripts/gate_{name}.sh")
            for line in text.splitlines():
                if "run_evidence_rows.py" in line and not line.lstrip().startswith("#"):
                    if name == "composition" and '"${@:3}"' in line:
                        continue  # run_rows: the self-test runs the executing runner on a throwaway workspace
                    self.assertIn("--static", line, f"gate_{name}.sh executes rows itself: {line}")
        self.assertNotIn("cargo test", self.read("scripts/gate_gcm_conformance.sh").replace("cargo bench", ""))


class OnlyAndCurrencyGates(unittest.TestCase):
    """`--only` selection, per-batch recording, and CI executing neither lane."""

    root = Path(__file__).resolve().parents[1]

    def read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def test_only_matches_substring_or_glob_on_selector_path_and_id(self):
        item = {"selector": "p::t::mod::name", "evidence_test": "crates/p/tests/t.rs", "id": "cap.one"}
        self.assertTrue(sr.matches_only(item, []))
        self.assertTrue(sr.matches_only(item, ["mod::name"]))
        self.assertTrue(sr.matches_only(item, ["p::t::*::name"]))
        self.assertTrue(sr.matches_only(item, ["crates/p/tests"]))
        self.assertTrue(sr.matches_only(item, ["cap.one"]))
        self.assertFalse(sr.matches_only(item, ["other"]))

    def test_run_records_after_every_batch(self):
        body = self.read("scripts/scientific_records.py")
        loop = body[body.index("for (crate, target), todo in sorted(groups.items())"):]
        self.assertLess(loop.index("record_passes"), loop.index("return failed"))

    def test_ci_checks_records_and_executes_neither_lane(self):
        gate = self.read("scripts/gate_release.sh")
        self.assertIn("step python3 scripts/scientific_records.py status --check", gate)
        self.assertIn("step python3 scripts/calibration_status.py --check", gate)
        self.assertNotIn("test.sh scientific", gate)
        self.assertNotIn("test.sh scientific", self.read("scripts/gate_release_candidate.sh"))
        workflow = self.read(".github/workflows/ci.yml")
        job = workflow[workflow.index("\n  scientific:"):workflow.index("\n  python-lint:")]
        self.assertNotIn("nextest", job)
        self.assertIn("--section scientific", job)


if __name__ == "__main__":
    unittest.main()
