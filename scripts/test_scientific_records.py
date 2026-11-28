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

    def test_scientific_target_feature_mapping_is_fingerprinted(self):
        self.write("crates/top/Cargo.toml", '[package]\nname = "top"\nversion = "1.0.0"\n'
                   '[features]\na = []\nb = []\n'
                   '[package.metadata.scientific-test-features]\nt = ["a"]\n')
        before_fp = sr.Fingerprinter(self.root, self.metadata, "", scientific_build=True).fingerprint(self.row("top"))
        before = sr.Fingerprinter(self.root, self.metadata, "", scientific_build=True).parts(self.row("top"))["features"]
        calibration_before = sr.Fingerprinter(self.root, self.metadata, "").parts(self.row("top"))["features"]
        self.write("crates/top/Cargo.toml", '[package]\nname = "top"\nversion = "1.0.0"\n'
                   '[features]\na = []\nb = []\n'
                   '[package.metadata.scientific-test-features]\nt = ["a", "b"]\n')
        after_fp = sr.Fingerprinter(self.root, self.metadata, "", scientific_build=True).fingerprint(self.row("top"))
        after = sr.Fingerprinter(self.root, self.metadata, "", scientific_build=True).parts(self.row("top"))["features"]
        calibration_after = sr.Fingerprinter(self.root, self.metadata, "").parts(self.row("top"))["features"]
        self.assertNotEqual(before, after)
        self.assertNotEqual(before_fp, after_fp)
        self.assertEqual(calibration_before, calibration_after)

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

    def test_region_coverage_keeps_executed_code_regions_of_functions_that_ran(self):
        name = str(self.root / "crates/top/src/lib.rs")
        export = {"data": [{"functions": [
            {"count": 2, "filenames": [name], "regions": [[1, 1, 3, 2, 2, 0, 0, 0], [2, 5, 2, 9, 0, 0, 0, 0],
                                                          [4, 1, 4, 3, 1, 0, 0, 1]]},
            {"count": 0, "filenames": [name], "regions": [[7, 1, 8, 2, 0, 0, 0, 0]]}]}]}
        coverage = sr.region_coverage(export)
        self.assertEqual(coverage[str((self.root / "crates/top/src/lib.rs").resolve())], {(1, 1)})

    def test_the_capture_runs_the_exact_test_under_coverage(self):
        self.assertIn("-C instrument-coverage", sr.capture_env(self.root)["RUSTFLAGS"])
        seen = []
        self.run_capture(["crates/top/src/lib.rs"], seen=seen)
        self.assertEqual(seen[0][:3], ["/bin/top-test", "--exact", "check_top"])

    def test_scientific_build_uses_stable_target_feature_union(self):
        self.write("crates/top/Cargo.toml", '[package]\nname = "top"\nversion = "1.0.0"\n'
                   '[features]\na = []\nb = []\n'
                   '[package.metadata.scientific-test-features]\nt = ["b", "a"]\n')
        from _repo import cargo_target_feature_args, scientific_target_feature_args
        features = scientific_target_feature_args(self.root / "crates/top/Cargo.toml", ["--test", "t"])
        self.assertEqual(features[features.index("--features") + 1], "a,b")
        self.assertEqual(cargo_target_feature_args(self.root / "crates/top/Cargo.toml", ["--test", "t"]), [])

    def test_capture_executes_the_test_in_its_inline_module(self):
        self.write("crates/top/tests/t.rs", "mod truth { #[test] fn check_top() {} }\n")
        seen = []
        self.run_capture(["crates/top/src/lib.rs"], seen=seen)
        self.assertEqual(seen[0][:3], ["/bin/top-test", "--exact", "truth::check_top"])

    def run_capture(self, rels, passed=True, seen=None):
        import json
        import subprocess

        def fake(cmd, **kwargs):
            if seen is not None:
                seen.append(cmd)
            if cmd[0] == "/bin/top-test":
                text = "test result: ok. 1 passed; 0 failed" if passed else "test result: FAILED"
                return subprocess.CompletedProcess(cmd, 0 if passed else 101, text, "")
            if cmd[1] == "export":
                return subprocess.CompletedProcess(cmd, 0, json.dumps(self.fake_export(*rels)), "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        return sr.capture_one(
            self.root, "/bin/top-test", sr.rust_execution_info(self.root, self.row("top")), {}, fake,
            llvm_tools={"llvm-profdata": "llvm-profdata", "llvm-cov": "llvm-cov"},
        )

    def test_a_failing_test_is_not_captured(self):
        self.assertIsNone(self.run_capture(["crates/top/src/lib.rs"], passed=False))


class CarryForwardTests(Fixture):
    """Output digests and carry chains (ADR 0031, carrying a pass forward)."""

    TRACE = ("output\tsuite\tm::name\tbbbb\n"
             "unit\tsuite\tm::name\tx\n"
             "output\tsuite\tm::other\tcccc\n"
             "output\tsuite\tm::name\taaaa\n")

    def test_the_output_digest_is_of_the_tests_own_outputs_in_any_order(self):
        digest = sr.trace_digest(self.TRACE, "m::name")
        self.assertEqual(len(digest), 16)
        reordered = "output\tsuite\tm::name\taaaa\noutput\tsuite\tm::name\tbbbb\n"
        self.assertEqual(sr.trace_digest(reordered, "m::name"), digest)
        self.assertNotEqual(sr.trace_digest(self.TRACE, "m::other"), digest)
        self.assertIsNone(sr.trace_digest(self.TRACE, "m::absent"))
        self.assertIsNone(sr.trace_digest("", "m::name"))

    def test_only_facade_targets_build_and_run_with_the_trace(self):
        import json
        import subprocess

        self.write("crates/antecedent/Cargo.toml", '[package]\nname = "antecedent"\nversion = "1.0.0"\n'
                   '[features]\nevidence-trace = []\n')
        from _repo import scientific_target_feature_args
        facade = scientific_target_feature_args(self.root / "crates/antecedent/Cargo.toml", ["--test", "t"])
        self.assertEqual(facade[facade.index("--features") + 1], "evidence-trace")
        self.assertEqual(scientific_target_feature_args(self.root / "crates/top/Cargo.toml", ["--test", "t"]), [])
        seen = {}

        def fake(cmd, **kwargs):
            if cmd[0] == "/bin/suite":
                seen["trace"] = kwargs["env"].get("ANTECEDENT_EVIDENCE_TRACE")
                Path(seen["trace"]).write_text(self.TRACE, encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "test result: ok. 1 passed; 0 failed", "")
            if cmd[1] == "export":
                return subprocess.CompletedProcess(cmd, 0, json.dumps({"data": []}), "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        info = {"selector": "antecedent::suite::m::name", "crate": "antecedent", "libtest": "m::name"}
        sr.trace_file(self.root, info["selector"]).parent.mkdir(parents=True)
        sr.trace_file(self.root, info["selector"]).write_text("output\tsuite\tm::name\tstale\n", encoding="utf-8")
        self.assertIsNotNone(sr.capture_one(
            self.root, "/bin/suite", info, {}, fake,
            llvm_tools={"llvm-profdata": "llvm-profdata", "llvm-cov": "llvm-cov"},
        ))
        self.assertEqual(seen["trace"], str(sr.trace_file(self.root, info["selector"])))
        self.assertEqual(sr.output_digest(self.root, info), sr.trace_digest(self.TRACE, "m::name"))
        self.assertIsNone(sr.output_digest(self.root, {**info, "crate": "top"}))

    def earned(self, owed, fingerprint, digest):
        return [{**owed, "fingerprint": fingerprint, "dependency_manifest": "d" * 64, "output_digest": digest}]

    def test_an_equal_output_digest_extends_the_chain_and_a_different_one_resets_it(self):
        owed, _ = sr.classify(self.root, [self.row("leaf")])
        selector = owed[0]["selector"]
        sr.record_passes(self.root, self.earned(owed[0], "f1", "aaaa"))
        self.assertNotIn("carry", sr.load_records(self.root)[selector])
        for fingerprint in ("f2", "f3"):
            previous = sr.load_records(self.root)
            sr.withdraw_records(self.root, owed)
            sr.record_passes(self.root, self.earned(owed[0], fingerprint, "aaaa"), previous=previous)
        record = sr.load_records(self.root)[selector]
        self.assertEqual(record["output_digest"], "aaaa")
        self.assertEqual([(e["kind"], e["from_fingerprint"], e["to_fingerprint"]) for e in record["carry"]],
                         [("output_equivalent", "f1", "f2"), ("output_equivalent", "f2", "f3")])
        self.assertEqual(record["carry"][0]["reason"], "output digest unchanged")
        previous = sr.load_records(self.root)
        sr.withdraw_records(self.root, owed)
        sr.record_passes(self.root, self.earned(owed[0], "f4", "bbbb"), previous=previous)
        self.assertNotIn("carry", sr.load_records(self.root)[selector])
        sr.record_passes(self.root, self.earned(owed[0], "f5", None))
        self.assertNotIn("output_digest", sr.load_records(self.root)[selector])

    def test_a_judgment_is_current_only_while_the_fingerprint_matches(self):
        from types import SimpleNamespace
        from unittest import mock

        rows = [self.row("leaf"), self.row("top")]
        owed, _ = sr.classify(self.root, rows)
        sr.record_passes(self.root, self.earned(owed[0], "f1", "aaaa"))
        args = SimpleNamespace(only=["leaf", "top"], reason="docs only", author="charles")
        def fingerprinter(root):
            return sr.Fingerprinter(root, self.metadata, "", scientific_build=True)

        with mock.patch.object(sr, "all_scientific_rows", lambda root: rows), \
                mock.patch.object(sr, "current_fingerprinter", fingerprinter):
            self.assertEqual(sr.cmd_justify(self.root, args), 1, "top has no pass to carry")
            self.assertNotIn("carry", sr.load_records(self.root)[owed[0]["selector"]])
            self.assertEqual(sr.cmd_justify(self.root, SimpleNamespace(**{**vars(args), "only": ["leaf"]})), 0)
            [entry] = sr.load_records(self.root)[owed[0]["selector"]]["carry"]
            self.assertEqual((entry["kind"], entry["from_fingerprint"], entry["reason"], entry["author"]),
                             ("judgment", "f1", "docs only", "charles"))
            _, current = sr.classify(self.root, rows)
            self.assertEqual([c["selector"] for c in current], [owed[0]["selector"]])
            self.assertEqual(current[0]["manifest_reasons"][-1], "justified")
            self.write("crates/leaf/src/lib.rs", "pub fn leaf() -> u8 { 2 }\n")
            self.assertEqual(sr.classify(self.root, rows)[1], [])
            self.assertEqual(sr.cmd_justify(self.root, SimpleNamespace(**{**vars(args), "only": ["leaf"]})), 0)
            chain = sr.load_records(self.root)[owed[0]["selector"]]["carry"]
            self.assertEqual(chain[1]["from_fingerprint"], chain[0]["to_fingerprint"])
            self.assertEqual(len(sr.classify(self.root, rows)[1]), 1)

    def test_records_round_trip_and_rows_without_carry_render_as_before(self):
        records = {
            "a::t::plain": {"roles": ["scientific"], "commit": "c", "dirty": False, "fingerprint": "f",
                            "recorded_at": "t", "dependency_manifest": "pass:scientific:a::t::plain"},
            "a::t::carried": {"roles": ["scientific"], "commit": "c", "dirty": True, "fingerprint": "f2",
                              "recorded_at": "t", "output_digest": "aaaa", "carry": [
                                  sr.carry_entry("output_equivalent", "f1", "f2", "output digest unchanged", "s"),
                                  sr.carry_entry("judgment", "f2", "g", 'quote " and \\ slash', "charles")]},
        }
        sr.write_records(self.root, records)
        text = (self.root / sr.RECORDS).read_text(encoding="utf-8")
        loaded = sr.load_records(self.root)
        self.assertEqual(loaded["a::t::carried"]["carry"], records["a::t::carried"]["carry"])
        self.assertNotIn("output_digest", loaded["a::t::plain"])
        sr.write_records(self.root, loaded)
        self.assertEqual((self.root / sr.RECORDS).read_text(encoding="utf-8"), text)
        self.assertTrue(text.endswith(
            '\n[[record]]\ntest = "a::t::plain"\nroles = ["scientific"]\npassed = true\ncommit = "c"\n'
            'dirty = false\nfingerprint = "f"\nrecorded_at = "t"\n'
            'dependency_manifest = "pass:scientific:a::t::plain"\n'))

    def test_the_committed_ledger_rows_render_byte_identically(self):
        committed = Path(__file__).resolve().parents[1] / sr.RECORDS
        if not committed.is_file():
            self.skipTest("no committed ledger")
        self.write(sr.RECORDS, committed.read_text(encoding="utf-8"))
        sr.write_records(self.root, sr.load_records(self.root))
        rendered, original = (self.root / sr.RECORDS).read_text(encoding="utf-8"), committed.read_text(encoding="utf-8")
        self.assertEqual(rendered[rendered.index("\n[[record]]"):], original[original.index("\n[[record]]"):])

    def test_status_reports_chain_lengths_and_judged_records(self):
        import contextlib
        import io

        entry = sr.carry_entry("output_equivalent", "a", "b", "output digest unchanged", "s")
        judged = sr.carry_entry("judgment", "b", "c", "why", "charles")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sr.report_carry({"x": {"carry": [entry]}, "y": {"carry": [entry, judged]}, "z": {}})
        self.assertIn("carried passes by chain length: 1: 1, 2: 1", out.getvalue())
        self.assertIn("judged: y (chain 2)", out.getvalue())
        self.assertNotIn("judged: x", out.getvalue())


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
        self.assertIn("step python3 scripts/gate_support.py --check-scientific", gate)
        self.assertIn("scientific_records.report_status(ROOT, owed, current, check=True)",
                      self.read("scripts/gate_support.py"))
        self.assertIn("step python3 scripts/calibration_status.py --check", gate)
        self.assertNotIn("test.sh scientific", gate)
        self.assertNotIn("test.sh scientific", self.read("scripts/gate_release_candidate.sh"))
        workflow = self.read(".github/workflows/ci.yml")
        job = workflow[workflow.index("\n  scientific:"):workflow.index("\n  python-lint:")]
        self.assertNotIn("nextest", job)
        self.assertIn("--section scientific", job)


if __name__ == "__main__":
    unittest.main()


class ExactTestNameTests(unittest.TestCase):
    def test_inline_modules_are_part_of_the_name_and_a_missing_test_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "crates/c/src/analysis/execute.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                "pub fn run() {}\n#[cfg(test)]\nmod tests {\n    mod only {\n"
                "        #[test]\n        fn pinned_truth() {}\n    }\n}\n#[test]\nfn top_level() {}\n",
                encoding="utf-8",
            )
            row = {"evidence_test": "crates/c/src/analysis/execute.rs"}
            info = {"selector": "c::lib::analysis::execute::pinned_truth"}
            self.assertEqual(sr.exact_test_name(root, row, info), "analysis::execute::tests::only::pinned_truth")
            info = {"selector": "c::lib::analysis::execute::top_level"}
            self.assertEqual(sr.exact_test_name(root, row, info), "analysis::execute::top_level")
            with self.assertRaises(sr.SelectorError):
                sr.exact_test_name(root, row, {"selector": "c::lib::analysis::execute::absent"})

    def test_a_run_that_matches_no_test_is_an_error_not_a_failure(self):
        from types import SimpleNamespace

        def run(*_args, **_kwargs):
            return SimpleNamespace(returncode=0, stdout="\nrunning 0 tests\n\ntest result: ok. 0 passed\n", stderr="")
        info = {"selector": "c::lib::m::t", "crate": "c", "libtest": "m::t"}
        with self.assertRaises(sr.SelectorError):
            sr.capture_one(Path("/tmp"), "/bin/test", info, {}, run)
