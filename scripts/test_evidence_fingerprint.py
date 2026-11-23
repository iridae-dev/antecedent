#!/usr/bin/env python3
"""The shared test fingerprint: one implementation for scientific and calibration records."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import evidence_fingerprint as ef  # noqa: E402
import scientific_records as sr  # noqa: E402


def package(root: Path, name: str, deps=()):
    return {
        "name": name,
        "manifest_path": str(root / "crates" / name / "Cargo.toml"),
        "dependencies": [{"name": d, "kind": None} for d in deps],
    }


class _TreeCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.metadata = {"packages": [package(self.root, "leaf"), package(self.root, "top", ["leaf"])]}
        for name in ("leaf", "top"):
            self.write(f"crates/{name}/Cargo.toml", f'[package]\nname = "{name}"\nversion = "1.0.0"\n')
            self.write(f"crates/{name}/src/lib.rs", f"pub fn {name}() -> u8 {{ 1 }}\n")
        self.write("crates/top/tests/cov.rs", "#[test]\nfn cov_case() {}\n")
        self.write("crates/top/tests/dgp.rs", "pub fn dgp() {}\n")
        self.write("Cargo.toml", '[workspace]\nmembers = ["crates/*"]\n')
        self.write("Cargo.lock", "version = 4\n")
        self.write("rust-toolchain.toml", '[toolchain]\nchannel = "1.90"\n')

    def write(self, rel: str, text: str):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def record(self, **extra):
        return {
            "id": "cov.x",
            "test": "crates/top/tests/cov.rs::cov_case",
            "dgp": "crates/top/tests/dgp.rs::dgp",
            **extra,
        }

    def fingerprint(self, record=None, platform=True, environ=None):
        fp = ef.Fingerprinter(self.root, self.metadata, "", platform=platform, environ=environ or {})
        return ef.coverage_fingerprint(fp, record or self.record())


class SharedFingerprintTests(_TreeCase):
    def test_scientific_records_use_the_same_fingerprinter(self):
        self.assertIs(sr.Fingerprinter, ef.Fingerprinter)
        self.assertIs(sr.locate, ef.locate)

    def test_a_coverage_record_fingerprints_like_the_scientific_row_of_its_test(self):
        fp = ef.Fingerprinter(self.root, self.metadata, "")
        row = {
            "evidence_test": "crates/top/tests/cov.rs",
            "evidence_assertion": "cov_case",
            "fixtures": ["crates/top/tests/dgp.rs"],
        }
        self.assertEqual(self.fingerprint(), fp.fingerprint(row))

    def test_coverage_row_splits_the_selector_and_keeps_a_foreign_dgp_as_a_fixture(self):
        row = ef.coverage_row("crates/a/tests/t.rs::name", "crates/a/tests/t.rs::dgp")
        self.assertEqual((row["evidence_test"], row["evidence_assertion"], row["fixtures"]), ("crates/a/tests/t.rs", "name", []))
        self.assertEqual(ef.coverage_row("crates/a/tests/t.rs::n", "crates/b/tests/d.rs::g")["fixtures"], ["crates/b/tests/d.rs"])

    def test_dependency_dgp_and_test_edits_change_it_and_comments_do_not(self):
        before = self.fingerprint()
        self.write("crates/top/tests/cov.rs", "// note\n#[test]\nfn cov_case() {}\n")
        self.assertEqual(before, self.fingerprint())
        self.write("crates/leaf/src/lib.rs", "pub fn leaf() -> u8 { 2 }\n")
        after_leaf = self.fingerprint()
        self.assertNotEqual(before, after_leaf)
        self.write("crates/top/tests/dgp.rs", "pub fn dgp() { let _ = 1; }\n")
        self.assertNotEqual(after_leaf, self.fingerprint())

    def test_platform_can_be_left_out(self):
        self.assertEqual(self.fingerprint(platform=False), self.fingerprint(platform=False))
        self.assertNotEqual(self.fingerprint(platform=False), self.fingerprint(platform=True))

    def test_an_unlocatable_test_has_no_fingerprint(self):
        self.assertIsNone(self.fingerprint(self.record(test="not-a-test-path::x")))


class _InputCase(_TreeCase):

    ROW = {"evidence_test": "crates/top/tests/cov.rs", "evidence_assertion": "cov_case", "fixtures": []}

    def fp(self, environ=None):
        return ef.Fingerprinter(self.root, self.metadata, "", environ=environ or {})

    def closure_fp(self, environ=None):
        return self.fp(environ).fingerprint(self.ROW)


class EmbeddedAndRuntimeInputTests(_InputCase):
    """Inputs the sources reach without declaring them."""

    def test_the_version_salt_owes_every_record(self):
        before = self.closure_fp()
        with mock.patch.object(ef, "FINGERPRINT_VERSION", ef.FINGERPRINT_VERSION + 1):
            self.assertNotEqual(before, self.closure_fp())

    def test_editing_an_include_str_target_owes_its_dependents(self):
        self.write("crates/leaf/data/reg.json", '{"a": 1}\n')
        self.write("crates/leaf/src/lib.rs", 'pub const R: &str = include_str!("../data/reg.json");\n')
        before = self.closure_fp()
        self.write("crates/leaf/data/reg.json", '{"a": 2}\n')
        self.assertNotEqual(before, self.closure_fp())

    def test_concat_manifest_dir_and_include_bytes_and_include_resolve(self):
        self.write("crates/top/data/b.bin", "one")
        self.write("crates/top/src/part.rs", "pub fn part() {}\n")
        self.write("crates/top/data/c.json", "{}")
        self.write(
            "crates/top/src/lib.rs",
            'pub const B: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/b.bin"));\n'
            'include!("part.rs");\n'
            'pub const C: &str = include_str!(\n    concat!(env!("CARGO_MANIFEST_DIR"), "/data", "/c.json"));\n',
        )
        before = self.closure_fp()
        for rel, text in (("crates/top/data/b.bin", "two"), ("crates/top/data/c.json", '{"x":1}')):
            self.write(rel, text)
            after = self.closure_fp()
            self.assertNotEqual(before, after, rel)
            before = after

    def test_an_include_in_a_comment_or_string_is_not_followed(self):
        self.write(
            "crates/top/src/lib.rs",
            '// include_str!("../data/none.json")\npub const S: &str = "include_str!(x)";\n',
        )
        self.closure_fp()

    def test_an_unresolvable_include_raises(self):
        for body in (
            'include_str!(concat!(env!("SOME_DIR"), "/x.json"))',
            "include_str!(path_from_macro!())",
            'include_str!("../data/missing.json")',
            'include_str!("../../../../outside.json")',
        ):
            self.write("crates/top/src/lib.rs", f"pub const X: &str = {body};\n")
            with self.assertRaises(ef.UnresolvedInput, msg=body):
                self.closure_fp()
            # A caller asking for a coverage fingerprint is told loudly, not given None.
            with self.assertRaises(ef.UnresolvedInput, msg=body):
                self.fingerprint()

    def test_an_out_dir_include_needs_a_build_script_and_hashes_it(self):
        self.write("crates/top/src/lib.rs", 'include!(concat!(env!("OUT_DIR"), "/gen.rs"));\n')
        with self.assertRaises(ef.UnresolvedInput):
            self.closure_fp()
        self.write("crates/top/build.rs", "fn main() {}\n")
        before = self.closure_fp()
        self.write("crates/top/build.rs", "fn main() { let _ = 1; }\n")
        self.assertNotEqual(before, self.closure_fp())

    def test_a_test_file_include_target_owes_the_test(self):
        self.write("crates/top/tests/oracle_data/ref.json", "[1]")
        self.write(
            "crates/top/tests/cov.rs",
            '#[test]\nfn cov_case() { let _ = include_str!("oracle_data/ref.json"); }\n',
        )
        before = self.closure_fp()
        self.write("crates/top/tests/oracle_data/ref.json", "[2]")
        self.assertNotEqual(before, self.closure_fp())

    def test_a_runtime_read_parity_file_owes_the_test_that_reads_it(self):
        self.write("parity/read.toml", "a = 1\n")
        self.write("parity/other.toml", "b = 1\n")
        self.write(
            "crates/top/tests/cov.rs",
            '#[test]\nfn cov_case() {\n'
            '    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join("parity/read.toml");\n'
            "    let _ = std::fs::read_to_string(p);\n}\n",
        )
        before = self.closure_fp()
        self.write("parity/other.toml", "b = 2\n")
        self.assertEqual(before, self.closure_fp())
        self.write("parity/read.toml", "a = 2\n")
        self.assertNotEqual(before, self.closure_fp())

    def _data_reads(self, crate: str, test_crate: str | None):
        fp = ef.Fingerprinter(self.root, self.metadata, "", platform=False, environ={})
        with fp.tracking() as used:
            fp.crate_scan(crate)
        reads = used.data_reads(None if test_crate is None else fp.crate_dir(test_crate))
        return {p.relative_to(self.root).as_posix() for p in reads}

    def test_cfg_test_reads_count_only_for_their_crates_unit_tests(self):
        self.write("parity/lib.toml", "a = 1\n")
        self.write("parity/unit.toml", "a = 1\n")
        self.write("parity/module.toml", "a = 1\n")
        self.write(
            "crates/leaf/src/lib.rs",
            'pub fn leaf() -> &str { include_str!("../../../parity/lib.toml") }\n'
            "#[cfg(test)]\nmod unit {\n"
            '    #[test]\n    fn reads() { let _ = include_str!("../../../parity/unit.toml"); }\n}\n'
            "#[cfg(test)]\nmod gated;\n"
            '#[cfg(any(test, feature = "x"))]\n'
            'pub fn either() -> &str { include_str!("../../../parity/lib.toml") }\n',
        )
        self.write("crates/leaf/src/gated.rs", 'pub fn g() -> &str { include_str!("../../../parity/module.toml") }\n')
        self.assertEqual(self._data_reads("leaf", None), {"parity/lib.toml"})
        self.assertEqual(self._data_reads("leaf", "top"), {"parity/lib.toml"})
        self.assertEqual(self._data_reads("leaf", "leaf"), {"parity/lib.toml", "parity/unit.toml", "parity/module.toml"})

    def test_cfg_test_reads_in_a_test_target_are_data_reads(self):
        self.write("parity/t.toml", "a = 1\n")
        self.write(
            "crates/top/tests/cov.rs",
            '#[cfg(test)]\nmod m {\n    #[test]\n    fn cov_case() { let _ = include_str!("../../../parity/t.toml"); }\n}\n',
        )
        fp = ef.Fingerprinter(self.root, self.metadata, "", platform=False, environ={})
        with fp.tracking() as used:
            fp.rust_test_side({"evidence_test": "crates/top/tests/cov.rs", "evidence_assertion": "cov_case"}, "top")
        self.assertIn(self.root / "parity/t.toml", used.data_reads(None))

    def test_a_shared_directory_member_no_source_reads_is_reported_unread(self):
        # A sibling test's oracle sits beside a shared helper module. It is hashed with the
        # directory, but only the member this target embeds is one of its data reads.
        self.write("crates/top/tests/oracle/support.rs", "pub fn helper() {}\n")
        self.write("crates/top/tests/oracle/mine.json", "{}")
        self.write("crates/top/tests/oracle/other.json", "{}")
        self.write("crates/top/tests/oracle/other.py", "print()\n")
        self.write("crates/top/tests/t.rs",
                   '#[path = "oracle/support.rs"]\nmod support;\n'
                   'const M: &str = include_str!("oracle/mine.json");\n#[test]\nfn a() {}\n')
        fp = ef.Fingerprinter(self.root, self.metadata, "", platform=False, environ={})
        with fp.tracking() as used:
            fp.parts({"evidence_test": "crates/top/tests/t.rs", "evidence_assertion": "a"})
        oracle = self.root / "crates/top/tests/oracle"
        self.assertEqual(used.shared_tree_files - used.declared_files - used.data_reads(None),
                         {oracle / "other.json", oracle / "other.py"})

    def test_host_metadata_in_a_read_directory_is_never_an_input(self):
        self.write("parity/data/a.toml", "a = 1\n")
        self.write("parity/data/.DS_Store", "finder")
        self.write("parity/data/._a.toml", "appledouble")
        self.assertEqual({p.name for p in ef.tree_files(self.root / "parity/data")}, {"a.toml"})

    def test_runtime_reads_from_sources_in_every_literal_form(self):
        forms = {
            "../../parity/lit.toml": 'const P: &str = "../../parity/lit.toml";\n',
            "parity/cat.toml": 'fn f() { let _ = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/cat.toml"); }\n',
            "tests/data/rel.csv": 'fn f() { let _ = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/rel.csv"); }\n',
            "conformance/x/expected.json": 'fn f() { let _ = Path::new("conformance").join("x").join("expected.json"); }\n',
            "conformance/dir/a.json": 'fn f() { let _ = root.join("conformance").join("dir").join(name); }\n',
        }
        for target, source in forms.items():
            rel = target if target.startswith(("parity", "conformance")) else f"crates/top/{target}"
            self.write(rel, "v1")
            self.write("crates/top/src/lib.rs", source)
            before = self.closure_fp()
            self.write(rel, "v2")
            self.assertNotEqual(before, self.closure_fp(), target)

    def test_a_path_literal_to_nothing_is_ignored(self):
        self.write("crates/top/src/lib.rs", 'const P: &str = "../../parity/not/there.toml";\n')
        before = self.closure_fp()
        self.write("parity/unrelated.toml", "x")
        self.assertEqual(before, self.closure_fp())

    def test_python_rows_hash_repo_paths_and_local_imports(self):
        row = {"evidence_test": "python/tests/test_x.py", "evidence_assertion": "test_a", "fixtures": []}
        self.write("python/tests/test_x.py", (
            "from pathlib import Path\nfrom _helper import load\n"
            'ROOT = Path(__file__).resolve().parents[2]\n'
            'def test_a():\n    load(ROOT / "parity" / "py.toml")\n    load(ROOT / "conformance/z/e.json")\n'
        ))
        self.write("python/tests/_helper.py", "def load(p):\n    return p\n")
        self.write("parity/py.toml", "a=1\n")
        self.write("conformance/z/e.json", "{}")
        self.write("parity/unrelated.toml", "a=1\n")

        def fingerprint(environ=None):
            return ef.Fingerprinter(self.root, self.metadata, "", environ=environ or {}).fingerprint(row)

        before = fingerprint()
        self.write("parity/unrelated.toml", "a=2\n")
        self.assertEqual(before, fingerprint())
        for rel, text in (
            ("parity/py.toml", "a=2\n"),
            ("conformance/z/e.json", '{"v":1}'),
            ("python/tests/_helper.py", "def load(p):\n    return None\n"),
        ):
            self.write(rel, text)
            after = fingerprint()
            self.assertNotEqual(before, after, rel)
            before = after
        self.write("python/tests/test_x.py", 'import os\ndef test_a():\n    os.environ.get("ANTECEDENT_PYFLAG")\n')
        before = fingerprint()
        self.assertNotEqual(before, fingerprint({"ANTECEDENT_PYFLAG": "1"}))

    def test_an_antecedent_env_value_change_owes_and_other_names_do_not(self):
        self.write(
            "crates/top/src/lib.rs",
            'pub fn f() { let _ = std::env::var("ANTECEDENT_KNOB"); let _ = std::env::var_os("HOME"); }\n',
        )
        base = {"ANTECEDENT_KNOB": "a", "HOME": "/h/one"}
        for fingerprint in (self.closure_fp,):
            before = fingerprint(base)
            self.assertNotEqual(before, fingerprint({**base, "ANTECEDENT_KNOB": "b"}))
            self.assertNotEqual(before, fingerprint({"HOME": "/h/one"}))  # unset is a value
            self.assertEqual(before, fingerprint({**base, "HOME": "/h/two"}))

    def test_env_names_are_found_in_every_read_form(self):
        forms = (
            'const E: &str = "ANTECEDENT_CONST";\nfn f() { let _ = std::env::var(E); }\n',
            'fn f() { let _ = option_env!("ANTECEDENT_OPT"); }\n',
            'fn f() { let _ = std::env::var_os("ANTECEDENT_OS"); }\n',
        )
        for source, name in zip(forms, ("ANTECEDENT_CONST", "ANTECEDENT_OPT", "ANTECEDENT_OS"), strict=True):
            self.write("crates/top/src/lib.rs", source)
            self.assertNotEqual(self.closure_fp(), self.closure_fp({name: "1"}), name)

    def test_a_new_non_antecedent_env_read_is_a_listed_input(self):
        before = self.closure_fp()
        self.write("crates/top/src/lib.rs", 'pub fn f() { let _ = std::env::var("HOME"); }\n')
        self.assertNotEqual(before, self.closure_fp())
        part = self.fp().parts(self.ROW)["env"]
        self.assertEqual(part, {"hashed": {}, "named_only": ["HOME"]})

    def test_build_script_declared_inputs_are_hashed(self):
        self.write("crates/leaf/build.rs", 'fn main() {\n    println!("cargo:rerun-if-changed=assets");\n    println!("cargo:rerun-if-env-changed=ANTECEDENT_BUILD");\n}\n')
        self.write("crates/leaf/assets/table.txt", "1")
        before = self.closure_fp()
        self.write("crates/leaf/assets/table.txt", "2")
        self.assertNotEqual(before, self.closure_fp())
        self.assertNotEqual(self.closure_fp(), self.closure_fp({"ANTECEDENT_BUILD": "x"}))

    DYNAMIC_SOURCE = (
        'pub fn f(n: &str) { let _ = std::env::var(n); '
        'let _ = format!("{}/x", env!("CARGO_MANIFEST_DIR")); }\n'
    )

    def declare(self, *rows):
        body = "".join(
            "[[input]]\n"
            f"source = {json.dumps(r['source'])}\n"
            f"inputs = {json.dumps(r.get('inputs', []))}\n"
            f"env = {json.dumps(r.get('env', []))}\n"
            'why = "test"\n\n'
            for r in rows
        )
        self.write("parity/fingerprint_inputs.toml", body)

    def test_dynamic_reads_are_listed(self):
        self.write("crates/top/src/lib.rs", self.DYNAMIC_SOURCE)
        fp = self.fp()
        fp.crate_scan("top")
        listed = fp.dynamic_inputs()
        self.assertTrue(any("crates/top/src/lib.rs:1" in d and "non-literal" in d for d in listed), listed)
        self.assertTrue(any("CARGO_MANIFEST_DIR" in d for d in listed), listed)

    def test_an_undeclared_dynamic_site_raises_naming_file_and_line(self):
        self.write("crates/top/src/lib.rs", "// pad\n" + self.DYNAMIC_SOURCE)
        for fingerprint in (self.closure_fp,):
            with self.assertRaisesRegex(ef.UnresolvedInput, r"crates/top/src/lib\.rs:2"):
                fingerprint()
        # A declaration for another file does not cover it.
        self.declare({"source": "crates/leaf/src/lib.rs"})
        with self.assertRaises(ef.UnresolvedInput):
            self.closure_fp()

    def test_a_dynamic_site_in_a_test_file_raises_and_a_declaration_covers_it(self):
        self.write(
            "crates/top/tests/cov.rs",
            '#[test]\nfn cov_case() { let r = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."); '
            "let _ = r.join(name); }\n",
        )
        with self.assertRaisesRegex(ef.UnresolvedInput, r"crates/top/tests/cov\.rs:2"):
            self.closure_fp()
        self.write("parity/seen.toml", "a = 1\n")
        self.declare({"source": "crates/top/tests/cov.rs", "inputs": ["parity/seen.toml"]})
        before = self.closure_fp()
        self.write("parity/seen.toml", "a = 2\n")
        self.assertNotEqual(before, self.closure_fp())

    def test_declared_inputs_are_hashed_in_both_modes_with_globs_dirs_and_env(self):
        self.write("crates/top/src/lib.rs", self.DYNAMIC_SOURCE)
        self.write("parity/dir/a.toml", "a = 1\n")
        self.write("crates/top/Cargo.extra", "x")
        self.write("conf/one.toml", "1")
        self.declare(
            {
                "source": "crates/top/src/lib.rs",
                "inputs": ["parity/dir", "crates/*/Cargo.toml", "conf/one.toml"],
                "env": ["ANTECEDENT_DECLARED", "HOME"],
            }
        )
        for round_, fingerprint in enumerate(
            (self.closure_fp,), start=2
        ):
            before = fingerprint()
            for rel, text in (
                ("parity/dir/a.toml", f"a = {round_}\n"),
                (f"parity/dir/new{round_}.toml", "n = 1\n"),
                ("conf/one.toml", str(round_)),
                ("crates/leaf/Cargo.toml", f'[package]\nname = "leaf"\nversion = "1.0.0"\n[features]\nx{round_} = []\n'),
            ):
                self.write(rel, text)
                after = fingerprint()
                self.assertNotEqual(before, after, rel)
                before = after
            self.assertNotEqual(before, fingerprint({"ANTECEDENT_DECLARED": "1"}))
            self.assertEqual(fingerprint({"HOME": "/a"}), fingerprint({"HOME": "/b"}))

    def test_a_malformed_declaration_raises(self):
        self.write("parity/fingerprint_inputs.toml", '[[input]]\nsource = "x.rs"\ninputs = []\nenv = []\nwhy = ""\n')
        with self.assertRaises(ef.UnresolvedInput):
            self.closure_fp()

    def test_a_path_included_helper_owes_its_includers(self):
        self.write("crates/top/tests/helpers_dir/retained/mod.rs", "pub fn h() -> u8 { 1 }\n")
        self.write(
            "crates/top/tests/cov.rs",
            '#[path = "helpers_dir/retained/mod.rs"]\nmod retained;\n#[test]\nfn cov_case() {}\n',
        )
        # A module the test file reaches only through #[path], outside the conventional helper dirs.
        self.write("crates/top/other/shared.rs", "pub fn s() -> u8 { 1 }\n")
        self.write(
            "crates/top/tests/cov.rs",
            '#[path = "helpers_dir/retained/mod.rs"]\nmod retained;\n'
            '#[path = "../other/shared.rs"]\nmod shared;\n#[test]\nfn cov_case() {}\n',
        )
        before = self.closure_fp()
        for rel in ("crates/top/tests/helpers_dir/retained/mod.rs", "crates/top/other/shared.rs"):
            self.write(rel, "pub fn h() -> u8 { 2 }\n")
            after = self.closure_fp()
            self.assertNotEqual(before, after, rel)
            before = after

    def test_a_path_module_in_crate_source_is_followed(self):
        self.write("crates/leaf/other/mod_a.rs", "pub fn a() -> u8 { 1 }\n")
        self.write("crates/leaf/src/lib.rs", '#[path = "../other/mod_a.rs"]\nmod a;\npub fn leaf() {}\n')
        before = self.closure_fp()
        self.write("crates/leaf/other/mod_a.rs", "pub fn a() -> u8 { 2 }\n")
        self.assertNotEqual(before, self.closure_fp())

    def test_a_path_inside_an_inline_module_is_followed(self):
        self.write("crates/leaf/src/inner/x.rs", "pub fn x() -> u8 { 1 }\n")
        self.write("crates/leaf/data/y.rs", "pub fn y() {}\n")
        self.write(
            "crates/leaf/src/lib.rs",
            'mod wrap {\n    #[path = "../../data/y.rs"]\n    mod y;\n}\n',
        )
        before = self.closure_fp()
        self.write("crates/leaf/data/y.rs", "pub fn y() { let _ = 1; }\n")
        self.assertNotEqual(before, self.closure_fp())

    def test_a_path_leaving_the_repository_raises(self):
        self.write("crates/leaf/src/lib.rs", '#[path = "../../../../../outside.rs"]\nmod o;\n')
        with self.assertRaises(ef.UnresolvedInput):
            self.closure_fp()

    def test_a_cross_file_env_constant_needs_no_declaration(self):
        self.write("crates/top/tests/consts.rs", 'pub const KNOB: &str = "ANTECEDENT_CROSS";\n')
        self.write(
            "crates/top/tests/cov.rs",
            '#[path = "consts.rs"]\nmod consts;\n#[test]\nfn cov_case() { let _ = std::env::var(consts::KNOB); }\n',
        )
        self.assertNotEqual(self.closure_fp(), self.closure_fp({"ANTECEDENT_CROSS": "1"}))

    def test_the_committed_declarations_name_existing_sources_and_inputs(self):
        import glob
        import tomllib

        repo = Path(__file__).resolve().parent.parent
        rows = tomllib.loads((repo / ef.INPUT_DECLARATIONS).read_text(encoding="utf-8"))["input"]
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue((repo / row["source"]).is_file(), row["source"])
            self.assertTrue(row["why"].strip())
            for pattern in row["inputs"]:
                self.assertTrue(glob.glob(str(repo / pattern)), pattern)


class CycleGuardTests(_InputCase):
    """Support status is derived from the records: re-deriving it must not owe the records."""

    COVERAGE = "crates/antecedent-io-core/data/coverage_records.json"
    SUPPORT = "crates/antecedent/data/support_matrix.json"
    UNIT = "crates/antecedent-io-core/data/unit_support_status.json"

    def setUp(self):
        super().setUp()
        self.coverage = {
            "records": [{"id": "cov.a", "observed": "0.95", "mcse": "0.01", "calibration_sha": "a" * 40}],
            "attesting_record_ids": ["cov.a"],
        }
        self.support = {
            "na_rules": [],
            "supported": [
                {
                    "query": "AverageEffect",
                    "estimators": ["aipw"],
                    "route_estimator": "aipw",
                    "status": "unsupported",
                    "missing_evidence": ["scientific_test"],
                }
            ],
        }
        self.unit = {
            "graphless": [{"id": "g", "status": "unsupported", "missing_evidence": ["x"]}],
            "capabilities": [{"id": "c", "status": "unsupported", "missing_evidence": ["y"]}],
        }
        self.write_json()
        self.write(
            "crates/top/src/lib.rs",
            'pub const C: &str = include_str!("../../antecedent-io-core/data/coverage_records.json");\n'
            'pub const S: &str = include_str!("../../antecedent/data/support_matrix.json");\n'
            'pub const U: &str = include_str!("../../antecedent-io-core/data/unit_support_status.json");\n',
        )

    def write_json(self):
        self.write(self.COVERAGE, json.dumps(self.coverage, indent=2))
        self.write(self.SUPPORT, json.dumps(self.support, indent=2))
        self.write(self.UNIT, json.dumps(self.unit, indent=2))

    def test_status_only_changes_leave_fingerprints_unchanged(self):
        before = self.closure_fp()
        self.coverage["attesting_record_ids"] = []
        self.support["supported"][0]["status"] = "supported"
        self.support["supported"][0]["missing_evidence"] = []
        for array in ("graphless", "capabilities"):
            self.unit[array][0]["status"] = "supported"
            self.unit[array][0]["missing_evidence"] = []
        self.write_json()
        self.assertEqual(before, self.closure_fp())

    def test_a_repeated_measurement_leaves_fingerprints_unchanged(self):
        before = self.closure_fp()
        self.coverage["records"][0].update(
            observed="0.90", mcse="0.02", replicates=7, calibration_sha="other",
            grid=[{"point": 0, "n_min": 1, "n_max": 1, "observed": "0.90", "mcse": "0.02",
                   "replicates": 7, "boundary": False}],
        )
        self.write_json()
        moved = self.closure_fp()
        self.coverage["records"][0]["grid"][0].update(observed="0.5", mcse="0.1", replicates=9)
        self.write_json()
        self.assertEqual(moved, self.closure_fp())
        self.coverage["records"][0]["grid"][0].update(boundary=True)
        self.write_json()
        self.assertNotEqual(moved, self.closure_fp())
        del self.coverage["records"][0]["grid"]
        self.write_json()
        self.assertEqual(before, self.closure_fp())

    def test_evidence_changes_change_fingerprints(self):
        for mutate in (
            lambda: self.coverage["records"][0].update(boundary=True),
            lambda: self.coverage["records"][0].update(n_max=9),
            lambda: self.coverage["records"][0].update(id="cov.b"),
            lambda: self.support["supported"][0].update(estimators=["ols"]),
            lambda: self.support["supported"].append({"query": "Q"}),
            lambda: self.unit["graphless"][0].update(id="g2"),
            lambda: self.unit["capabilities"].append({"id": "c2"}),
        ):
            before = self.closure_fp()
            mutate()
            self.write_json()
            after = self.closure_fp()
            self.assertNotEqual(before, after)

    def test_a_ledger_toml_read_is_hashed_without_its_stored_fingerprints(self):
        self.write("parity/coverage_records.toml", '[[record]]\nid = "a"\nobserved = "1"\nfingerprint = "x"\n')
        self.write("crates/top/src/lib.rs", 'pub const R: &str = include_str!("../../../parity/coverage_records.toml");\n')
        before = self.closure_fp()
        self.write("parity/coverage_records.toml", '[[record]]\nid = "a"\nobserved = "1"\nfingerprint = "y"\n')
        self.assertEqual(before, self.closure_fp())
        self.write("parity/coverage_records.toml", '[[record]]\nid = "a"\nobserved = "2"\nfingerprint = "y"\n')
        self.assertNotEqual(before, self.closure_fp())

    def test_the_normalizer_table_names_the_embedded_registries(self):
        self.assertIn(self.COVERAGE, ef.DERIVED_STATUS_FIELDS)
        self.assertIn(self.SUPPORT, ef.DERIVED_STATUS_FIELDS)
        self.assertIn(self.UNIT, ef.DERIVED_STATUS_FIELDS)
        data = {"a": [{"status": 1, "keep": 2, "b": {"status": 3}}], "status": 4}
        ef._drop_fields(data, ["**", "status"])
        self.assertEqual(data, {"a": [{"keep": 2, "b": {}}]})


if __name__ == "__main__":
    unittest.main()
