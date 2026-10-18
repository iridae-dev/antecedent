#!/usr/bin/env python3
"""Negative and actual-history oracles for restricted Rust source waivers."""

import unittest
from unittest.mock import patch

import _rust_equivalence as proof
import calibration_facets as facets

PATH = "crates/example/src/analysis/parent.rs"
OLD = "pub use engine::nested::{OLD};\npub fn value() -> f64 { 1.0 }\n"
NEW = "// reviewed docs\npub use engine::nested::{\n OLD, NEW,\n};\npub fn value() -> f64 { 1.0 }\n"


def tree(source, caller=""):
    return {
        "crates/example/Cargo.toml": '[package]\nname="example"\n',
        PATH: source,
        "crates/example/src/analysis/mod.rs": "pub mod parent;",
        "crates/example/src/lib.rs": "pub mod analysis;",
        "crates/example/tests/caller.rs": caller,
    }


class RustEquivalenceTests(unittest.TestCase):
    def check(self, old=OLD, new=NEW, caller="", kind="rust_pub_use_addition", new_caller=None):
        return proof.prove(
            kind,
            PATH,
            old,
            new,
            tree(old, caller),
            tree(new, caller if new_caller is None else new_caller),
        )

    def test_addition_nested_comments_and_multiline_use(self):
        self.assertIsNone(
            self.check(
                new=NEW.replace("// reviewed docs", "/* outer /* nested */ still comment */")
            )
        )
        self.assertIsNone(
            self.check(
                old="pub fn f(){let x=1;}",
                new="// doc\npub fn f() { let x = 1; }",
                kind="rust_comments",
            )
        )

    def test_literal_expression_call_and_cfg_changes_fail(self):
        for before, after in [
            ("1.0", "2.0"),
            ("1 + 2", "1 - 2"),
            ("f()", "g()"),
            ('"a b"', '"ab"'),
            ('r##"a // b"##', 'r##"ab"##'),
            ("' '", "'x'"),
            ('b"a b"', 'b"ab"'),
            ('#[cfg(feature="a")] fn f(){}', '#[cfg(feature="b")] fn f(){}'),
        ]:
            with self.subTest(before=before):
                self.assertIsNotNone(self.check(old=before, new=after, kind="rust_comments"))
        self.assertIsNotNone(self.check(new=NEW.replace("1.0", "2.0")))

    def test_comment_delimiters_inside_literals_are_retained(self):
        for literal in ['"// not a comment"', 'r#"/* not a comment */"#', "'/'", "b'/'"]:
            old = f"const X: &str = {literal};"
            self.assertIsNone(self.check(old=old, new="/* docs */\n" + old, kind="rust_comments"))
        self.assertNotEqual(proof.tokens('"//a"'), proof.tokens('"//b"'))

    def test_token_merges_and_operator_jointness_fail(self):
        for before, after in [
            ("a b", "ab"),
            ("a > > b", "a >> b"),
            ("a < < b", "a << b"),
            ("1e-2", "1e - 2"),
            ("'a", "' a"),
            ('b"x"', 'b "x"'),
            ("b'x'", "b 'x'"),
        ]:
            self.assertNotEqual(proof.tokens(before), proof.tokens(after))

    def test_exports_cannot_remove_retarget_alias_glob_or_change_execution(self):
        for changed in [
            NEW.replace("OLD, NEW,", "NEW,"),
            NEW.replace("engine::nested", "other::nested"),
            NEW.replace("OLD, NEW,", "OLD, NEW as N,"),
            NEW.replace("OLD, NEW,", "OLD, *,"),
            NEW.replace("1.0", "2.0"),
            NEW.replace("OLD, NEW,", "OLD, self,"),
            NEW.replace("pub use engine", "pub use ::engine"),
        ]:
            self.assertIsNotNone(self.check(new=changed))

    def test_qualified_unrelated_same_name_import_is_safe(self):
        caller = "use example::analysis::other::NEW; fn f()->f64 {NEW}"
        self.assertIsNone(self.check(caller=caller))
        # New callers may deliberately opt into the new public export; the
        # equivalence claim concerns preserving old caller bindings.
        self.assertIsNone(
            self.check(new_caller="use example::analysis::parent::NEW; fn f()->f64 {NEW}")
        )

    def test_target_alias_glob_reexport_or_old_reference_is_not_certified(self):
        for caller in [
            "use example::analysis::parent as p; fn f(){p::NEW;}",
            "use example::analysis::parent::*; fn f(){NEW;}",
            "pub use example::analysis::parent::*;",
            "use example::analysis as a; fn f(){a::parent::NEW;}",
            "use example::analysis::{self}; fn f(){analysis::parent::NEW;}",
            "fn f(){example::analysis::parent::NEW;}",
            "use example::analysis::parent::NEW;",
            "fn f(){NEW;}",
        ]:
            with self.subTest(caller=caller):
                self.assertIsNotNone(self.check(caller=caller))

    def test_macro_and_procedural_attribute_hazards_fail(self):
        for extra in [
            "fn f(){line!();}",
            "fn f(){column!();}",
            "fn f(){file!();}",
            "fn f(){stringify!(x);}",
            'const X:&str=include_str!("x");',
            "#[custom] fn f(){}",
            "#[derive(Custom)] struct F;",
            '#[cfg_attr(feature="a", custom)] fn f(){}',
            "fn f(){unknown!();}",
        ]:
            self.assertIsNotNone(self.check(old=OLD + extra, new=NEW + extra))
        for before, after in [
            ("#[derive(Debug)]\n/// old\nstruct X;", "#[derive(Debug)]\n/// new\nstruct X;"),
            ("#[derive(Debug)]\n/** old */ struct X;", "#[derive(Debug)]\n/** new */ struct X;"),
            (
                "use custom::Debug; #[derive(Debug)] struct X;",
                "use custom::Debug; #[derive(Debug)] struct X;",
            ),
        ]:
            self.assertIsNotNone(self.check(old=OLD + before, new=NEW + after))
        old_tree, new_tree = tree(OLD), tree(NEW)
        old_tree["crates/example/src/analysis/mod.rs"] = "#[custom] pub mod parent;"
        self.assertIsNotNone(
            proof.prove("rust_pub_use_addition", PATH, OLD, NEW, old_tree, new_tree)
        )
        caller = 'const X: &str = include_str!("../src/analysis/parent.rs");'
        self.assertIsNotNone(self.check(caller=caller))
        for caller in [
            '#[path="../src/analysis/parent.rs"] mod alternate;',
            'const X:&str=include_str!(concat!("../src/analysis/", "parent", ".rs"));',
            r'const X:&str=include_str!("../src/analysis/p\x61rent.rs");',
        ]:
            self.assertIsNotNone(self.check(caller=caller))

    def test_ancestor_procedural_derive_shadowing_fails_closed(self):
        before, after = OLD + "#[derive(Debug)] struct X;", NEW + "#[derive(Debug)] struct X;"
        for imports in ["pub use custom::*;", "pub use custom::Shadow as Debug;"]:
            old_tree, new_tree = tree(before), tree(after)
            for sources in (old_tree, new_tree):
                sources["crates/example/src/lib.rs"] = "pub mod analysis; " + imports
            self.assertIsNotNone(
                proof.prove("rust_pub_use_addition", PATH, before, after, old_tree, new_tree)
            )
        old_tree, new_tree = tree(before), tree(after)
        for sources in (old_tree, new_tree):
            sources["crates/example/src/lib.rs"] = "pub mod analysis; pub use api::*;"
            sources["crates/example/src/api.rs"] = "pub use custom::Shadow as Debug;"
        self.assertIsNotNone(
            proof.prove("rust_pub_use_addition", PATH, before, after, old_tree, new_tree)
        )

    def test_compile_context_must_be_unambiguous(self):
        for manifest in [
            '[package]\nname="renamed"',
            '[package]\nname="example"\nbuild="custom.rs"',
            '[package]\nname="example"\n[lib]\npath="different.rs"',
        ]:
            old_tree, new_tree = tree(OLD), tree(NEW)
            old_tree["crates/example/Cargo.toml"] = manifest
            self.assertIsNotNone(
                proof.prove("rust_pub_use_addition", PATH, OLD, NEW, old_tree, new_tree)
            )

    def test_direct_source_reads_and_compile_dependency_scanners_fail(self):
        for caller in [
            'fn f(){std::fs::read_to_string("../src/analysis/parent.rs");}',
            'use std::fs::read as load; fn f(){load("../src/analysis/parent.rs");}',
            'fn f(){std::fs::File::open(concat!("../src/analysis/", "parent.rs"));}',
            r'fn f(){std::fs::read("../src/analysis/p\x61rent.rs");}',
        ]:
            self.assertIsNotNone(self.check(caller=caller))
        old_tree, new_tree = tree(OLD), tree(NEW)
        for sources in (old_tree, new_tree):
            sources["crates/example/Cargo.toml"] += '[dependencies]\nhelper={path="../helper"}\n'
            sources["crates/helper/Cargo.toml"] = '[package]\nname="helper"\n'
            sources["crates/helper/build.rs"] = 'fn main(){std::fs::read_dir("../../crates");}'
        self.assertIsNotNone(
            proof.prove("rust_pub_use_addition", PATH, OLD, NEW, old_tree, new_tree)
        )
        old_tree, new_tree = tree(OLD), tree(NEW)
        for sources in (old_tree, new_tree):
            sources["crates/unrelated/build.rs"] = 'fn main(){std::fs::read_dir("../../crates");}'
        self.assertIsNotNone(
            proof.prove("rust_pub_use_addition", PATH, OLD, NEW, old_tree, new_tree)
        )
        old_tree, new_tree = tree(OLD), tree(NEW)
        for sources in (old_tree, new_tree):
            sources["crates/unrelated/Cargo.toml"] = (
                '[package]\nname="unrelated"\nbuild="scanner.rs"\n'
            )
            sources["crates/unrelated/scanner.rs"] = 'fn main(){std::fs::read_dir("../../crates");}'
        self.assertIsNotNone(
            proof.prove("rust_pub_use_addition", PATH, OLD, NEW, old_tree, new_tree)
        )

    def test_numerical_owner_only_bypasses_replay_after_proof_and_exact_blobs(self):
        old, new = "a" * 40, "b" * 40
        path = "crates/antecedent-estimate/src/example.rs"
        before, after = "pub const X:f64=1.0;", "// docs\npub const X:f64=1.0;"
        old_blob, new_blob = facets.git_blob_identity(before), facets.git_blob_identity(after)
        repo = facets.FakeRepo(
            {old: old, new: new},
            {(old, new): [path]},
            {(old, path): old_blob, (new, path): new_blob},
        )
        manifest = {
            "crates/antecedent-estimate/Cargo.toml": '[package]\nname="antecedent-estimate"\n'
        }
        repo._rust_trees = {old: {path: before, **manifest}, new: {path: after, **manifest}}
        surface = facets.Surface(entries=[("estimator.example", path)])
        certificate = facets.ReviewedEquivalence(
            path, old_blob, new_blob, "rust_comments", "docs", "token proof"
        )
        waiver = facets.Waiver(
            "rust-source-proof",
            old,
            new,
            "reviewer",
            "source proof",
            [path],
            [],
            reviewed_equivalence=[certificate],
        )
        refs = facets.References({}, [], [])

        def check():
            with patch.object(facets, "_crate_deps", return_value={}):
                return facets.validate_waivers(
                    surface,
                    [waiver],
                    [{"id": "sample", "calibration_sha": old}],
                    repo,
                    refs,
                    require_outcome=False,
                )

        self.assertEqual(check().problems, [])
        certificate.to_blob = "c" * 40
        self.assertTrue(any("exact from/to blobs" in x for x in check().problems))
        certificate.to_blob = new_blob
        certificate.from_blob = ""
        self.assertTrue(any("exact from/to blobs" in x for x in check().problems))
        certificate.from_blob = old_blob
        certificate.kind = "retained_formula"
        self.assertTrue(
            any("numerical owners require exercised numeric replay" in x for x in check().problems)
        )
        certificate.kind = "rust_comments"
        repo._rust_trees[new][path] = after.replace("1.0", "2.0")
        result = check()
        self.assertTrue(any("Rust equivalence proof failed" in x for x in result.problems))
        self.assertTrue(
            any("numerical owners require exercised numeric replay" in x for x in result.problems)
        )

    def test_actual_facade_history_has_exact_qualified_reference_proof(self):
        repo = facets.Repo()
        old = repo.rust_sources("c30e10304f4bd85a01ad81dd133a360ddc95c2f1")
        new = repo.rust_sources("e15c4b0c33daf638d11f64b6a7fff6e3bc216022")
        path = "crates/antecedent/src/analysis/sample_split_parent_set_effect.rs"
        self.assertIn(path, old)
        self.assertIsNone(
            proof.prove("rust_pub_use_addition", path, old[path], new[path], old, new)
        )


class TerminalObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = facets.Repo()
        cls.old = repo.rust_sources("c30e10304f4bd85a01ad81dd133a360ddc95c2f1")
        cls.new = repo.rust_sources("HEAD")
        cls.path = "crates/antecedent-stats/src/ci/calibration.rs"

    def check(self, path=None, before=None, after=None, extra=None, complete=False):
        path = path or self.path
        old, new = dict(self.old), dict(self.new)
        if not complete:
            # Negative fixtures retain real primitive source and its entire crate,
            # compile manifests/build scripts, and reviewed Python reader context.
            def fixture(sources):
                return {
                    p: text
                    for p, text in sources.items()
                    if p.startswith("crates/antecedent-stats/")
                    or p.startswith("python/")
                    or p.endswith("Cargo.toml")
                    or p.endswith("build.rs")
                    or p == path
                }

            old, new = fixture(old), fixture(new)
        if extra:
            old.update(extra)
            new.update(extra)
        before = old[path] if before is None else before
        after = new[path] if after is None else after
        old[path], new[path] = before, after
        return proof.prove("rust_terminal_test_observer", path, before, after, old, new)

    def test_actual_exact_two_terminal_primitive_observers(self):
        for path in proof._TERMINAL_TEST_OBSERVERS:
            with self.subTest(path=path):
                self.assertIsNone(self.check(path, complete=True))

    def test_unreviewed_path_and_changed_primitive_binding_reject(self):
        self.assertIsNotNone(self.check(path="crates/antecedent-estimate/src/propensity/knn.rs"))
        self.assertIsNotNone(self.check(before=self.old[self.path].replace("2000", "2001", 1)))
        self.assertIsNotNone(self.check(after=self.new[self.path].replace("2000", "2001", 1)))

    def test_format_arguments_literals_aliases_and_position_reject(self):
        source = self.new[self.path]
        begin = source.index("    ::std::eprintln!(")
        end = source.index("    );", begin) + len("    );")
        observer = source[begin:end]
        variants = [
            source.replace("::std::eprintln!", "eprintln!"),
            source.replace('width\\":\\"not_applicable', 'width\\":\\"changed'),
            source[: end - 2] + ", dangerous()" + source[end - 2 :],
            source[:begin] + "    let hidden = dangerous();\n" + source[begin:],
            source[:begin] + source[end:] + "\n" + observer,
            source[:begin] + observer + "\n" + source[begin:],
            source.replace("{ran}", "{trials}"),
        ]
        # Move the exact observer before the last assertion, preserving its text.
        without = source[:begin] + source[end:]
        insertion = without.rfind("    assert!(", 0, begin)
        variants.append(without[:insertion] + observer + "\n" + without[insertion:])
        for variant in variants:
            with self.subTest(variant=variant[begin : begin + 60]):
                self.assertNotEqual(variant, source)
                self.assertIsNotNone(self.check(after=variant))

    def test_source_reads_scanners_macro_shadow_and_build_context_reject(self):
        cases = [
            ("crates/antecedent-stats/src/shadow.rs", "extern crate hostile as std;"),
            (
                "crates/antecedent-stats/src/shadow.rs",
                "macro_rules! eprintln {($($t:tt)*)=>{panic!()}}",
            ),
            ("crates/other/src/read.rs", 'fn f(){std::fs::read_to_string("ci/calibration.rs");}'),
            (
                "crates/other/src/read.rs",
                'fn f(){std::fs::read_to_string("ci/\\x63alibration.rs");}',
            ),
            ("crates/other/src/read.rs", 'fn f(){std::fs::read_dir("src"); let ext="rs";}'),
            ("crates/other/build.rs", 'fn main(){std::fs::read_dir("src");}'),
            ("crates/other/src/read.rs", '#[path="../../../python/src/defaults.rs"] mod imported;'),
        ]
        for path, source in cases:
            with self.subTest(path=path, source=source):
                self.assertIsNotNone(self.check(extra={path: source}))
        manifest = self.new["crates/antecedent-stats/Cargo.toml"]
        manifest = manifest.replace("[dependencies]", '[dependencies]\nstd = "1"')
        self.assertIsNotNone(self.check(extra={"crates/antecedent-stats/Cargo.toml": manifest}))
        changed = self.new["python/src/defaults.rs"].replace(
            '.join("src")', '.join("../../crates")'
        )
        self.assertIsNotNone(self.check(extra={"python/src/defaults.rs": changed}))


class ReviewedLintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = facets.Repo()
        cls.old = repo.rust_sources("70a71dd33")
        cls.new = repo.rust_sources("48da497f4")
        cls.path = "crates/antecedent/tests/sample_split_parent_set_effect_calibration.rs"

    def check(self, *, after=None, path=None, extra=None):
        path = path or self.path
        old, new = dict(self.old), dict(self.new)
        if extra:
            old.update(extra)
            new.update(extra)
        if after is not None:
            new[path] = after
        return proof.prove("rust_reviewed_test_lint", path, old[path], new[path], old, new)

    def test_actual_exact_lint_insertion(self):
        self.assertIsNone(self.check())

    def test_cfg_derive_body_literal_reason_and_other_path_reject(self):
        source = self.new[self.path]
        for changed in (
            source.replace("clippy::too_many_lines", "clippy::other_lint"),
            source.replace("#[allow(", "#[cfg(", 1),
            source.replace("#[allow(", "#[derive(", 1),
            source.replace("keep one complete discovery", "different reason"),
            source.replace("0x5E7_0017", "0x5E7_0018"),
            source + "\nfn extra() {}\n",
        ):
            self.assertNotEqual(changed, source)
            self.assertIsNotNone(self.check(after=changed))
        self.assertIsNotNone(self.check(path="crates/antecedent-stats/src/ci/calibration.rs"))

    def test_tool_registration_and_source_observation_reject(self):
        for path, source in (
            ("crates/antecedent/src/other.rs", "#![register_tool(clippy)]"),
            (
                "crates/other/src/other.rs",
                'fn f(){std::fs::read_to_string("sample_split_parent_set_effect_calibration.rs");}',
            ),
        ):
            self.assertIsNotNone(self.check(extra={path: source}))


if __name__ == "__main__":
    unittest.main()
