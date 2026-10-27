#!/usr/bin/env python3
"""Broken-evidence self-test for scripts/test_evidence.py (gate_support_matrix.sh --self-test).

Builds a throwaway cargo workspace with one conformance fixture and test files
that hide a cited test in every way a text scan fails open on: `#[ignore]`
before `#[test]`, a `#[cfg(...)]` that compiles it out, a block comment, a raw
string, a file no `mod` declares, a Python helper that is not a collected test,
a skipped pytest, a test that never consumes the fixture another test consumes,
a test that never builds the row's axis values, and checked_execution rows whose
cited test keeps its builder alive, is not a test, or does not cover every
licensed estimator. Each broken row, alone, must
be rejected with its expected reason; the positive control must pass, so a
checker that rejects everything cannot satisfy the self-test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

IT_RS = r"""
use antecedent_test_support::{conformance_dir as fixture_root, load_expected_json};

#[test]
fn fixture_with_only_a_tautology() {
    let _expected = load_expected_json(&fixture_root("demo/truth"));
    assert_eq!(1, 1, "{}", axes());
}

#[test]
fn shared_helpers_consume_fixture() {
    let expected = load_expected_json(&fixture_root("demo/truth"));
    assert!(expected.contains("2.0"), "{}", axes());
}

#[test]
fn shared_helpers_are_not_called() {
    assert!(true, "{}", axes());
}

fn fake_root(name: &str) -> String {
    name.to_string()
}

#[test]
fn relative_string_without_conformance_root() {
    let expected = load_expected_json(&fake_root("demo/truth"));
    assert!(expected.contains("2.0"), "{}", axes());
}

fn load_expected(name: &str) -> String {
    let dir = format!("{}/../../conformance/demo/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(format!("{dir}/expected.json")).unwrap()
}

/// Builds `AverageEffectQuery` on a `Dag` via `.graph(`, `Frequentist`, `RefuteSuite::None`.
fn axes() -> &'static str {
    "AverageEffectQuery Dag .graph( Frequentist RefuteSuite::None"
}

#[test]
fn good_consumes_fixture() {
    let expected = load_expected("truth");
    assert!(expected.contains("2.0"), "{}", axes());
}

#[ignore]
#[test]
fn ignored_before_test() {
    let expected = load_expected("truth");
    assert!(expected.contains("2.0"), "{}", axes());
}

#[cfg(any())]
#[test]
fn compiled_out() {
    let expected = load_expected("truth");
    assert!(expected.contains("2.0"), "{}", axes());
}

/*
#[test]
fn in_block_comment() {
    let expected = load_expected("truth");
    assert!(expected.contains("2.0"), "{}", axes());
}
*/

const RAW: &str = r#"
#[test]
fn in_raw_string() {
    let expected = load_expected("truth");
    assert!(expected.contains("2.0"));
}
"#;

#[test]
fn never_reads_the_fixture() {
    assert!(!RAW.is_empty(), "{}", axes());
}

#[test]
fn never_builds_the_axes() {
    let expected = load_expected("truth");
    assert!(expected.contains("2.0"));
}

fn execute(input: &str) -> String {
    input.to_string()
}

#[test]
fn checked_execution_drops_builder() {
    let builder = axes();
    let plan = load_expected("truth");
    drop(builder);
    assert!(execute(&plan).contains("2.0"), "{}", axes());
}

#[test]
fn checked_execution_keeps_builder() {
    let builder = axes();
    let plan = load_expected("truth");
    assert!(execute(&plan).contains("2.0"), "{}", builder);
}

struct Executed { logical_plan: String }

#[test]
fn checked_execution_inspects_logical_plan() {
    let builder = axes();
    let input = load_expected("truth");
    drop(builder);
    let result = Executed { logical_plan: execute(&input) };
    assert!(result.logical_plan.contains("2.0"));
}

#[test]
fn checked_execution_logical_plan_decoys_do_not_count() {
    let builder = axes();
    let input = load_expected("truth");
    drop(builder);
    let executed = execute(&input);
    // `result.logical_plan` in prose does not inspect the retained output.
    assert!(executed.contains("2.0"), "result.logical_plan");
}
"""

ORPHAN_RS = r"""
#[test]
fn orphan_test() {
    let expected = std::fs::read_to_string("conformance/demo/truth/expected.json").unwrap();
    assert!(expected.contains("2.0"), "AverageEffectQuery Dag .graph( Frequentist RefuteSuite::None load_expected");
}
"""

REQUIRED_FEATURE_RS = r"""
#[cfg(feature = "unrelated-evidence")]
compile_error!("only the selected target's required feature may be enabled");

use antecedent_test_support::{conformance_dir as fixture_root, load_expected_json};

#[test]
fn required_feature_target_control() {
    let expected = load_expected_json(&fixture_root("demo/truth"));
    assert!(
        expected.contains("2.0"),
        "AverageEffectQuery Dag .graph( explicit Frequentist RefuteSuite::None"
    );
}
"""

PY = """
import json

import pytest

AXES = "AverageEffectQuery Dag .graph( Frequentist RefuteSuite::None"


def load(path="conformance/demo/truth/expected.json"):
    return json.loads(open(path).read())


def helper_not_a_test():
    assert load()["true_effect"] == 2.0, AXES


@pytest.mark.skip(reason="off")
def test_skipped():
    assert load()["true_effect"] == 2.0, AXES


@pytest.mark.skipif(
    True,
    reason="opt-in",
)
def test_skipped_multiline_decorator():
    assert load()["true_effect"] == 2.0, AXES


def test_skipped_in_body():
    pytest.skip("off")
    assert load()["true_effect"] == 2.0, AXES


@pytest.mark.parametrize("case", [])
def test_empty_parametrisation(case):
    assert load()["true_effect"] == 2.0, AXES


def run(plan):
    return plan


def test_checked_execution_discards_builder():
    builder = load
    plan = builder()
    builder = None
    assert run(plan)["true_effect"] == 2.0, AXES
"""

BASE_ROW = {
    "query": "AverageEffect",
    "graph_class": "Dag",
    "structure": "explicit",
    "inference": "Frequentist",
    "validation": "none",
    "evidence_kind": "internal_known_truth",
    "known_truth_fixture": "conformance/demo/truth",
    "evidence_test": "crates/selftest/tests/it.rs",
}


def build(tmp: Path) -> None:
    (tmp / "scripts").mkdir()
    for helper in (
        "test_evidence.py",
        "evidence_kinds.py",
        "run_evidence_rows.py",
        "_py_closure.py",
        "_rust_source.py",
        "_source_cache.py",
        "_repo.py",
    ):
        shutil.copy(HERE / helper, tmp / "scripts" / helper)
    (tmp / "Cargo.toml").write_text(
        '[workspace]\nresolver = "2"\nmembers = ["crates/selftest", "crates/antecedent-test-support"]\n',
        encoding="utf-8",
        newline="\n",
    )
    support = tmp / "crates" / "antecedent-test-support"
    (support / "src").mkdir(parents=True)
    (support / "Cargo.toml").write_text(
        '[package]\nname = "antecedent-test-support"\nversion = "0.0.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    (support / "src/lib.rs").write_text(
        "pub fn conformance_dir(relative: &str) -> String {\n"
        '    format!("{}/../../conformance/{relative}", env!("CARGO_MANIFEST_DIR"))\n}\n'
        "pub fn load_expected_json(dir: &str) -> String {\n"
        '    std::fs::read_to_string(format!("{dir}/expected.json")).unwrap()\n}\n',
        encoding="utf-8",
    )
    crate = tmp / "crates" / "selftest"
    (crate / "src").mkdir(parents=True)
    (crate / "tests" / "sub").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "selftest"\nversion = "0.0.0"\nedition = "2021"\npublish = false\n'
        "[features]\ntarget-evidence = []\nunrelated-evidence = []\n"
        '[[test]]\nname = "required_feature"\nrequired-features = ["target-evidence"]\n'
        '[dev-dependencies]\nantecedent-test-support = { path = "../antecedent-test-support" }\n',
        encoding="utf-8",
        newline="\n",
    )
    (crate / "src" / "lib.rs").write_text("pub fn noop() {}\n", encoding="utf-8", newline="\n")
    (crate / "tests" / "it.rs").write_text(IT_RS, encoding="utf-8", newline="\n")
    (crate / "tests" / "required_feature.rs").write_text(
        REQUIRED_FEATURE_RS, encoding="utf-8", newline="\n"
    )
    (crate / "tests" / "sub" / "orphan.rs").write_text(ORPHAN_RS, encoding="utf-8", newline="\n")
    (crate / "tests" / "path.rs").write_text(
        '#[path = "path_assertions.rs"] mod path_assertions;\n'
        '#[path = "disabled_functions.rs"] mod disabled_functions;\n'
        "#[test]\nfn path_module_fixture_helper() { path_assertions::consume_fixture(); }\n",
        encoding="utf-8",
        newline="\n",
    )
    (crate / "tests" / "path_assertions.rs").write_text(
        "use antecedent_test_support::{conformance_dir as fixture_root, load_expected_json};\n"
        "pub fn consume_fixture() {\n"
        '    let expected = load_expected_json(&fixture_root("demo/truth"));\n'
        '    assert!(expected.contains("2.0"), "{}", axes());\n'
        "}\n"
        'fn axes() -> &\'static str { "AverageEffectQuery Dag .graph( explicit Frequentist RefuteSuite::None" }\n',
        encoding="utf-8",
        newline="\n",
    )
    (crate / "tests" / "disabled_functions.rs").write_text(
        "use antecedent_test_support::{conformance_dir as fixture_root, load_expected_json};\n"
        "#[cfg(any())]\n"
        "pub fn check() {\n"
        '    let expected = load_expected_json(&fixture_root("demo/truth"));\n'
        '    assert!(expected.contains("2.0"));\n'
        "}\n"
        "pub fn check() {\n"
        '    let expected = load_expected_json(&fixture_root("demo/truth"));\n'
        "    let _ = expected;\n"
        "}\n",
        encoding="utf-8",
        newline="\n",
    )
    (crate / "tests" / "disabled_helper.rs").write_text(
        '#[path = "disabled_functions.rs"] mod helpers;\n'
        "#[test]\nfn disabled_helper_assertion_is_not_evidence() { helpers::check(); }\n",
        encoding="utf-8",
        newline="\n",
    )
    (crate / "tests" / "nested_function.rs").write_text(
        '#[path = "nested_function_helpers.rs"] mod helpers;\n'
        "#[test]\nfn nested_same_name_helper_is_not_evidence() { helpers::check(); }\n",
        encoding="utf-8",
        newline="\n",
    )
    (crate / "tests" / "nested_function_helpers.rs").write_text(
        "use antecedent_test_support::{conformance_dir as fixture_root, load_expected_json};\n"
        "pub fn check() {\n"
        '    let expected = load_expected_json(&fixture_root("demo/truth"));\n'
        "    let _ = expected;\n"
        "}\n"
        "mod unrelated {\n"
        "    use antecedent_test_support::{conformance_dir as fixture_root, load_expected_json};\n"
        "    pub fn check() {\n"
        '        let expected = load_expected_json(&fixture_root("demo/truth"));\n'
        '        assert!(expected.contains("2.0"));\n'
        "    }\n"
        "}\n",
        encoding="utf-8",
        newline="\n",
    )
    for target, declaration in (
        ("commented_path.rs", '// #[path = "sub/commented.rs"] mod orphan;\n'),
        ("string_path.rs", 'const DECOY: &str = "#[path = \\"sub/string.rs\\"] mod orphan;";\n'),
        ("disabled_path.rs", '#[cfg(any())]\n#[path = "sub/disabled.rs"] mod orphan;\n'),
    ):
        (crate / "tests" / target).write_text(
            declaration + "#[test]\nfn ordinary_test() { assert!(true); }\n",
            encoding="utf-8",
            newline="\n",
        )
    for orphan in ("commented.rs", "string.rs", "disabled.rs"):
        (crate / "tests" / "sub" / orphan).write_text(
            "#[test]\nfn orphan_is_test() { assert!(true); }\n",
            encoding="utf-8",
            newline="\n",
        )
    (crate / "tests" / "nested_scope.rs").write_text(
        "#[allow(non_camel_case_types)] struct checks;\n"
        "impl checks { fn close_abs_1e9(_: f64, _: f64, _: &str) {} }\n"
        'mod unrelated { #[path = "nested_checks.rs"] mod checks; }\n'
        'fn close() { checks::close_abs_1e9(1.0, 1.0, ""); }\n'
        "#[test]\nfn a_nested_module_cannot_attest_the_same_qualified_name() { close(); }\n",
        encoding="utf-8",
        newline="\n",
    )
    (crate / "tests" / "nested_checks.rs").write_text(
        "use antecedent_test_support::{conformance_dir as fixture_root, load_expected_json};\n"
        "pub fn close_abs_1e9(_: f64, _: f64, _: &str) {\n"
        '    let expected = load_expected_json(&fixture_root("demo/truth"));\n'
        '    assert!(expected.contains("2.0"));\n'
        "}\n",
        encoding="utf-8",
        newline="\n",
    )
    # A #[path] inside an inline module resolves relative to that module's
    # directory (tests/unrelated/ here), not beside nested_scope.rs.
    for directory in ("nested_scope", "nested_scope/unrelated", "unrelated"):
        nested_file = crate / "tests" / directory / "nested_checks.rs"
        nested_file.parent.mkdir(parents=True, exist_ok=True)
        nested_file.write_text(
            "use antecedent_test_support::{conformance_dir as fixture_root, load_expected_json};\n"
            "pub fn close_abs_1e9(_: f64, _: f64, _: &str) {\n"
            '    let expected = load_expected_json(&fixture_root("demo/truth"));\n'
            '    assert!(expected.contains("2.0"));\n'
            "}\n",
            encoding="utf-8",
            newline="\n",
        )
    fixture = tmp / "conformance" / "demo" / "truth"
    fixture.mkdir(parents=True)
    # A synthetic origin declaration for reader controls, never scientific evidence.
    # The behavioral-parity controls require an external origin; known-truth rows
    # can also consume it. Origin/field falsification has separate fault controls.
    (fixture / "expected.json").write_text(
        '{"true_effect": 2.0, "oracle": {"kind": "external_package", "project": "selftest_stub"}}\n',
        encoding="utf-8", newline="\n"
    )
    (tmp / "python" / "tests").mkdir(parents=True)
    (tmp / "python" / "tests" / "test_demo.py").write_text(PY, encoding="utf-8", newline="\n")


def toml_value(value) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {toml_value(v)}" for k, v in value.items()) + " }"
    return "[" + ", ".join(toml_value(v) for v in value) + "]"


def toml_row(row: dict) -> str:
    return "[[cell]]\n" + "".join(f"{k} = {toml_value(v)}\n" for k, v in row.items())


def check(tmp: Path, name: str, row: dict, expected: str | None) -> bool:
    rows = tmp / "rows.toml"
    rows.write_text(toml_row(row), encoding="utf-8", newline="\n")
    env = {**os.environ, "CARGO_TARGET_DIR": str(tmp / "target")}
    proc = subprocess.run(
        [sys.executable, str(tmp / "scripts" / "test_evidence.py"), "rows", str(rows)],
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    if expected is None:
        if proc.returncode != 0:
            print(f"SELF-TEST FAIL: {name}: the valid row was rejected:\n{out}")
            return False
        print(f"self-test ok: {name}: passes")
        return True
    if proc.returncode == 0:
        print(f"SELF-TEST FAIL: {name}: broken evidence passed")
        return False
    if expected not in out:
        print(f"SELF-TEST FAIL: {name}: rejected without {expected!r}:\n{out}")
        return False
    print(f"self-test ok: {name}: rejected ({expected})")
    return True


def check_required_feature_execution(tmp: Path) -> bool:
    """Exercise required-feature rejection, evidence collection and exact execution."""
    env = {**os.environ, "CARGO_TARGET_DIR": str(tmp / "target")}
    bare = subprocess.run(
        ["cargo", "test", "-q", "-p", "selftest", "--test", "required_feature", "--", "--list"],
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
    )
    if bare.returncode == 0 or "requires the features" not in (bare.stdout + bare.stderr):
        print("SELF-TEST FAIL: target without its declared required feature was not refused")
        return False

    rows = tmp / "required_feature_rows.toml"
    rows.write_text(
        "[[capabilities]]\n"
        'id = "required_feature_target"\n'
        'evidence_test = "crates/selftest/tests/required_feature.rs"\n'
        'evidence_assertion = "required_feature_target_control"\n',
        encoding="utf-8",
        newline="\n",
    )
    runner = subprocess.run(
        [
            sys.executable,
            str(tmp / "scripts" / "run_evidence_rows.py"),
            str(tmp),
            rows.name,
            str(tmp),
            "capabilities",
            "feature-required-selftest",
        ],
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
    )
    output = runner.stdout + runner.stderr
    if (
        runner.returncode != 0
        or "ok: required_feature_target (cited Rust test executed)" not in output
    ):
        print(f"SELF-TEST FAIL: required-feature target did not resolve and execute:\n{output}")
        return False
    print("self-test ok: target required feature is listed and exactly executed")
    return True


def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw).resolve()
        build(tmp)
        rs = lambda assertion, **kw: {**BASE_ROW, "evidence_assertion": assertion, **kw}  # noqa: E731
        py = lambda assertion: {  # noqa: E731
            **BASE_ROW,
            "evidence_test": "python/tests/test_demo.py",
            "evidence_assertion": assertion,
        }

        def checked_entry(estimator: str, assertion: str) -> dict:
            return {
                "estimator": estimator,
                "test": "crates/selftest/tests/it.rs",
                "assertion": assertion,
            }

        def checked(assertions: list[str]) -> dict:
            estimators = [f"e{i}" for i in range(len(assertions))]
            return {
                "estimators": estimators,
                "checked_execution": [
                    checked_entry(e, a) for e, a in zip(estimators, assertions, strict=True)
                ],
            }

        cases = [
            ("control", rs("good_consumes_fixture"), None),
            ("behavioral_fixture_control", rs("good_consumes_fixture", evidence_kind="behavioral_parity"), None),
            ("behavioral_reference_in_another_test_is_not_evidence",
             rs("shared_helpers_are_not_called", evidence_kind="behavioral_parity"),
             "never names conformance/demo/truth"),
            ("fixture_with_only_a_tautology", rs("fixture_with_only_a_tautology"),
             "every assertion is a constant or self-comparison"),
            (
                "path_module_helper_control",
                rs("path_module_fixture_helper", evidence_test="crates/selftest/tests/path.rs"),
                None,
            ),
            (
                "target_required_feature_evidence_control",
                rs(
                    "required_feature_target_control",
                    evidence_test="crates/selftest/tests/required_feature.rs",
                ),
                None,
            ),
            (
                "disabled_qualified_helper_is_not_evidence",
                rs(
                    "disabled_helper_assertion_is_not_evidence",
                    evidence_test="crates/selftest/tests/disabled_helper.rs",
                ),
                "asserts nothing",
            ),
            (
                "nested_same_name_qualified_helper_is_not_evidence",
                rs(
                    "nested_same_name_helper_is_not_evidence",
                    evidence_test="crates/selftest/tests/nested_function.rs",
                ),
                "asserts nothing",
            ),
            (
                "nested_unrelated_qualified_helper_is_not_evidence",
                rs(
                    "a_nested_module_cannot_attest_the_same_qualified_name",
                    evidence_test="crates/selftest/tests/nested_scope.rs",
                ),
                "never names conformance/demo/truth",
            ),
            ("shared_helper_control", rs("shared_helpers_consume_fixture"), None),
            (
                "uncalled_shared_helpers",
                rs("shared_helpers_are_not_called"),
                "never names conformance/demo/truth",
            ),
            (
                "relative_path_without_root",
                rs("relative_string_without_conformance_root"),
                "never names conformance/demo/truth",
            ),
            ("ignore_before_test", rs("ignored_before_test"), "is #[ignore]d"),
            ("cfg_compiled_out", rs("compiled_out"), "compiled only under #[cfg(any())]"),
            ("block_comment", rs("in_block_comment"), "no fn in_block_comment outside comments"),
            ("raw_string", rs("in_raw_string"), "no fn in_raw_string outside comments"),
            (
                "file_never_compiled",
                {**rs("orphan_test"), "evidence_test": "crates/selftest/tests/sub/orphan.rs"},
                "is not compiled into any cargo target",
            ),
            (
                "commented_path_attribute_does_not_compile_helper",
                {
                    **rs("orphan_is_test"),
                    "evidence_test": "crates/selftest/tests/sub/commented.rs",
                },
                "is not compiled into any cargo target",
            ),
            (
                "string_path_attribute_does_not_compile_helper",
                {
                    **rs("orphan_is_test"),
                    "evidence_test": "crates/selftest/tests/sub/string.rs",
                },
                "is not compiled into any cargo target",
            ),
            (
                "disabled_path_module_does_not_compile_helper",
                {
                    **rs("orphan_is_test"),
                    "evidence_test": "crates/selftest/tests/sub/disabled.rs",
                },
                "is not compiled into any cargo target",
            ),
            ("python_helper_not_a_test", py("helper_not_a_test"), "is not a collected pytest name"),
            ("python_skipped", py("test_skipped"), "carries a skip/xfail marker"),
            (
                "python_skipped_multiline_decorator",
                py("test_skipped_multiline_decorator"),
                "carries a skip/xfail marker",
            ),
            (
                "python_skipped_in_body",
                py("test_skipped_in_body"),
                "calls pytest.skip() unconditionally",
            ),
            (
                "python_empty_parametrisation",
                py("test_empty_parametrisation"),
                "parametrised over an empty list",
            ),
            (
                "fixture_consumed_only_elsewhere",
                rs("never_reads_the_fixture"),
                "never names conformance/demo/truth",
            ),
            (
                "axes_never_built",
                rs("never_builds_the_axes"),
                "never exercises query 'AverageEffect'",
            ),
            (
                "row_axis_not_built",
                rs("good_consumes_fixture", inference="Bayesian"),
                "never exercises inference 'Bayesian'",
            ),
            ("missing_test", rs("no_such_test"), "no fn no_such_test"),
            # checked_execution: one entry per licensed estimator, each citing an
            # executing test that discards its builder, executes, and inspects the plan.
            (
                "checked_execution_control",
                rs("good_consumes_fixture", **checked(["checked_execution_drops_builder"])),
                None,
            ),
            (
                "checked_execution_python_control",
                rs(
                    "good_consumes_fixture",
                    estimators=["a"],
                    checked_execution=[
                        {
                            "estimator": "a",
                            "test": "python/tests/test_demo.py",
                            "assertion": "test_checked_execution_discards_builder",
                        }
                    ],
                ),
                None,
            ),
            (
                "checked_execution_keeps_builder",
                rs("good_consumes_fixture", **checked(["checked_execution_keeps_builder"])),
                "does not explicitly discard its builder",
            ),
            (
                "checked_execution_logical_plan_control",
                rs("good_consumes_fixture", **checked(["checked_execution_inspects_logical_plan"])),
                None,
            ),
            (
                "checked_execution_logical_plan_decoys",
                rs(
                    "good_consumes_fixture",
                    **checked(["checked_execution_logical_plan_decoys_do_not_count"]),
                ),
                "does not inspect a retained program or plan",
            ),
            (
                "checked_execution_not_a_test",
                rs("good_consumes_fixture", **checked(["execute"])),
                "is not an executing test",
            ),
            (
                "checked_execution_missing_estimator",
                rs(
                    "good_consumes_fixture",
                    estimators=["a", "b"],
                    checked_execution=[checked_entry("a", "checked_execution_drops_builder")],
                ),
                "!= licensed estimators",
            ),
            (
                "checked_execution_duplicate_estimator",
                rs(
                    "good_consumes_fixture",
                    estimators=["a"],
                    checked_execution=[
                        checked_entry("a", "checked_execution_drops_builder"),
                        checked_entry("a", "checked_execution_drops_builder"),
                    ],
                ),
                "names an estimator twice",
            ),
            (
                "checked_execution_absent",
                rs("good_consumes_fixture", estimators=["a"], checked_execution=[]),
                "checked_execution is required",
            ),
        ]
        results = [check(tmp, name, row, expected) for name, row, expected in cases]
        results.append(check_required_feature_execution(tmp))
    if not all(results):
        print(f"support-matrix evidence self-test: {results.count(False)} case(s) failed")
        return 1
    print(f"support-matrix evidence self-test: ok ({len(results)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
