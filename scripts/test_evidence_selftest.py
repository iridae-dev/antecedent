#!/usr/bin/env python3
"""Broken-evidence self-test for scripts/test_evidence.py (gate_support_matrix.sh --self-test).

Builds a throwaway cargo workspace with one conformance fixture and test files
that hide a cited test in every way a text scan fails open on: `#[ignore]`
before `#[test]`, a `#[cfg(...)]` that compiles it out, a block comment, a raw
string, a file no `mod` declares, a Python helper that is not a collected test,
a skipped pytest, a test that never consumes the fixture another test consumes,
and a test that never builds the row's axis values. Each broken row, alone, must
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
"""

ORPHAN_RS = r"""
#[test]
fn orphan_test() {
    let expected = std::fs::read_to_string("conformance/demo/truth/expected.json").unwrap();
    assert!(expected.contains("2.0"), "AverageEffectQuery Dag .graph( Frequentist RefuteSuite::None load_expected");
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
    shutil.copy(HERE / "test_evidence.py", tmp / "scripts" / "test_evidence.py")
    (tmp / "Cargo.toml").write_text('[workspace]\nresolver = "2"\nmembers = ["crates/selftest"]\n')
    crate = tmp / "crates" / "selftest"
    (crate / "src").mkdir(parents=True)
    (crate / "tests" / "sub").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "selftest"\nversion = "0.0.0"\nedition = "2021"\npublish = false\n'
    )
    (crate / "src" / "lib.rs").write_text("pub fn noop() {}\n")
    (crate / "tests" / "it.rs").write_text(IT_RS)
    (crate / "tests" / "sub" / "orphan.rs").write_text(ORPHAN_RS)
    fixture = tmp / "conformance" / "demo" / "truth"
    fixture.mkdir(parents=True)
    (fixture / "expected.json").write_text('{"true_effect": 2.0}\n')
    (tmp / "python" / "tests").mkdir(parents=True)
    (tmp / "python" / "tests" / "test_demo.py").write_text(PY)


def toml_row(row: dict) -> str:
    return "[[cell]]\n" + "".join(f'{k} = "{v}"\n' for k, v in row.items())


def check(tmp: Path, name: str, row: dict, expected: str | None) -> bool:
    rows = tmp / "rows.toml"
    rows.write_text(toml_row(row))
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
        cases = [
            ("control", rs("good_consumes_fixture"), None),
            ("ignore_before_test", rs("ignored_before_test"), "is #[ignore]d"),
            ("cfg_compiled_out", rs("compiled_out"), "compiled only under #[cfg(any())]"),
            ("block_comment", rs("in_block_comment"), "no fn in_block_comment outside comments"),
            ("raw_string", rs("in_raw_string"), "no fn in_raw_string outside comments"),
            (
                "file_never_compiled",
                {**rs("orphan_test"), "evidence_test": "crates/selftest/tests/sub/orphan.rs"},
                "is not compiled into any cargo target",
            ),
            ("python_helper_not_a_test", py("helper_not_a_test"), "is not a collected pytest name"),
            ("python_skipped", py("test_skipped"), "carries a skip/xfail marker"),
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
        ]
        results = [check(tmp, name, row, expected) for name, row, expected in cases]
    if not all(results):
        print(f"support-matrix evidence self-test: {results.count(False)} case(s) failed")
        return 1
    print(f"support-matrix evidence self-test: ok ({len(results)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
