#!/usr/bin/env python3
"""Adversarial cases for the one Rust lexer and the one Python closure walk."""

from __future__ import annotations

import ast
from pathlib import Path

import _py_closure
import _rust_source as rust


def test_lint_attributes_accept_reasons_and_multiple_lints() -> None:
    assert rust.has_lint_attribute(
        '#![allow(\n unused_imports, unsafe_code, reason = "scoped FFI (owner)"\n)]',
        "allow",
        "unsafe_code",
        inner=True,
    )
    assert rust.has_lint_attribute("#[allow(unsafe_code)]", "allow", "unsafe_code", inner=False)
    assert rust.has_lint_attribute(
        "#![forbid ( unsafe_code )]", "forbid", "unsafe_code", inner=True
    )


def test_lint_attributes_reject_inactive_or_different_policy() -> None:
    for text in (
        "// #![forbid(unsafe_code)]",
        "/* #![forbid(unsafe_code)] */",
        'const S: &str = r#"#![forbid(unsafe_code)]"#;',
        '#![allow(unused_imports, reason = "forbid(unsafe_code)")]',
        "#![forbid(unsafe_code_extra)]",
        "#[forbid(unsafe_code)]",
        '#![cfg_attr(feature = "strict", forbid(unsafe_code))]',
    ):
        assert not rust.has_lint_attribute(text, "forbid", "unsafe_code", inner=True), text


def test_lifetime_is_not_a_character() -> None:
    text = "fn f<'a>(x: &'a u8) -> &'a u8 { x }\n"
    parsed = rust.lex(text)
    assert "<'a>" in parsed.code
    assert len(parsed.skeleton) == len(text)
    assert parsed.skeleton == text


def test_character_literal_is_blanked() -> None:
    text = "let c = 'x';\n"
    parsed = rust.lex(text)
    assert "' '" in parsed.code
    assert "'x'" not in parsed.skeleton
    assert "'x'" in parsed.kept
    assert parsed.kept_skeleton[parsed.kept_skeleton.find("'") + 1] == " "


def test_byte_raw_string_and_nested_braces() -> None:
    text = 'let s = br#"a{b}"#;\n'
    parsed = rust.lex(text)
    assert 'br#"a{b}"#' not in parsed.code
    assert "{" not in parsed.skeleton
    assert 'br#"a{b}"#' in parsed.kept
    assert "{" not in parsed.kept_skeleton


def test_escaped_string_keeps_newlines_in_the_code_view() -> None:
    text = 'let s = "a\\n\\"b";\n'
    parsed = rust.lex(text)
    assert parsed.code.endswith('"";\n')
    assert "\\n" not in parsed.code
    assert len(parsed.skeleton) == len(text)


def test_doc_comment_span_skips_a_fourth_slash() -> None:
    text = "/// real\n//// not a doc\nfn f() {}\n"
    parsed = rust.lex(text)
    assert parsed.line_docs == ((0, len("/// real")),)
    assert "real" not in parsed.code
    assert "fn f()" in parsed.code


def test_nested_block_comment() -> None:
    text = "let x = 1; /* outer /* inner */ still */ let y = 2;\n"
    parsed = rust.lex(text)
    assert "inner" not in parsed.code
    assert "still" not in parsed.code
    assert "let y = 2;" in parsed.code
    assert parsed.skeleton.count("\n") == text.count("\n")


def test_identifier_ending_in_r_is_not_a_raw_string() -> None:
    text = 'for"loop"\n'
    parsed = rust.lex(text)
    assert parsed.code.startswith("for")
    assert "loop" not in parsed.code


def test_extern_c_delimiters_stay_in_the_kept_skeleton() -> None:
    text = 'extern "C" fn f() {}\n'
    parsed = rust.lex(text)
    assert '"C"' not in parsed.skeleton
    assert '" "' in parsed.kept_skeleton or '"' in parsed.kept_skeleton
    assert "C" not in parsed.kept_skeleton.replace("extern", "")
    assert parsed.kept_skeleton.strip().startswith('extern "')


def test_python_closure_follows_calls_not_strings() -> None:
    import tempfile

    path = Path(tempfile.mkdtemp()) / "corpus.py"
    path.write_text(
        "def helper():\n"
        "    return 'conformance/estimate/aipw'\n"
        "\n"
        "def cited():\n"
        "    return helper()\n"
        "\n"
        "def ignored():\n"
        "    return 'not this'\n"
        "\n"
        "def mentions():\n"
        "    return 'ignored'\n",
        encoding="utf-8",
    )
    try:
        text = _py_closure.closure_source(path, "cited")
        assert "helper" in text
        assert "conformance/estimate/aipw" in text
        assert "not this" not in text
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = _py_closure.closure_nodes(tree, "mentions")
        assert [type(n).__name__ for n in nodes] == ["FunctionDef"]
        assert nodes[0].name == "mentions"
    finally:
        path.unlink(missing_ok=True)
        path.parent.rmdir()


def test_forwarded_ignored_test_macros_bind_only_actual_arguments() -> None:
    from tempfile import TemporaryDirectory

    from promotion_source import ignored_test_literals

    source = r"""
const SHARED: &[&str] = &["shared_rho_0.0", "shared_rho_0.5"];
const UNUSED: &[&str] = &["separate_streams", "wrong_coupling"];
macro_rules! leaf {
    ($labels:expr, $name:ident, $dgp:literal) => {
        #[test]
        #[ignore = "coverage"]
        fn $name() { run(stringify!($name), $labels, $dgp); }
    };
}
macro_rules! forward {
    ($name:ident, $unused:expr, $labels:expr) => {
        leaf!($labels, $name, "actual_dgp");
    };
}
macro_rules! outer {
    ($name:ident, $labels:expr) => { forward!($name, UNUSED, $labels); };
}
outer!(current, SHARED);
"""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "forwarded.rs"
        path.write_text(source, encoding="utf-8")
        emitted = ignored_test_literals(path)
    assert set(emitted) == {"current"}
    assert set(emitted["current"]) == {"current", "actual_dgp", "shared_rho_0.0", "shared_rho_0.5"}


def test_macro_cycles_and_uninvoked_definitions_emit_no_tests() -> None:
    from tempfile import TemporaryDirectory

    from promotion_source import ignored_test_literals

    source = r"""
macro_rules! unused {
    ($name:ident) => { #[test] #[ignore] fn $name() { run("foreign_label"); } };
}
macro_rules! first { ($name:ident) => { second!($name); }; }
macro_rules! second { ($name:ident) => { first!($name); }; }
first!(not_a_generated_test);
"""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cycle.rs"
        path.write_text(source, encoding="utf-8")
        assert ignored_test_literals(path) == {}


if __name__ == "__main__":
    test_lint_attributes_accept_reasons_and_multiple_lints()
    test_lint_attributes_reject_inactive_or_different_policy()
    test_lifetime_is_not_a_character()
    test_character_literal_is_blanked()
    test_byte_raw_string_and_nested_braces()
    test_escaped_string_keeps_newlines_in_the_code_view()
    test_doc_comment_span_skips_a_fourth_slash()
    test_nested_block_comment()
    test_identifier_ending_in_r_is_not_a_raw_string()
    test_extern_c_delimiters_stay_in_the_kept_skeleton()
    test_python_closure_follows_calls_not_strings()
    test_forwarded_ignored_test_macros_bind_only_actual_arguments()
    test_macro_cycles_and_uninvoked_definitions_emit_no_tests()
    print("rust source lexer: ok")
