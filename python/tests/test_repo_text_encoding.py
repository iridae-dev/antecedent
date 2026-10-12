"""Guard: tests must not decode repository text with the Windows locale."""

from __future__ import annotations

import ast
from pathlib import Path

from _repo_text import REPO_ROOT, load_json, read_text

_TESTS = Path(__file__).resolve().parent
_ALLOWED_BARE_READ = {"_repo_text.py"}
_HELPER_FUNCS = {"read_text", "write_text", "load_toml"}


def test_locale_decode_cannot_round_trip_times_sign() -> None:
    raw = (
        REPO_ROOT / "conformance" / "bayesian" / "temporal_class_prior_transfer" / "expected.json"
    ).read_bytes()
    assert "\u00d7".encode() in raw
    assert "\u00d7" not in raw.decode("cp1252")


def test_class_prior_transfer_pin_preserves_times_sign() -> None:
    pin = load_json(
        REPO_ROOT / "conformance" / "bayesian" / "temporal_class_prior_transfer" / "expected.json"
    )
    cell = pin["source_cells"]["same_design_pulse"]
    assert "\u00d7" in cell
    assert "\ufffd" not in cell
    assert cell == "PulseEffect \u00d7 TemporalCpdag \u00d7 explicit \u00d7 Bayesian \u00d7 none"


def test_test_modules_do_not_read_repo_text_with_locale_encoding() -> None:
    offenders: list[str] = []
    for path in sorted(_TESTS.rglob("*.py")):
        if path.name in _ALLOWED_BARE_READ:
            continue
        tree = ast.parse(read_text(path), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            keywords = {kw.arg for kw in node.keywords if kw.arg}
            if "encoding" in keywords:
                continue
            # `read_text(path)` is the UTF-8 helper. `path.read_text()` is locale.
            if isinstance(func, ast.Name) and func.id == "read_text":
                continue
            if isinstance(func, ast.Attribute) and func.attr == "read_text":
                offenders.append(f"{path.relative_to(_TESTS)}:{node.lineno}")
                continue
            if isinstance(func, ast.Name) and func.id == "open":
                if _is_binary_open(node):
                    continue
                offenders.append(f"{path.relative_to(_TESTS)}:{node.lineno}")
    assert not offenders, (
        "read repository text as UTF-8 (`_repo_text.read_text` / `load_json`, "
        "or pass encoding='utf-8'). Locale decode breaks U+00D7 on Windows: " + ", ".join(offenders)
    )


def _is_binary_open(node: ast.Call) -> bool:
    """``open(path, "rb")`` puts the mode in args[1]; ``path.open("rb")`` puts it in args[0]."""
    path_open = isinstance(node.func, ast.Attribute)
    indexes = (0,) if path_open else (1,)
    for index in indexes:
        if len(node.args) <= index:
            continue
        arg = node.args[index]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "b" in arg.value:
            return True
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            value = kw.value.value
            return isinstance(value, str) and "b" in value
    return False


def _helper_module_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "_repo",
            "scripts._repo",
            "_repo_text",
        }:
            for alias in node.names:
                if alias.name == "*" or alias.name in _HELPER_FUNCS:
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"_repo", "scripts._repo", "_repo_text"}:
                    bound = alias.asname or alias.name.split(".")[-1]
                    names.add(bound)
    return names


def _text_io_offenders(path: Path, root: Path) -> list[str]:
    tree = ast.parse(read_text(path), filename=str(path))
    helpers = _helper_module_names(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        keywords = {kw.arg for kw in node.keywords if kw.arg}
        if "encoding" in keywords:
            continue
        if isinstance(func, ast.Name) and func.id in _HELPER_FUNCS | helpers:
            continue
        if isinstance(func, ast.Attribute) and func.attr in _HELPER_FUNCS:
            receiver = func.value
            if isinstance(receiver, ast.Name) and receiver.id in helpers:
                continue
            found.append(f"{path.relative_to(root)}:{node.lineno}")
            continue
        if (
            isinstance(func, ast.Name)
            and func.id == "open"
            or (isinstance(func, ast.Attribute) and func.attr == "open")
        ):
            if _is_binary_open(node):
                continue
            found.append(f"{path.relative_to(root)}:{node.lineno}")
    return found


def test_scripts_and_conformance_text_io_declares_utf8() -> None:
    offenders: list[str] = []
    for base in ("scripts", "conformance"):
        for path in sorted((REPO_ROOT / base).rglob("*.py")):
            offenders.extend(_text_io_offenders(path, REPO_ROOT))
    assert not offenders, (
        "read repository text as UTF-8 (`scripts/_repo.py` or encoding='utf-8'). "
        "Locale decode breaks U+00D7 on Windows: " + ", ".join(offenders)
    )
