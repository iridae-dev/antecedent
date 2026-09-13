"""Guard: tests must not decode repository text with the Windows locale."""

from __future__ import annotations

import ast
from pathlib import Path

from _repo_text import REPO_ROOT, load_json, read_text

_TESTS = Path(__file__).resolve().parent
_ALLOWED_BARE_READ = {"_repo_text.py"}


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
    for arg in node.args[1:2]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "b" in arg.value:
            return True
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            value = kw.value.value
            return isinstance(value, str) and "b" in value
    return False
