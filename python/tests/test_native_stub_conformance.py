"""`_native.pyi` must describe exactly the surface `antecedent._native` exposes.

The stub is hand-maintained: no generator can infer its `Literal`s, keyword-only
markers, or docstrings, and running one would need `maturin develop` inside the
wheel job. So the stub drifts silently whenever a `#[pyfunction]` or `#[pyclass]`
is added, renamed, or removed on the Rust side.

The repo's mypy gate would catch some of that, but it is local-only
(`scripts/gate_python_lint.sh`). This test lives in the pytest job that already
runs in wheel CI, so it is the gate that actually fires on a release build.

It compares top-level names only — signatures are mypy's job, not this test's.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("antecedent")
from antecedent import _native

_STUB = Path(_native.__file__).with_name("_native.pyi")

# Stub-only names: type aliases and re-spellings that exist to make the stub
# readable and have no runtime counterpart. Anything added here must be a name
# the extension genuinely does not define.
_STUB_ONLY = {
    "CiArg",  # `str | Callable[..., Any] | None`, used across discovery signatures
    "TemporalAnalysisResult",  # readable alias for the temporal `AnalysisResult`
}


def _stub_toplevel_names() -> set[str]:
    tree = ast.parse(_STUB.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


# Dunders that are real API rather than module machinery.
_PUBLIC_DUNDERS = {"__version__"}


def _extension_public_names() -> set[str]:
    return {n for n in dir(_native) if not n.startswith("_") or n in _PUBLIC_DUNDERS}


def test_stub_file_exists():
    assert _STUB.is_file(), f"expected a type stub beside the extension at {_STUB}"


def test_every_extension_name_is_declared_in_the_stub():
    missing = sorted(_extension_public_names() - _stub_toplevel_names())
    assert not missing, (
        "antecedent._native exposes names the stub does not declare "
        f"(add them to _native.pyi): {missing}"
    )


def test_stub_declares_nothing_the_extension_lacks():
    extra = sorted(_stub_toplevel_names() - _extension_public_names() - _STUB_ONLY)
    assert not extra, (
        "_native.pyi declares names the extension does not define — stale stub "
        f"entries, or additions to _STUB_ONLY if genuinely stub-only: {extra}"
    )


def test_stub_only_allowlist_stays_justified():
    """Every allowlisted name must really be absent from the extension."""
    not_actually_stub_only = sorted(_STUB_ONLY & _extension_public_names())
    assert not not_actually_stub_only, (
        "these are declared stub-only but the extension defines them; "
        f"drop them from _STUB_ONLY: {not_actually_stub_only}"
    )
