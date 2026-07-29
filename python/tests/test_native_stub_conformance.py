"""`_native.pyi` must describe exactly the surface `antecedent._native` exposes.

The stub is hand-maintained: no generator can infer its `Literal`s, keyword-only
markers, or docstrings, and running one would need `maturin develop` inside the
wheel job. So the stub drifts silently whenever a `#[pyfunction]` or `#[pyclass]`
is added, renamed, or removed on the Rust side — or whenever one's parameters change
shape without the stub following along (a stub entry can keep declaring a parameter
long after Rust drops it, or omit one Rust added).

The repo's mypy gate would catch some of that, but it is local-only
(`scripts/gate_python_lint.sh`). This test lives in the pytest job that already
runs in wheel CI, so it is the gate that actually fires on a release build.

Two tiers: top-level name presence (below), and — for every stub function whose real
signature this process can introspect — full per-parameter comparison (name, keyword-only
position, and default value) against the compiled extension's own `inspect.signature` /
`__text_signature__`. A function that can't be introspected at all is skipped individually
(via `_SIGNATURE_UNAVAILABLE`, with a reason), never silently dropped from the sweep.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Per-parameter signature comparison (TODO #15).
#
# The tests above only ever compared *that* a name exists on both sides. That is exactly
# the gap that let `_native.pyi` declare `prior_mapping`/`composed_prior` on every temporal
# entry point (`analyze`, `analyze_temporal_pag`, `analyze_events`, `analyze_panel`,
# `analyze_panel_discover`, `analyze_temporal_discover`, `analyze_temporal_mediation`) —
# parameters their real Rust `#[pyo3(signature = ...)]` blocks never had — while several of
# them simultaneously omitted `refute`/`validators`, which those same Rust signatures did
# have. `test_every_extension_name_is_declared_in_the_stub` passed the whole time: `analyze`
# was declared, `analyze` existed: the name-only check has no way to see a parameter list.
# ---------------------------------------------------------------------------

_MISSING = object()  # sentinel: "this parameter has no default" (i.e. required)

# Functions this test cannot introspect a real signature for (neither `inspect.signature`
# nor a manual `__text_signature__` parse succeeds), keyed to a reason. Every name here
# skips the per-parameter comparison entirely, so adding one is a deliberate weakening of
# the gate for that one function — it must be individually justified, and re-checked
# whenever pyo3 or the build changes, rather than used to quiet a real mismatch.
_SIGNATURE_UNAVAILABLE: dict[str, str] = {}

# Per-parameter exemptions from the default-value comparison, keyed by `(function_name,
# parameter_name)` with a required human-readable reason. Unlike `_SIGNATURE_UNAVAILABLE`
# (which skips a whole function), an entry here only excuses the *default value* of one
# named parameter — its name and keyword-only-ness are still compared, and every other
# parameter of the function is still compared in full. Use this only when the native
# default genuinely cannot be expressed as a literal (e.g. a named Rust constant that
# PyO3's auto `__text_signature__` renders as `...`/`Ellipsis` rather than the constant's
# value), never to quiet a real stub/extension drift.
_PARAM_DEFAULT_EXEMPT: dict[tuple[str, str], str] = {
    ("antecedent_state_append", "cache_bytes"): (
        "the Rust default is the named constant `DEFAULT_CACHE_BYTES` "
        "(`python/src/state_api.rs`), not a literal token in the `#[pyo3(signature = ...)]` "
        "macro, so PyO3's auto text_signature rendering reports it as `...` (`Ellipsis`) "
        "rather than `1_048_576`. The stub's literal default is the more useful "
        "documentation here and is kept deliberately; see also `CausalState.__init__`'s "
        "`cache_bytes` (same constant), which is out of this test's scope because it is a "
        "class method, not a top-level function."
    ),
}

# The functions Defect C fixed. The signature sweep below must actually compare these —
# if introspection ever silently stopped covering them (e.g. by falling into
# `_SIGNATURE_UNAVAILABLE` without anyone noticing), this test would stop being able to
# catch a regression of exactly the bug it was written for.
_MUST_BE_CHECKED = (
    "analyze",
    "analyze_temporal_pag",
    "analyze_events",
    "analyze_panel",
    "analyze_panel_discover",
    "analyze_temporal_discover",
    "analyze_temporal_mediation",
)


def _stub_functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(_STUB.read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _stub_param_specs(fn: ast.FunctionDef) -> list[tuple[str, bool, Any]]:
    """`(name, keyword_only, default)` for every real parameter of a stub function, in
    declared order. `default` is the `_MISSING` sentinel for a required parameter.

    Every default in this hand-written stub is a literal (`None` / `True` / `False` / a
    number / a string) — never a name reference or expression — so `ast.literal_eval`
    always succeeds; there is deliberately no fallback that would mask a non-literal
    default silently comparing unequal.
    """
    args = fn.args
    positional = [*args.posonlyargs, *args.args]
    pad: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    pos_defaults = pad + list(args.defaults)
    out: list[tuple[str, bool, Any]] = []
    for arg, default_node in zip(positional, pos_defaults, strict=True):
        default = _MISSING if default_node is None else ast.literal_eval(default_node)
        out.append((arg.arg, False, default))
    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        default = _MISSING if default_node is None else ast.literal_eval(default_node)
        out.append((arg.arg, True, default))
    return out


def _native_signature(obj: object) -> inspect.Signature | None:
    """`inspect.signature`, falling back to a manual `__text_signature__` parse for PyO3
    builtins the primary path can't handle. Returns `None` (never raises) when neither
    works, so one function's introspection limits can't crash the whole sweep.
    """
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError):
        pass
    text_sig = getattr(obj, "__text_signature__", None)
    if not text_sig:
        return None
    try:
        # `inspect.signature` already tries this internally for most builtins; kept as an
        # explicit fallback for PyO3 objects where the primary path raises before reaching
        # it. Private API, but stable since Python 3.4 and this call is best-effort only —
        # a failure here degrades to "unavailable", not a crash.
        return inspect._signature_fromstr(  # type: ignore[attr-defined]
            inspect.Signature, obj, text_sig
        )
    except Exception:  # noqa: BLE001 - best-effort fallback; "unavailable" is a valid outcome
        return None


def _native_param_specs(sig: inspect.Signature) -> list[tuple[str, bool, Any]]:
    out: list[tuple[str, bool, Any]] = []
    for p in sig.parameters.values():
        keyword_only = p.kind == inspect.Parameter.KEYWORD_ONLY
        default = _MISSING if p.default is inspect.Parameter.empty else p.default
        out.append((p.name, keyword_only, default))
    return out


def test_stub_signatures_match_the_extension():
    stub_fns = _stub_functions()
    mismatches: list[str] = []
    unavailable: list[str] = []
    checked: set[str] = set()

    for name in sorted(_extension_public_names() & stub_fns.keys()):
        obj = getattr(_native, name)
        if not inspect.isroutine(obj):
            continue  # classes: not this test's job, and not what Defect C touched

        sig = _native_signature(obj)
        if sig is None:
            reason = _SIGNATURE_UNAVAILABLE.get(name)
            if reason is None:
                mismatches.append(
                    f"{name}: no introspectable signature, and not listed in "
                    "_SIGNATURE_UNAVAILABLE with a reason — add it there if this is "
                    "genuinely expected, otherwise fix extraction"
                )
            else:
                unavailable.append(f"{name} (skipped: {reason})")
            continue

        checked.add(name)
        stub_params = _stub_param_specs(stub_fns[name])
        native_params = _native_param_specs(sig)

        exempt_params = {p for fn, p in _PARAM_DEFAULT_EXEMPT if fn == name}
        if exempt_params:
            stub_defaults_by_name = {pname: default for pname, _, default in stub_params}
            adjusted: list[tuple[str, bool, Any]] = []
            for pname, keyword_only, ndefault in native_params:
                if pname in exempt_params:
                    reason = _PARAM_DEFAULT_EXEMPT[(name, pname)]
                    assert ndefault is Ellipsis, (
                        f"{name}.{pname}: _PARAM_DEFAULT_EXEMPT claims native reports an "
                        f"unrenderable (`Ellipsis`) default, but native actually reports "
                        f"{ndefault!r} — the exemption is stale and hiding this from "
                        f"comparison; drop it and let the real default be compared "
                        f"(reason on file: {reason})"
                    )
                    # Only the default is excused; name and keyword-only-ness above still
                    # came from the real native signature and remain fully compared.
                    exempted_default = stub_defaults_by_name.get(pname, ndefault)
                    adjusted.append((pname, keyword_only, exempted_default))
                else:
                    adjusted.append((pname, keyword_only, ndefault))
            native_params = adjusted

        if stub_params != native_params:
            mismatches.append(f"{name}:\n  stub:   {stub_params!r}\n  native: {native_params!r}")

    missing_coverage = [n for n in _MUST_BE_CHECKED if n not in checked]
    assert not missing_coverage, (
        "signature comparison did not run for temporal entry points it must cover "
        f"(Defect C regression coverage would be silently lost): {missing_coverage}; "
        f"unavailable={unavailable}"
    )
    assert not mismatches, "stub/extension parameter-list drift:\n" + "\n".join(mismatches)
