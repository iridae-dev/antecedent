"""Response-family support-matrix enforcement and structure-source threading.

Covers two branch-review defects fixed together:

* The native `analyze_response` (python/src/response_api.rs) now consults the
  generated support matrix itself (`antecedent::support::refuse_if_not_applicable`)
  instead of being enforced only by hand-typed Python literals that could drift
  from `parity/support_closed.toml`. `test_support_closed.py` already pins the
  resulting error text end-to-end and must keep passing unchanged; this file adds
  a drift guard for the one Python literal that intentionally remains (it doubles
  as a routing gate, not just a refusal message).
* `AcceptedGraph.analyze(..., query=ResponseCurve(...))` now threads
  `structure_accepted` down to the native call's `accepted=` parameter, so the
  response family is not silently misclassified as `explicit` structure the way
  `prepare`/`prepare_response` already record `accepted` correctly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUPPORT_CLOSED_TOML = _REPO_ROOT / "parity" / "support_closed.toml"


def _curve_table(n: int = 60, seed: int = 7):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.2, 1.8, n)
    y = 2.0 * t + rng.normal(scale=0.1, size=n)
    return {"t": t, "y": y}


_DATA = _curve_table()
_EDGES = [("t", "y")]
_DAG = antecedent.Dag.from_edges(["t", "y"], _EDGES)
_ACCEPTED = antecedent.AcceptedGraph.from_graph(_DAG, algorithm_id="hand")
_CURVE = antecedent.ResponseCurve("t", "y", grid=[0.5, 1.0, 1.5])


def test_response_curve_pag_literal_matches_support_closed_toml():
    """`_analyze.py`'s remaining hand-rolled ResponseCurve/Pag refusal must not drift.

    This check has to stay in Python (see the comment in `handle_response`): it is
    the routing gate deciding whether the call reaches `_analyze_response` at all
    versus `_analyze_response_pag`, so it fires before the native support-matrix
    consultation ever runs. Since it can't be derived from the TOML across the
    language boundary, pin it here so a TOML edit fails this test loudly instead
    of silently drifting.
    """
    rules = tomllib.loads(_SUPPORT_CLOSED_TOML.read_text())["closed"]
    matches = [
        rule
        for rule in rules
        if rule.get("queries") == ["ResponseCurve"]
        and set(rule.get("graph_classes", [])) == {"Pag", "Cpdag", "Admg"}
    ]
    assert len(matches) == 1, matches
    reason = matches[0]["reason"]
    assert reason == (
        "ResponseCurve is licensed only on a static Dag or a temporal TemporalDag attachment."
    )

    from antecedent._analyze import handle_response
    from antecedent.errors import CausalUnsupportedError

    with pytest.raises(CausalUnsupportedError) as ei:
        handle_response(
            _DATA,
            _CURVE,
            graph=antecedent.Pag.from_marked_edges(["t", "y"], [("t", "y", "circle", "arrow")]),
            discovery=None,
            inference=antecedent.Frequentist(),
            identifier=None,
            estimator=None,
            estimator_config=None,
            validators=None,
            refute_requested=False,
            refute=False,
            bootstrap_requested=False,
            seed=1,
            threads=1,
        )
    assert str(ei.value) == f"refused: {reason}"


def test_intervention_response_pag_literal_matches_support_closed_toml():
    """`_analyze.py`'s hand-rolled InterventionResponse/off-Dag refusal must not drift.

    Same routing gate as `test_response_curve_pag_literal_matches_support_closed_toml`
    (`handle_response`'s `isinstance(graph, (Admg, Cpdag, Pag))` check), but
    `InterventionResponse` gets its own closed-rule text so the raised message doesn't
    misdescribe the query as `ResponseCurve`.
    """
    rules = tomllib.loads(_SUPPORT_CLOSED_TOML.read_text())["closed"]
    matches = [
        rule
        for rule in rules
        if rule.get("queries") == ["InterventionResponse"]
        and set(rule.get("graph_classes", [])) == {"Pag", "Cpdag", "Admg"}
    ]
    assert len(matches) == 1, matches
    reason = matches[0]["reason"]

    from antecedent._analyze import handle_response
    from antecedent.errors import CausalUnsupportedError

    with pytest.raises(CausalUnsupportedError) as ei:
        handle_response(
            _DATA,
            antecedent.InterventionResponse("y", intervention={"t": 1.0}),
            graph=antecedent.Pag.from_marked_edges(["t", "y"], [("t", "y", "circle", "arrow")]),
            discovery=None,
            inference=antecedent.Frequentist(),
            identifier=None,
            estimator=None,
            estimator_config=None,
            validators=None,
            refute_requested=False,
            refute=False,
            bootstrap_requested=False,
            seed=1,
            threads=1,
        )
    assert str(ei.value) == f"refused: {reason}"


def test_accepted_graph_analyze_response_curve_threads_accepted(monkeypatch):
    """`AcceptedGraph.analyze` with a ResponseCurve must reach native with accepted=True.

    `analyze()`'s one-shot result has no exposed `structure_source` field (unlike
    `PreparedAnalysis.structure_source`, asserted in test_accepted_graph.py /
    test_licensed_staged.py), so the only way to observe the marker on this path
    is at the native call boundary itself.
    """
    captured: dict[str, object] = {}
    real = antecedent._analyze._analyze_response

    def spy(*args, **kwargs):
        captured["accepted"] = kwargs.get("accepted")
        return real(*args, **kwargs)

    monkeypatch.setattr(antecedent._analyze, "_analyze_response", spy)

    result = _ACCEPTED.analyze(_DATA, query=_CURVE, refute=False, seed=1)
    assert captured["accepted"] is True
    assert result.response is not None
    assert np.isfinite(result.response.values).all()

    captured.clear()
    result = antecedent.analyze(_DATA, graph=_DAG, query=_CURVE, refute=False, seed=1)
    assert captured["accepted"] is False
    assert result.response is not None


def test_path_distribution_literals_match_support_closed_toml():
    """The three hand-rolled path/distribution refusals in `_analyze.py` must not drift.

    They fire as routing gates (discovery= and accepted-structure pre-checks)
    before the native support-matrix consultation, so they cannot be derived
    from the TOML across the language boundary. Pin source text against the
    TOML reason so either side changing alone fails loudly.
    """
    import inspect

    import antecedent._analyze as _analyze

    rules = tomllib.loads(_SUPPORT_CLOSED_TOML.read_text())["closed"]
    matches = [
        rule
        for rule in rules
        if set(rule.get("queries", [])) == {"PathSpecificEffect", "InterventionalDistribution"}
    ]
    assert len(matches) == 1, matches
    reason = matches[0]["reason"]
    source = inspect.getsource(_analyze)
    # The literal is wrapped across two adjacent string fragments; normalize.
    collapsed = source.replace('"\n            "', "")
    count = collapsed.count(f"refused: {reason}")
    assert count >= 3, (
        f"expected the TOML reason to appear at the three hand-rolled sites; "
        f"found {count}: {reason!r}"
    )
