"""Response-family support-matrix enforcement and structure-source threading.

Covers two branch-review defects fixed together:

* The native `analyze_response` (python/src/response_api.rs) now consults the
  generated support matrix itself (`antecedent::support::refuse_if_not_applicable`)
  instead of being enforced only by hand-typed Python literals that could drift
  from `parity/support_closed.toml`. Admg InterventionResponse cheap/full is
  licensed on the plugin-level suite; this file pins that the closed rule is gone.
* `AcceptedGraph.analyze(..., query=ResponseCurve(...))` now threads
  `structure_accepted` down to the native call's `accepted=` parameter, so the
  response family is not silently misclassified as `explicit` structure the way
  `prepare`/`prepare_response` already record `accepted` correctly.
"""

from __future__ import annotations

from pathlib import Path

import antecedent
import numpy as np

from _repo_text import load_toml

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


def test_admg_intervention_cheap_is_not_a_closed_rule():
    """Admg IR cheap/full is licensed plugin-level; closed rule #3 is gone."""
    rules = load_toml(_SUPPORT_CLOSED_TOML)["closed"]
    matches = [
        rule
        for rule in rules
        if "InterventionResponse" in set(rule.get("queries", []))
        and "Admg" in set(rule.get("graph_classes", []))
        and set(rule.get("validations", [])) == {"cheap", "full"}
    ]
    assert matches == []


def test_accepted_graph_analyze_response_curve_threads_accepted():
    """The retained contract preserves reviewed versus explicit structure."""
    result = _ACCEPTED.analyze(_DATA, query=_CURVE, refute=False, seed=1)
    assert result.response is not None
    assert np.isfinite(result.response.values).all()
    assert antecedent.artifacts.loads(result.export()).contract["structure_source"] == "accepted"

    result = antecedent.analyze(_DATA, graph=_DAG, query=_CURVE, refute=False, seed=1)
    assert result.response is not None
    assert antecedent.artifacts.loads(result.export()).contract["structure_source"] == "explicit"


def test_path_distribution_literals_match_support_closed_toml():
    """The prepare routing-gate refusal must not drift from the TOML.

    It fires before the native support-matrix consultation, so it cannot be
    derived from the TOML across the language boundary. Pin source text
    against the TOML reason so either side changing alone fails loudly.
    """
    import inspect

    import antecedent.estimation as estimation

    rules = load_toml(_SUPPORT_CLOSED_TOML)["closed"]
    matches = [
        rule
        for rule in rules
        if set(rule.get("queries", [])) == {"PathSpecificEffect", "InterventionalDistribution"}
        and set(rule.get("structures", [])) == {"graph_posterior"}
    ]
    assert len(matches) == 1, matches
    reason = matches[0]["reason"]
    source = inspect.getsource(estimation)
    # The literal is wrapped across adjacent string fragments; normalize.
    collapsed = source.replace('"\n            "', "")
    count = collapsed.count(reason.rstrip("."))
    assert count >= 1, (
        f"expected the TOML reason at the prepare routing gate; found {count}: {reason!r}"
    )


def _handle_response_kwargs(**overrides):
    kwargs = dict(
        graph=_DAG,
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
    kwargs.update(overrides)
    return kwargs


def test_handle_response_bayesian_curve_uses_staged_path():
    """analyze is prepare().estimate(); Bayesian ResponseCurve stays licensed."""
    result = antecedent.analyze(
        _DATA,
        query=_CURVE,
        graph=_DAG,
        inference=antecedent.Bayesian(backend="conjugate", n_draws=256),
        refute=False,
    )
    assert result.response is not None
    assert np.isfinite(result.response.values).all()


def test_handle_response_bayesian_derivative_is_licensed():
    result = antecedent.analyze(
        _DATA,
        query=antecedent.AverageDerivative("t", "y"),
        graph=_DAG,
        inference=antecedent.Bayesian(backend="conjugate", n_draws=32),
        refute=False,
    )
    assert result is not None
