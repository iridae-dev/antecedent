"""Response-family support-matrix enforcement and structure-source threading.

Covers two branch-review defects fixed together:

* The native response preparation (python/src/response_api.rs) now consults the
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

import antecedent
import numpy as np
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, load_toml

_REPO_ROOT = REPO_ROOT
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
    result = require_success(_ACCEPTED.analyze(_DATA, query=_CURVE, refute=False, seed=1))
    assert result.response is not None
    assert np.isfinite(result.response.values).all()
    assert (
        antecedent.artifacts.loads(bytes(result.export())).contract["structure_source"]
        == "accepted"
    )

    result = require_success(
        antecedent.analyze(_DATA, graph=_DAG, query=_CURVE, refute=False, seed=1)
    )
    assert result.response is not None
    assert (
        antecedent.artifacts.loads(bytes(result.export())).contract["structure_source"]
        == "explicit"
    )


def test_path_distribution_graph_posterior_rule_executes_and_the_prepare_gate_is_gone():
    """A graph posterior of path-restricted effects or distributions executes, unlicensed.

    The rule in the TOML says every atom is answered by the executor its own structure has, so
    the prepare routing gate that once refused the cell must not remain beside it.
    """
    rules = load_toml(_SUPPORT_CLOSED_TOML)["closed"]
    matches = [
        rule
        for rule in rules
        if set(rule.get("queries", [])) == {"PathSpecificEffect", "InterventionalDistribution"}
        and set(rule.get("structures", [])) == {"graph_posterior"}
    ]
    assert len(matches) == 1, matches
    assert matches[0].get("executes_unlicensed") is True
    rng = np.random.default_rng(5)
    treatment = (rng.random(200) < 0.5).astype(float)
    outcome = (rng.random(200) < 0.3 + 0.4 * treatment).astype(float)
    prepared = antecedent.prepare(
        {"t": treatment, "y": outcome},
        query=antecedent.PathSpecificEffect("t", "y"),
        discovery=antecedent.discovery.ExactDagPosterior(),
        refute=False,
        bootstrap=0,
    )
    assert prepared is not None


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
    result = require_success(
        antecedent.analyze(
            _DATA,
            query=_CURVE,
            graph=_DAG,
            inference=antecedent.Bayesian(backend="conjugate", n_draws=256),
            refute=False,
        )
    )
    assert result.response is not None
    assert np.isfinite(result.response.values).all()


def test_handle_response_bayesian_derivative_is_licensed():
    result = require_success(
        antecedent.analyze(
            _DATA,
            query=antecedent.AverageDerivative("t", "y"),
            graph=_DAG,
            inference=antecedent.Bayesian(backend="conjugate", n_draws=32),
            refute=False,
        )
    )
    assert result is not None
