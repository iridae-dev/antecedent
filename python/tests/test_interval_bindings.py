"""The intervals an execution reports, read back as typed bindings on the outcome.

Each binding states how one reported interval was formed (method, level, dependence rule,
replicate and draw counts) as derived by the facade from the result's own content. The
circular-block families are exercised through the temporal route that resamples that family:
one test per family.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import antecedent as ant
import numpy as np
import pytest
from antecedent.outcomes import require_success
from antecedent.results import IntervalBindingView

from _interval_fixtures import (
    BOOTSTRAP,
    mediation_data,
    mixture_data,
    row_block_data,
    sequential_data,
    single_window_data,
)
from known_truth import TEMPORAL, temporal_posterior

_EXECUTION = ant.Execution(seed=3, threads=1)


def _primary(result) -> IntervalBindingView:
    bindings = result.interval_bindings
    assert bindings and all(isinstance(b, IntervalBindingView) for b in bindings)
    return bindings[0]


def _assert_block_interval(binding: IntervalBindingView, family: str) -> None:
    assert binding.method == "circular_block_se"
    assert binding.dependence == f"circular_block:{family}"
    assert binding.block_family == family
    assert binding.level == pytest.approx(0.95)
    assert binding.replicates_ok is not None and 0 < binding.replicates_ok <= BOOTSTRAP
    assert binding.posterior_draws is None


def _analyze(data, *, query, graph=None, discovery=None, bootstrap=BOOTSTRAP):
    options = {"graph": graph} if discovery is None else {"discovery": discovery}
    return require_success(
        ant.analyze(
            data,
            query=query,
            inference=ant.Frequentist(bootstrap=bootstrap),
            refute=False,
            execution=_EXECUTION,
            **options,
        )
    )


def test_a_single_window_fit_reports_a_single_window_circular_block_interval():
    data, graph = single_window_data()
    result = _analyze(
        data,
        graph=graph,
        query=ant.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
    )
    assert result.estimate.se_bootstrap is not None
    _assert_block_interval(_primary(result), "single_window")


def test_a_sequential_window_reports_a_sequential_circular_block_interval():
    data, graph = sequential_data()
    result = _analyze(
        data,
        graph=graph,
        query=ant.SustainedEffect("t", "y", treatment_lag=2, horizon_steps=1, window=(-2, -1)),
    )
    assert result.estimate.se_bootstrap is not None
    _assert_block_interval(_primary(result), "sequential")


def test_a_temporal_mediation_reports_a_mediation_circular_block_interval():
    data, graph = mediation_data()
    result = _analyze(
        data,
        graph=graph,
        query=ant.TemporalMediationEffect("t", "m", "y", contrast="mediated"),
    )
    assert result.estimate.se_bootstrap is not None
    _assert_block_interval(_primary(result), "mediation")


def test_a_graph_posterior_mixture_reports_a_mixture_circular_block_interval():
    data = mixture_data(int(TEMPORAL["n"]), int(TEMPORAL["seed"]))
    result = _analyze(
        data,
        discovery=temporal_posterior(),
        query=ant.PulseEffect(
            "pressure", "defect", treatment_lag=1, horizon_steps=1, active_level=1.0
        ),
    )
    assert result.estimate.se_bootstrap is not None
    _assert_block_interval(_primary(result), "mixture")


def test_a_temporal_admg_window_reports_a_row_block_circular_block_interval():
    data, graph = row_block_data()
    result = _analyze(
        data,
        graph=graph,
        query=ant.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1),
    )
    assert result.estimate.se_bootstrap is not None
    _assert_block_interval(_primary(result), "row_block")


def test_without_replicates_no_block_interval_is_reported():
    data, graph = single_window_data()
    result = _analyze(
        data,
        graph=graph,
        query=ant.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
        bootstrap=0,
    )
    binding = _primary(result)
    assert binding.method == "none"
    assert binding.level is None
    assert binding.block_family is None
    assert binding.replicates_ok is None


def _static_rows(n: int = 300, seed: int = 4):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
    y = 1.0 + 2.0 * t + 0.7 * z + rng.normal(scale=0.5, size=n)
    return {"t": t, "y": y, "z": z}


def test_analytic_bootstrap_and_posterior_intervals_name_their_own_construction():
    data = _static_rows()
    graph = ant.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    query = ant.AverageEffect("t", "y")

    def run(inference):
        return require_success(
            ant.analyze(
                data,
                graph=graph,
                query=query,
                inference=inference,
                refute=False,
                execution=_EXECUTION,
            )
        )

    analytic = _primary(run(ant.Frequentist(bootstrap=0)))
    assert analytic.method == "analytic_se"
    assert analytic.dependence == "iid" and analytic.block_family is None
    assert analytic.level == pytest.approx(0.95)
    assert analytic.replicates_ok is None and analytic.posterior_draws is None

    bootstrap = _primary(run(ant.Frequentist(bootstrap=30)))
    assert bootstrap.method == "bootstrap_se"
    assert bootstrap.dependence == "iid"
    assert bootstrap.replicates_ok is not None and 0 < bootstrap.replicates_ok <= 30

    posterior = _primary(run(ant.Bayesian(backend="conjugate", n_draws=64)))
    assert posterior.method == "posterior_quantile"
    assert posterior.posterior_draws == 64
    assert posterior.replicates_ok is None


def test_the_view_reads_the_native_json_without_rederiving_it():
    payload = json.dumps(
        [
            {
                "method": "circular_block_se",
                "se_kind": None,
                "level": 0.95,
                "dependence": "circular_block:mixture",
                "replicates_ok": 38,
                "replicates_failed": 2,
                "posterior_draws": None,
            },
            {
                "method": "none",
                "se_kind": None,
                "level": None,
                "dependence": "iid",
                "replicates_ok": None,
                "replicates_failed": None,
                "posterior_draws": None,
            },
        ]
    )
    from antecedent._from_raw import _interval_bindings_from_raw

    views = _interval_bindings_from_raw(SimpleNamespace(interval_bindings_json=payload))
    assert [view.method for view in views] == ["circular_block_se", "none"]
    assert views[0].block_family == "mixture"
    assert (views[0].replicates_ok, views[0].replicates_failed) == (38, 2)
    assert views[1].level is None and views[1].block_family is None
    assert _interval_bindings_from_raw(SimpleNamespace(interval_bindings_json=None)) == ()
    assert _interval_bindings_from_raw(SimpleNamespace()) == ()


def _published(result) -> tuple[float, str]:
    """The one standard error a result publishes and the interval method that labels it."""
    wire = result.to_dict()["reasoning"]
    return (
        float(wire["uncertainty"]["payload"]["standard_error"]),
        wire["contract"]["claim"]["calibration"]["basis"]["key"]["interval_method"],
    )


def test_the_published_standard_error_is_the_selected_one_and_matches_the_ols_oracle():
    data = _static_rows()
    graph = ant.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])

    def run(inference):
        return require_success(
            ant.analyze(
                data,
                graph=graph,
                query=ant.AverageEffect("t", "y"),
                inference=inference,
                refute=False,
                execution=_EXECUTION,
            )
        )

    # Independent homoskedastic OLS standard error of the treatment coefficient in y ~ 1 + t + z.
    design = np.column_stack([np.ones(len(data["t"])), data["t"], data["z"]])
    coefficients, *_ = np.linalg.lstsq(design, data["y"], rcond=None)
    residuals = data["y"] - design @ coefficients
    sigma2 = residuals @ residuals / (len(residuals) - design.shape[1])
    oracle = float(np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1]))

    analytic = run(ant.Frequentist(bootstrap=0))
    published, method = _published(analytic)
    assert method == "analytic_se"
    assert analytic.estimate.se_bootstrap is None
    assert published == analytic.estimate.se_analytic
    assert published == pytest.approx(oracle, rel=1e-9)

    # With replicates the bootstrap standard error is the one published, not the analytic one.
    bootstrap = run(ant.Frequentist(bootstrap=30))
    published, method = _published(bootstrap)
    assert method == "bootstrap_se"
    assert bootstrap.estimate.se_bootstrap is not None
    assert published == bootstrap.estimate.se_bootstrap
    assert published != bootstrap.estimate.se_analytic
    assert bootstrap.estimate.se_analytic == pytest.approx(oracle, rel=1e-9)
