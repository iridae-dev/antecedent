"""Unlicensed Bayesian survival and fixed group-dose policy posteriors through the package.

The routes answer the same targets as their frequentist rows (the treated-minus-control
survival and restricted-mean contrast; the kernel-smoothed fixed group-dose policy value) under a
declared likelihood and prior. Every reference below is independent of the library: gamma
hazard updates, closed-form survival means and quadrature restricted means by hand, per-group
ridge posteriors from numpy with scipy window quadrature, and the frequentist row of the same
rows. The routes are never licensed and their calibration is unmeasured.
"""

from __future__ import annotations

import json
import subprocess
import sys

import antecedent
import numpy as np
import pytest
from antecedent.design_models import DosePolynomial, PiecewiseHazards, QuantileGrid
from antecedent.errors import CausalError
from antecedent.observation import IndependentGiven
from antecedent.outcomes import require_success
from antecedent.policy import ConditionalDoseResponse
from antecedent.query import Quantile
from antecedent.survival import CompetingRisksOutcome, SurvivalOutcome


def _small_survival() -> dict[str, list[float]]:
    duration = [0.5, 1.2, 2.5, 3.5, 1.8, 0.9, 0.7, 1.5, 2.2, 3.2, 3.8, 1.1]
    event = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    treatment = [0.0] * 6 + [1.0] * 6
    return {"duration": duration, "event": event, "treatment": treatment}


def _survival_query(tau: float = 3.0) -> SurvivalOutcome:
    return SurvivalOutcome(
        "duration",
        "event",
        "treatment",
        tau,
        randomized=True,
        observation_assumption=IndependentGiven(()),
    )


def _hazards(cuts=(1.0, 2.0, 3.0), shape: float = 1.0, rate: float = 1.0) -> antecedent.Bayesian:
    return antecedent.Bayesian(
        backend="conjugate",
        n_draws=40_000,
        design_model=PiecewiseHazards(cuts=cuts, prior_shape=shape, prior_rate=rate),
    )


def test_estimator_ids_are_public() -> None:
    assert antecedent.Estimator.BAYESIAN_PIECEWISE_SURVIVAL.value == "bayesian.piecewise_survival"
    assert antecedent.Estimator.BAYESIAN_DOSE_POLICY.value == "bayesian.continuous_dose_policy"


def test_survival_posterior_matches_the_gamma_hazard_reference() -> None:
    result = require_success(
        antecedent.analyze(
            _small_survival(), query=_survival_query(), inference=_hazards(), seed=11
        )
    )
    row = result.posterior_row
    assert row is not None and result.survival is None
    assert row.estimator == "bayesian.piecewise_survival"
    assert (row.uncertainty, row.calibration, row.licensed) == ("posterior", "unmeasured", False)
    assert row.coordinates == ("survival_difference", "rmst_difference")
    # Events and exposure by hand: Gamma(1, 1) hazard updates are shape 1 + events, rate
    # 1 + time at risk; survival means are (rate / (rate + length)) ** shape per interval.
    assert row.vectors["control_events"] == (2.0, 1.0, 1.0)
    assert row.vectors["control_shape"] == (3.0, 2.0, 2.0)
    assert row.vectors["control_rate"] == pytest.approx((6.4, 4.0, 2.5))
    assert row.extras["control_survival_mean"] == pytest.approx(0.2112363552413642, abs=1e-12)
    assert row.extras["treated_survival_mean"] == pytest.approx(0.4473262403291556, abs=1e-12)
    assert row.posterior_mean("survival_difference") == pytest.approx(0.2360898850877914, abs=1e-12)
    assert row.posterior_mean("rmst_difference") == pytest.approx(0.5380182589051, abs=1e-9)
    # Two-million-draw numpy simulation of the same posterior.
    lower, upper = row.credible_interval("survival_difference")
    assert lower == pytest.approx(-0.19149, abs=0.03) and upper == pytest.approx(0.64314, abs=0.03)
    lower, upper = row.credible_interval("rmst_difference")
    assert lower == pytest.approx(-0.40123, abs=0.06) and upper == pytest.approx(1.41953, abs=0.06)
    assert row.extras["probability_treated_survives_longer"] == pytest.approx(0.862216, abs=0.012)
    # The retained joint-draw covariance is symmetric and positive on the diagonal.
    covariance = row.covariance_matrix()
    assert covariance is not None and covariance.shape == (2, 2)
    assert covariance[0, 1] == pytest.approx(covariance[1, 0])
    assert covariance[0, 0] > 0 and covariance[1, 1] > 0
    assert result.estimate.ate is None
    assert result.answer.kind == "structured"
    assert result.answer.detail == "unlicensed_posterior_bayesian_piecewise_survival"
    assert "calibration is unmeasured" in result.claim()


def test_flat_prior_survival_posterior_tracks_the_frequentist_product_limit() -> None:
    n = 400
    duration: list[float] = []
    event: list[float] = []
    treatment: list[float] = []
    for arm, hazard in ((0.0, 0.4), (1.0, 0.25)):
        for i in range(n):
            time = -np.log(1.0 - (i + 0.5) / n) / hazard
            duration.append(float(min(time, 3.0)))
            event.append(float(time <= 3.0))
            treatment.append(arm)
    data = {"duration": duration, "event": event, "treatment": treatment}
    frequentist = require_success(antecedent.analyze(data, query=_survival_query())).survival
    assert frequentist is not None
    flat = antecedent.Bayesian(
        backend="conjugate",
        n_draws=2_000,
        design_model=PiecewiseHazards(
            cuts=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), prior_shape=1e-3, prior_rate=1e-3
        ),
    )
    row = require_success(
        antecedent.analyze(data, query=_survival_query(), inference=flat, seed=3)
    ).posterior_row
    assert row is not None
    survival_difference = frequentist.treated_survival[-1] - frequentist.control_survival[-1]
    assert row.posterior_mean("survival_difference") == pytest.approx(survival_difference, abs=0.02)
    assert row.posterior_mean("rmst_difference") == pytest.approx(
        frequentist.rmst_treated - frequentist.rmst_control, abs=0.04
    )
    assert row.credible_interval("survival_difference")[0] > 0.0


def test_survival_posterior_round_trips_in_this_process_and_a_fresh_one(tmp_path) -> None:
    result = require_success(
        antecedent.analyze(_small_survival(), query=_survival_query(), inference=_hazards(), seed=5)
    )
    artifact = bytes(result.export(artifact_id="bayesian-survival"))
    loaded = require_success(antecedent.load(artifact))
    assert loaded.answer == result.answer
    structured = loaded.answer.structured
    assert structured["estimator"] == "bayesian.piecewise_survival"
    assert structured["licensed"] is False and structured["calibration"] == "unmeasured"
    assert structured["mean"] == pytest.approx(list(result.posterior_row.mean))
    path = tmp_path / "survival.antecedent"
    path.write_bytes(artifact)
    script = (
        "import antecedent, json, sys\n"
        "answer = antecedent.load(open(sys.argv[1], 'rb').read()).answer\n"
        "print(json.dumps(answer.structured))\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    fresh = json.loads(done.stdout)
    assert fresh["mean"] == pytest.approx(list(result.posterior_row.mean))
    assert fresh["covariance"] == pytest.approx(list(result.posterior_row.covariance))


def test_survival_posterior_refusals_name_the_next_attempt() -> None:
    data = _small_survival()
    # Competing causes.
    with pytest.raises(CausalError, match="competing_causes_unsupported") as caught:
        require_success(
            antecedent.analyze(
                {**data, "cause": [2.0] + data["event"][1:]},
                query=CompetingRisksOutcome(
                    "duration",
                    "cause",
                    "treatment",
                    1,
                    3.0,
                    randomized=True,
                    observation_assumption=IndependentGiven(()),
                ),
                inference=_hazards(),
            )
        )
    assert "Next attempt" in str(caught.value)
    # Undeclared partition and prior.
    with pytest.raises(CausalError, match="model_not_declared") as caught:
        require_success(
            antecedent.analyze(
                data,
                query=_survival_query(),
                inference=antecedent.Bayesian(backend="conjugate", n_draws=1_000),
            )
        )
    assert "Next attempt" in str(caught.value)
    # The last cut is the horizon.
    with pytest.raises(CausalError, match="horizon_mismatch"):
        require_success(
            antecedent.analyze(data, query=_survival_query(), inference=_hazards(cuts=(1.0, 2.0)))
        )
    # A bootstrap has no meaning for a posterior.
    with pytest.raises(CausalError, match="bootstrap_not_applicable"):
        require_success(
            antecedent.analyze(data, query=_survival_query(), inference=_hazards(), bootstrap=300)
        )


def _dose_rows(rows: int = 800) -> dict[str, np.ndarray]:
    index = np.arange(rows)
    dose = ((index * 37) % 997) / 997.0
    group = np.where(index < rows // 2, "a", "b")
    outcome = (
        1.0
        + (index >= rows // 2).astype(float)
        + 0.35 * np.sin(0.13 * index)
        + np.where(index < rows // 2, 1.5, 2.0) * dose
    )
    return {"outcome": outcome, "dose": dose, "density": np.ones(rows), "group": group}


def _dose_query(**changes) -> ConditionalDoseResponse:
    arguments = {
        "outcome": "outcome",
        "dose": "dose",
        "baseline_group": "group",
        "dose_density": "density",
        "target_doses": (),
        "bandwidth": 0.2,
        "density_provenance": "known",
        "min_local_support": 80,
        "policy_doses": {"a": 0.7, "b": 0.8},
        "reference_doses": {"a": 0.3, "b": 0.4},
    }
    arguments.update(changes)
    return ConditionalDoseResponse(**arguments)


def _polynomial() -> antecedent.Bayesian:
    return antecedent.Bayesian(
        backend="conjugate",
        n_draws=4_000,
        design_model=DosePolynomial(
            support=(0.0, 1.1),
            coefficient_scales=(3.0, 2.0, 1.0),
            residual_shape=2.0,
            residual_scale=0.1,
        ),
    )


def test_dose_policy_posterior_tracks_the_frequentist_kernel_row() -> None:
    data = _dose_rows()
    query = _dose_query()
    frequentist = require_success(
        antecedent.analyze(data, query=query, refute="none")
    ).continuous_dose_response
    result = require_success(
        antecedent.analyze(data, query=query, inference=_polynomial(), refute="none", seed=7)
    )
    row = result.posterior_row
    assert row is not None and result.continuous_dose_response is None
    assert row.estimator == "bayesian.continuous_dose_policy"
    assert (row.calibration, row.licensed) == ("unmeasured", False)
    assert row.coordinates == ("policy_value", "reference_value", "incremental_value")
    # The truth is linear, so each group's symmetric window average equals the curve value:
    # 0.5 * 1.5 * 0.4 + 0.5 * 2.0 * 0.4 = 0.7.
    incremental = row.posterior_mean("incremental_value")
    assert incremental == pytest.approx(0.7, abs=0.1)
    assert incremental == pytest.approx(frequentist.fixed_policy.incremental_value, abs=0.1)
    assert row.posterior_mean("policy_value") == pytest.approx(
        frequentist.fixed_policy.policy_value, abs=0.1
    )
    assert row.exact_mean is None
    covariance = row.covariance_matrix()
    assert covariance is not None and covariance.shape == (3, 3)
    # Paired draws: the incremental variance is the paired one, not the sum of the two.
    assert covariance[2, 2] == pytest.approx(
        covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1], rel=1e-6
    )
    assert result.answer.detail == "unlicensed_posterior_bayesian_continuous_dose_policy"
    assert (
        require_success(
            antecedent.load(bytes(result.export(artifact_id="bayesian-dose-policy")))
        ).answer
        == result.answer
    )


def test_dose_policy_posterior_matches_the_per_group_conjugate_reference() -> None:
    dose = [0.2, 0.6, 1.1, 1.5, 1.9, 2.4, 2.8, 3.3, 0.4, 0.9, 1.3, 1.8, 2.2, 2.6, 3.0, 3.7]
    outcome = [1.0, 1.6, 2.2, 2.5, 2.4, 2.0, 1.7, 1.1, 0.5, 0.9, 1.0, 1.4, 1.9, 2.0, 2.4, 2.9]
    data = {
        "outcome": np.array(outcome),
        "dose": np.array(dose),
        "density": np.ones(16),
        "group": np.array(["A"] * 8 + ["B"] * 8),
    }
    query = _dose_query(
        bandwidth=1.0,
        min_local_support=2,
        policy_doses={"A": 1.5, "B": 2.5},
        reference_doses={"A": 2.0, "B": 1.0},
    )
    inference = antecedent.Bayesian(
        backend="conjugate",
        n_draws=40_000,
        design_model=DosePolynomial(
            support=(0.0, 4.0), coefficient_scales=(3.0, 2.0, 1.0), residual_variance=0.25
        ),
    )
    row = require_success(
        antecedent.analyze(data, query=query, inference=inference, refute="none", seed=5)
    ).posterior_row
    # numpy ridge posteriors per group, scipy adaptive quadrature of the window average.
    assert row.exact_mean == pytest.approx((1.91559566, 1.43144023, 0.48415543), abs=1e-7)
    exact = row.exact_covariance_matrix()
    assert exact is None or exact.shape == (3, 3)
    assert row.vectors["exact_variances"] == pytest.approx(
        (0.01998154, 0.02391098, 0.01808961), abs=1e-7
    )
    assert row.mean == pytest.approx(row.exact_mean, abs=0.01)


def test_dose_policy_refusals_name_the_next_attempt() -> None:
    data = _dose_rows()
    with pytest.raises(CausalError, match="model_not_declared") as caught:
        require_success(
            antecedent.analyze(
                data,
                query=_dose_query(),
                inference=antecedent.Bayesian(backend="conjugate", n_draws=1_000),
                refute="none",
            )
        )
    assert "Next attempt" in str(caught.value)
    with pytest.raises(CausalError, match="grid_not_read"):
        require_success(
            antecedent.analyze(
                data,
                query=_dose_query(target_doses=(0.5,)),
                inference=_polynomial(),
                refute="none",
            )
        )
    with pytest.raises(CausalError, match="policy_required"):
        require_success(
            antecedent.analyze(
                data,
                query=_dose_query(policy_doses=None, reference_doses=None, target_doses=(0.5,)),
                inference=_polynomial(),
                refute="none",
            )
        )


def _quantile_data() -> dict[str, list[float]]:
    control = [0.4, 0.8, 1.1, 1.3, 1.7, 2.2, 2.4, 2.9, 3.3, 3.6, 4.1, 4.6]
    active = [0.9, 1.4, 1.8, 2.1, 2.6, 2.8, 3.1, 3.4, 3.9, 4.3, 4.4, 4.8]
    return {"t": [0.0] * 12 + [1.0] * 12, "y": control + active}


def _quantile(concentration: float = 0.5, model: bool = True, **kwargs):
    inference = antecedent.Bayesian(
        backend="conjugate",
        n_draws=40_000,
        design_model=QuantileGrid(
            thresholds=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0), bin_concentration=concentration
        )
        if model
        else None,
    )
    graph = kwargs.pop("graph", antecedent.Dag.from_edges(["t", "y"], [("t", "y")]))
    return require_success(
        antecedent.analyze(
            kwargs.pop("data", _quantile_data()),
            graph=graph,
            query=antecedent.AverageEffect(
                "t", "y", outcome_functional=Quantile(kwargs.pop("tau", 0.5))
            ),
            estimator=antecedent.Estimator.BAYESIAN_RANDOMIZED_QUANTILE,
            inference=inference,
            refute="none",
            seed=kwargs.pop("seed", 9),
        )
    )


def test_quantile_posterior_matches_the_dirichlet_reference() -> None:
    result = _quantile()
    row = result.posterior_row
    assert row is not None
    assert row.estimator == "bayesian.randomized_quantile"
    assert (row.calibration, row.licensed) == ("unmeasured", False)
    assert row.coordinates == ("control_quantile", "treated_quantile", "quantile_difference")
    # Bin counts by hand (control 0,2,3,3,2,2,0; treated 0,1,2,3,3,3,0) under a 0.5 prior per
    # bin: total mass 15.5 and these cumulative masses at the grid.
    assert row.vectors["control_cdf_mean"] == pytest.approx(
        tuple(a / 15.5 for a in (0.5, 3.0, 6.5, 10.0, 12.5, 15.0)), abs=1e-12
    )
    assert row.vectors["treated_cdf_mean"] == pytest.approx(
        tuple(a / 15.5 for a in (0.5, 2.0, 4.5, 8.0, 11.5, 15.0)), abs=1e-12
    )
    assert row.extras["draws_requested"] == 40_000.0
    assert row.extras["draws_resolved"] >= 39_600.0
    assert result.estimate.ate == pytest.approx(row.posterior_mean("quantile_difference"), abs=1e-9)
    joint = row.covariance_matrix()
    assert joint is not None and joint.shape == (3, 3)
    assert abs(joint[0, 1]) < 0.05 * np.sqrt(joint[0, 0] * joint[1, 1])
    assert (
        require_success(
            antecedent.load(bytes(result.export(artifact_id="bayesian-quantile")))
        ).answer
        == result.answer
    )


def test_vague_prior_quantile_posterior_tracks_the_empirical_inversion() -> None:
    # Control grid distribution function 0, 2/12, 5/12, 8/12, 10/12, 1: median 2 + (1/2 - 5/12)/(1/4).
    row = _quantile(concentration=0.01, seed=4).posterior_row
    assert row.posterior_mean("control_quantile") == pytest.approx(7.0 / 3.0, abs=0.35)
    assert row.posterior_mean("treated_quantile") == pytest.approx(3.0, abs=0.35)


def test_quantile_posterior_refusals_name_the_next_attempt() -> None:
    with pytest.raises(CausalError, match="model_not_declared") as caught:
        _quantile(model=False)
    assert "Next attempt" in str(caught.value)
    with pytest.raises(CausalError, match="unresolved_draws") as caught:
        _quantile(tau=0.999)
    assert "Next attempt" in str(caught.value)
    data = _quantile_data()
    data["z"] = [float(i % 3) for i in range(24)]
    confounded = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    with pytest.raises(CausalError, match="adjustment_not_supported"):
        _quantile(data=data, graph=confounded)
