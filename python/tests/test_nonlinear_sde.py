# SPDX-License-Identifier: MIT OR Apache-2.0
"""Nonlinear SDE ensembles and quasi-likelihood fits against independent references.

The expected values come from ``conformance/gcm/nonlinear_sde_ensemble/reference.py``, a
NumPy implementation that shares no code with the native expression tape: a replay of the
documented epidemic ensemble, exact moment ODEs of a square-root diffusion, and the dense
Euler quasi-likelihood of trajectories drawn with ``numpy.random.default_rng``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from antecedent.cyclic import (
    SdeAction,
    SdeBound,
    SdeContrast,
    SdeDiffusion,
    SdeExpression,
    SdeInput,
    SdeObservationModel,
    SdeSimulation,
    SdeTrajectory,
    SdeWorld,
    StochasticDifferentialEquation,
)
from antecedent.outcomes import require_success

REFERENCE = json.loads(
    (
        Path(__file__).resolve().parents[2] / "conformance/gcm/nonlinear_sde_ensemble/expected.json"
    ).read_text(encoding="utf-8")
)
E = SdeExpression
INITIAL = (950.0, 50.0, 0.0)
TRUTH = (0.6, 0.3, 0.25)


def run(prepared) -> dict:
    return require_success(prepared.execute()).to_dict()["answer"]["result"]


def refusal_text(prepared) -> str:
    """The refusal of a request, whether raised at preparation or at execution."""
    if prepared.refusal is not None:
        return f"{prepared.refusal.code} {prepared.refusal.detail}"
    with pytest.raises(Exception) as error:  # noqa: PT011 - the text is what is asserted
        require_success(prepared.execute())
    return str(error.value)


LOCKDOWN = SdeWorld(
    "lockdown", inputs=(SdeInput("season", (0.0,), (-1.0,), "piecewise_constant", (-1.0, 1.0)),)
)
SUPPRESS = SdeWorld("suppress", actions=(SdeAction("I", (5.0,), (20.0,)),))


def epidemic_simulation(**changes) -> SdeSimulation:
    base = {
        "step": 0.05,
        "steps": 400,
        "paths": 200,
        "seed": 20260101,
        "output_every": 80,
        "weak_tolerance": 0.05,
    }
    return SdeSimulation(**{**base, **changes})


def test_the_epidemic_ensemble_replays_the_independent_numpy_reference():
    spec = REFERENCE["epidemic"]["spec"]
    assert spec["seed"] == 20260101
    result = run(
        StochasticDifferentialEquation.epidemic().simulate(
            INITIAL,
            epidemic_simulation(),
            interventions=[LOCKDOWN, SUPPRESS],
            contrasts=[
                SdeContrast("lockdown_vs_baseline", "lockdown", "baseline"),
                SdeContrast("suppress_vs_baseline", "suppress", "baseline"),
            ],
        )
    )["simulation"]
    assert [w["id"] for w in result["worlds"]] == ["baseline", "lockdown", "suppress"]
    for world in result["worlds"]:
        expected = REFERENCE["epidemic"]["worlds"][world["id"]]
        for key in ("mean", "variance"):
            np.testing.assert_allclose(world[key], expected[key], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(
            world["numerical"]["weak_difference"], expected["weak_difference"], rtol=1e-8, atol=1e-8
        )
    for contrast in result["contrasts"]:
        expected = REFERENCE["epidemic"]["contrasts"][contrast["id"]]
        np.testing.assert_allclose(
            contrast["mean_difference"], expected["mean_difference"], rtol=1e-8, atol=1e-8
        )
        np.testing.assert_allclose(
            contrast["monte_carlo_standard_error"], expected["standard_error"], rtol=1e-8, atol=1e-8
        )
    # Different error sources live in different fields.
    baseline = result["worlds"][0]
    assert set(baseline) >= {"monte_carlo_standard_error", "numerical", "covariance"}
    assert result["certificate"]["diffusion_lipschitz"] is None
    assert result["certificate"]["square_root_degenerate"] == [0, 1, 2]


def test_replay_is_exact_and_a_held_component_is_exact():
    system = StochasticDifferentialEquation.epidemic()
    simulation = epidemic_simulation(paths=30, steps=200, output_every=100)
    first = run(system.simulate(INITIAL, simulation, interventions=[SUPPRESS]))
    second = run(system.simulate(INITIAL, simulation, interventions=[SUPPRESS]))
    assert first == second
    held = first["simulation"]["worlds"][1]
    assert held["mean"][2][1] == 20.0 and held["variance"][2][1] == 0.0


def cir_system() -> StochasticDifferentialEquation:
    return StochasticDifferentialEquation(
        variables=("x",),
        drift=(
            E.add(
                E.multiply(E.parameter("a"), E.input("forcing")),
                E.negate(E.multiply(E.parameter("b"), E.state("x"))),
            ),
        ),
        diffusion=(
            SdeDiffusion(
                E.multiply(E.parameter("sigma"), E.parameter("sigma"), E.state("x")),
                square_root=True,
            ),
        ),
        parameters={"a": 1.5, "b": 0.5, "sigma": 0.5},
        inputs=(SdeInput("forcing", (0.0, 5.0), (1.0, 2.0)),),
        bounds={"x": SdeBound(lower=0.0)},
    )


def euler_moments(dt: float, steps: int, every: int):
    a, b, sigma = 1.5, 0.5, 0.5
    mean, var = 10.0, 0.0
    means, variances = [mean], [var]
    for n in range(steps):
        u = 2.0 if n * dt >= 5.0 else 1.0
        var, mean = (1 - b * dt) ** 2 * var + sigma**2 * mean * dt, mean + (a * u - b * mean) * dt
        if (n + 1) % every == 0:
            means.append(mean)
            variances.append(var)
    return np.array(means), np.array(variances)


def test_square_root_diffusion_matches_exact_moment_odes_with_monte_carlo_tolerance():
    cir = REFERENCE["cir"]
    paths = 4000
    result = run(
        cir_system().simulate(
            (10.0,),
            SdeSimulation(step=0.05, steps=200, paths=paths, seed=7, output_every=40),
        )
    )["simulation"]
    world = result["worlds"][0]
    mean = np.array(world["mean"])[:, 0]
    se = np.array(world["monte_carlo_standard_error"])[:, 0]
    scheme_mean, scheme_var = euler_moments(0.05, 200, 40)
    fine_mean, _ = euler_moments(0.025, 400, 80)
    exact_mean = np.array(cir["exact_mean"])
    assert np.all(np.abs(mean - scheme_mean) <= 4.5 * se + 1e-12)
    assert np.all(np.abs(scheme_mean - exact_mean) < 0.06)
    assert np.all(np.abs(fine_mean - exact_mean) <= np.abs(scheme_mean - exact_mean) + 1e-12)
    variance = np.array(world["variance"])[:, 0]
    tolerance = 6 * scheme_var * np.sqrt(2 / (paths - 1))
    assert np.all(np.abs(variance - scheme_var)[1:] <= tolerance[1:])
    numerical = world["numerical"]
    expected_difference = scheme_mean - fine_mean
    spread = np.array(numerical["weak_difference_standard_error"])[:, 0]
    assert np.all(
        np.abs(np.array(numerical["weak_difference"])[:, 0] - expected_difference)
        <= 4.5 * spread + 1e-12
    )
    assert numerical["refined_step"] == 0.025


def reference_trajectories(every: int = 1) -> list[SdeTrajectory]:
    out = []
    for k, unit in enumerate(REFERENCE["quasi_likelihood"]["units"]):
        keep = list(range(0, len(unit["times"]), every))
        out.append(
            SdeTrajectory(
                f"unit-{k}",
                tuple(unit["times"][i] for i in keep),
                tuple(tuple(unit["values"][i]) for i in keep),
            )
        )
    return out


def fit_request(every: int = 1, **changes):
    system = StochasticDifferentialEquation.epidemic(beta=0.5, amplitude=0.2, gamma=0.3)
    options = {
        "learning_rate": 0.05,
        "score_tolerance": 0.05,
        "max_iterations": 300,
        **changes,
    }
    return system.fit(
        reference_trajectories(every),
        ["beta", "amplitude", "gamma"],
        INITIAL,
        epidemic_simulation(paths=40, steps=200, output_every=100),
        interventions=[LOCKDOWN],
        contrasts=[SdeContrast("effect", "lockdown", "baseline")],
        **options,
    )


@pytest.fixture(scope="module")
def dense_fit():
    prepared = fit_request(parameter_draws=16, parameter_seed=3)
    return prepared, run(prepared)


def test_dense_trajectories_recover_the_epidemic_parameters_with_separate_evidence(dense_fit):
    _, result = dense_fit
    fit = result["fit"]
    assert fit["fitting"]["converged"], fit["fitting"]
    assert fit["names"] == ["beta", "amplitude", "gamma"]
    for value, truth, error in zip(fit["values"], TRUTH, fit["standard_errors"], strict=True):
        assert error > 0 and abs(value - truth) <= 5 * error, (value, truth, error)
    assert (
        fit["log_quasi_likelihood"] >= REFERENCE["quasi_likelihood"]["log_likelihood_truth"] - 1e-6
    )
    assert fit["identification"]["status"] != "not_identified"
    assert fit["discretization"]["status"] == "adequate"
    assert fit["discretization"]["likelihood"] == "euler_gaussian_transition"
    for block in ("fitting", "identification", "discretization"):
        assert block in fit
    assert "converged" not in fit["identification"]


def _season(time: float) -> float:
    return float(np.interp(time, [0.0, 5.0, 10.0, 15.0, 20.0], [-1.0, 0.0, 1.0, 0.0, -1.0]))


def _gaussian_fisher_information(theta: np.ndarray) -> np.ndarray:
    """Expected Fisher information of the Euler Gaussian transitions of the retained data.

    Each component of a transition is ``N(x + drift dt, variance dt)`` with ``drift`` and
    ``variance`` the epidemic flux terms, so the information of ``(beta, amplitude, gamma)`` is
    ``sum dt dd dd' / v + (1/2) dv dv' / v^2`` over components, computed here in numpy from the
    stated coefficients (population 1000) with analytic derivatives.
    """
    beta, amplitude, gamma = theta
    information = np.zeros((3, 3))
    for unit in REFERENCE["quasi_likelihood"]["units"]:
        times, states = unit["times"], unit["values"]
        for k in range(len(times) - 1):
            dt, u = times[k + 1] - times[k], _season(times[k])
            s, i = states[k][0], states[k][1]
            flux = beta * (1 + amplitude * u) * s * i / 1000.0
            d_flux = np.array([(1 + amplitude * u) * s * i / 1000.0, beta * u * s * i / 1000.0, 0.0])
            d_recovery = np.array([0.0, 0.0, i])
            d_drift = (-d_flux, d_flux - d_recovery, d_recovery)
            d_variance = (d_flux, d_flux + d_recovery, d_recovery)
            variance = (flux, flux + gamma * i, gamma * i)
            for dd, dv, v in zip(d_drift, d_variance, variance, strict=True):
                information += dt * np.outer(dd, dd) / v + 0.5 * np.outer(dv, dv) / v**2
    return information


def _mean_contrast(theta: np.ndarray) -> np.ndarray:
    """Deterministic (mean-field Euler) lockdown-minus-baseline contrast at t = 0, 5, 10."""
    beta, amplitude, gamma = theta
    paths = []
    for locked in (True, False):
        x = np.array(INITIAL)
        rows = [x.copy()]
        for n in range(200):
            u = -1.0 if locked else _season(n * 0.05)
            flux = beta * (1 + amplitude * u) * x[0] * x[1] / 1000.0
            recovery = gamma * x[1]
            x = x + np.array([-flux, flux - recovery, recovery]) * 0.05
            if (n + 1) % 100 == 0:
                rows.append(x.copy())
        paths.append(np.array(rows))
    return paths[0] - paths[1]


def test_simulation_is_at_the_fit_and_parameter_spread_is_separate(dense_fit):
    _, result = dense_fit
    assert result["simulation"]["parameters"] == result["fit"]["theta"]
    spread = result["parameter_uncertainty"]
    assert spread["completed_draws"] >= 2
    assert "excludes Monte Carlo and discretization error" in spread["meaning"]
    contrast = spread["contrasts"][0]["mean_difference_between_draw_variance"]
    assert all(v >= 0 for row in contrast for v in row)
    assert any(v > 0 for v in contrast[-1])
    assert result["simulation"]["worlds"][0]["monte_carlo_standard_error"][-1][1] > 0

    # Independent reference for the spread. The parameter draws are asymptotic-normal around the
    # fit, so the variance of a contrast mean across draws is the delta-method value g' Sigma g,
    # with Sigma the inverse Gaussian quasi-likelihood Fisher information of the retained data
    # (numpy, above) and g the sensitivity of the mean-field contrast to the parameters.
    theta = np.array(result["fit"]["theta"])
    covariance = np.linalg.inv(_gaussian_fisher_information(theta))
    # The fit's own standard errors are that information's inverse (observed ratio 0.98-1.00).
    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)), result["fit"]["standard_errors"], rtol=0.05
    )
    sensitivity = np.zeros((3, 3, 3))  # [time row, variable, parameter]
    for j in range(3):
        step = np.zeros(3)
        step[j] = 1e-5 * max(1.0, abs(theta[j]))
        sensitivity[:, :, j] = (_mean_contrast(theta + step) - _mean_contrast(theta - step)) / (
            2 * step[j]
        )
    expected = np.einsum("tvi,ij,tvj->tv", sensitivity, covariance, sensitivity)
    observed = np.array(contrast)
    assert np.all(observed[0] == 0.0), "no parameter acts before the first output"
    # A sample variance of 16 draws has a relative standard deviation of sqrt(2 / 15) = 0.37 and
    # the ensemble is stochastic where the reference is mean-field; the observed ratios are
    # 0.72-0.80 at t = 5 and 0.52-0.66 at t = 10, so a factor of 2.5 either way.
    ratio = observed[1:] / expected[1:]
    assert np.all((ratio > 0.4) & (ratio < 2.5)), ratio


def test_evidence_projects_the_quasi_likelihood_and_world_means_with_error_diagnostics(dense_fit):
    prepared, result = dense_fit
    schema = {
        "id": "sir",
        "population": "physical",
        "variables": [["S", "count"], ["I", "count"], ["R", "count"]],
    }
    selectors = [
        {"kind": "likelihood", "candidate": None},
        {"kind": "world", "world": 1, "variable": 2, "time": 2},
    ]
    columns = require_success(prepared.evidence(schema, selectors)).to_dict()["columns"]
    assert columns[0]["samples"] == [pytest.approx(result["fit"]["log_quasi_likelihood"])]
    assert set(columns[0]["diagnostics"]) >= {"fitting", "identification", "discretization"}
    assert columns[1]["samples"] == [pytest.approx(result["simulation"]["worlds"][1]["mean"][2][2])]
    diagnostics = columns[1]["diagnostics"]
    assert diagnostics["monte_carlo_standard_error"] is not None
    assert diagnostics["numerical_approximation"]["refined_step"] == 0.025
    assert diagnostics["parameter_between_draw_variance"] is not None


def test_coarse_observation_is_flagged_and_then_refused():
    flagged = run(fit_request(6))["fit"]
    assert flagged["discretization"]["status"] == "model_conditional_approximation"
    assert flagged["identification"]["status"] == "model_conditional"
    text = refusal_text(fit_request(40))
    assert "route_not_supported" in text and "sde_fit.observation_interval_too_coarse" in text


def test_unidentified_designs_are_refused_with_the_registered_reason():
    system = StochasticDifferentialEquation.epidemic()
    snapshots = [SdeTrajectory("a", (0.0,), ((950.0, 50.0, 0.0),))]
    text = refusal_text(
        system.fit(
            snapshots, ["beta"], INITIAL, epidemic_simulation(paths=4, steps=10, output_every=10)
        )
    )
    assert "dynamics_not_identified" in text and "no_transition_information" in text
    extra = StochasticDifferentialEquation.epidemic()
    extra = StochasticDifferentialEquation(
        extra.variables,
        extra.drift,
        extra.diffusion,
        {**extra.parameters, "unused": 1.0},
        extra.inputs,
        extra.bounds,
    )
    text = refusal_text(
        extra.fit(
            reference_trajectories(),
            ["beta", "amplitude", "gamma", "unused"],
            INITIAL,
            epidemic_simulation(paths=4, steps=10, output_every=10),
            score_tolerance=0.05,
            max_iterations=300,
            learning_rate=0.05,
        )
    )
    assert "dynamics_not_identified" in text and "singular_information" in text


def test_unsupported_expression_classes_and_schemes_are_refused_with_typed_reasons():
    unbounded = StochasticDifferentialEquation(
        variables=("x",),
        drift=(E.multiply(E.state("x"), E.state("x")),),
        diffusion=(SdeDiffusion(E.parameter("s")),),
        parameters={"s": 0.1},
    )
    text = refusal_text(unbounded.simulate((0.5,), SdeSimulation(0.1, 10, 4, 1)))
    assert "route_not_supported" in text and "sde.unbounded_growth" in text
    text = refusal_text(
        StochasticDifferentialEquation.epidemic().simulate(
            INITIAL, epidemic_simulation(scheme="milstein")
        )
    )
    assert "sde.milstein_noncommutative" in text
    off_grid = SdeWorld("late", actions=(SdeAction("I", (5.013,), (20.0,)),))
    text = refusal_text(
        StochasticDifferentialEquation.epidemic().simulate(
            INITIAL, epidemic_simulation(), interventions=[off_grid]
        )
    )
    assert "sde.action_time_off_grid" in text
    wide = SdeWorld("wide", inputs=(SdeInput("season", (0.0,), (2.0,), envelope=(-1.0, 2.0)),))
    text = refusal_text(
        StochasticDifferentialEquation.epidemic().simulate(
            INITIAL, epidemic_simulation(), interventions=[wide]
        )
    )
    assert "sde.input_outside_certified_envelope" in text


def test_partial_noisy_observation_is_an_extended_kalman_fit_and_says_so():
    system = StochasticDifferentialEquation.epidemic(beta=0.55, amplitude=0.3, gamma=0.3)
    truth = REFERENCE["quasi_likelihood"]["units"]
    rng = np.random.default_rng(5)
    trajectories = [
        SdeTrajectory(
            f"u{k}",
            tuple(unit["times"][::10]),
            tuple((row[1] + 5.0 * rng.standard_normal(),) for row in unit["values"][::10]),
        )
        for k, unit in enumerate(truth)
    ]
    prepared = system.fit(
        trajectories,
        ["beta", "gamma"],
        INITIAL,
        epidemic_simulation(paths=8, steps=100, output_every=50),
        observation=SdeObservationModel((0.0, 1.0, 0.0), (5.0,)),
        filter_prior=((950.0, 50.0, 0.0), (25.0, 0, 0, 0, 25.0, 0, 0, 0, 1.0)),
        filter_step=0.05,
        score_tolerance=0.05,
    )
    fit = run(prepared)["fit"]
    assert fit["discretization"]["likelihood"] == "extended_kalman_euler_innovation"
    assert fit["identification"]["unobserved_components"] == ["S", "R"]
    assert "extended_kalman_linearization" in fit["identification"]["conditions"]
    assert fit["identification"]["status"] == "model_conditional"
