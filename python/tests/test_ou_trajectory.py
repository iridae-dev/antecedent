# SPDX-License-Identifier: MIT OR Apache-2.0
"""Continuous-time OU fitting from continuous trajectories against independent references.

Data are simulated by exact discretization in NumPy/SciPy (a different code path from the
native matrix-exponential Kalman likelihood), and the likelihood is checked against the
brute-force joint Gaussian density of all observation times.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.cyclic import (
    OuMechanisms,
    OuObservationModel,
    OuPredictionDesign,
    OuScheduleSegment,
    OuStationaryDesign,
    OuTrajectory,
)
from antecedent.outcomes import require_success

scipy_linalg = pytest.importorskip("scipy.linalg")

A = np.array([[-1.0, 0.5], [-0.4, -0.8]])
B = np.array([0.6, -0.3])
L = np.array([[0.5, 0.0], [0.2, 0.4]])
TRUTH = [-1.0, 0.5, -0.4, -0.8, 0.6, -0.3, 0.5, 0.2, 0.4]
COORDINATES = [
    {"kind": "drift", "equation": 0, "parent": 0},
    {"kind": "drift", "equation": 0, "parent": 1},
    {"kind": "drift", "equation": 1, "parent": 0},
    {"kind": "drift", "equation": 1, "parent": 1},
    {"kind": "forcing", "equation": 0},
    {"kind": "forcing", "equation": 1},
    {"kind": "diffusion", "equation": 0, "source": 0},
    {"kind": "diffusion", "equation": 1, "source": 0},
    {"kind": "diffusion", "equation": 1, "source": 1},
]


def start_model() -> OuMechanisms:
    return OuMechanisms(("a", "b"), (-0.7, 0.2, -0.1, -1.1), (0.3, 0.0), (0.3, 0.0, 0.1, 0.3))


def simulate_exact(seed: int, units: int, per_unit: int, observe=None, noise: float = 0.0):
    """Irregular-interval stationary trajectories drawn by exact Gaussian transitions."""
    rng = np.random.default_rng(seed)
    mean = -np.linalg.solve(A, B)
    stationary = scipy_linalg.solve_continuous_lyapunov(A, -(L @ L.T))
    trajectories = []
    for unit in range(units):
        gaps = rng.uniform(0.3, 1.2, size=per_unit)
        times = np.cumsum(gaps)
        state = rng.multivariate_normal(mean, stationary)
        previous = 0.0
        rows = []
        for time in times:
            phi = scipy_linalg.expm(A * (time - previous))
            cov = stationary - phi @ stationary @ phi.T
            state = mean + phi @ (state - mean) + rng.multivariate_normal(np.zeros(2), cov)
            previous = time
            seen = state if observe is None else observe @ state
            rows.append(tuple(seen + noise * rng.standard_normal(len(seen))))
        trajectories.append(OuTrajectory(f"unit-{unit}", tuple(times), tuple(rows)))
    return trajectories


def joint_gaussian_loglik(model, observation, trajectories) -> float:
    """Brute-force density of all observation times jointly (no recursion)."""
    drift = np.array(model["drift"]).reshape(2, 2)
    forcing = np.array(model["forcing"])
    factor = np.array(model["diffusion_factor"]).reshape(2, 2)
    h = np.array(observation.matrix).reshape(len(observation.noise_standard_deviation), 2)
    noise = np.array(observation.noise_standard_deviation) ** 2
    mean = -np.linalg.solve(drift, forcing)
    stationary = scipy_linalg.solve_continuous_lyapunov(drift, -(factor @ factor.T))
    total = 0.0
    for trajectory in trajectories:
        times = np.array(trajectory.times)
        m = h.shape[0]
        k = len(times)
        cov = np.zeros((k * m, k * m))
        for i in range(k):
            for j in range(k):
                if i >= j:
                    block = scipy_linalg.expm(drift * (times[i] - times[j])) @ stationary
                else:
                    block = (scipy_linalg.expm(drift * (times[j] - times[i])) @ stationary).T
                cov[i * m : (i + 1) * m, j * m : (j + 1) * m] = h @ block @ h.T
            cov[i * m : (i + 1) * m, i * m : (i + 1) * m] += np.diag(noise)
        y = (np.array(trajectory.values) - h @ mean).reshape(-1)
        _, logdet = np.linalg.slogdet(cov)
        total += -0.5 * (len(y) * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(cov, y))
    return float(total)


def run(prepared) -> dict:
    return require_success(prepared.execute()).to_dict()["answer"]["result"]


@pytest.fixture(scope="module")
def full_fit():
    trajectories = simulate_exact(7, 4, 80)
    held_out = OuPredictionDesign(
        "hold-out",
        (0.5, 1.0, 1.5, 2.5, 4.0),
        schedule=(
            OuScheduleSegment(0.0),
            OuScheduleSegment(
                1.0,
                (
                    {
                        "equation": "a",
                        "drift": [-1.5, 0.0],
                        "forcing": 0.4,
                        "diffusion_factor": [0.3, 0.0],
                    },
                ),
            ),
        ),
        initial={"kind": "gaussian", "mean": [0.3, -0.2], "covariance": [0.2, 0.05, 0.05, 0.1]},
    )
    sustained = OuStationaryDesign(
        "sustained",
        (
            {
                "equation": "a",
                "drift": [-1.5, 0.0],
                "forcing": 0.4,
                "diffusion_factor": [0.3, 0.0],
            },
        ),
    )
    prepared = start_model().fit_trajectories(
        trajectories,
        COORDINATES,
        predictions=[held_out],
        stationary_worlds=[sustained],
        learning_rate=0.05,
        score_tolerance=1e-3,
        max_iterations=400,
    )
    return trajectories, prepared, run(prepared)


def test_irregular_trajectories_recover_the_parameters_with_separate_evidence(full_fit):
    _, _, result = full_fit
    fit = result["fit"]
    assert fit["fitting"]["converged"], fit["fitting"]
    for value, truth, error in zip(fit["values"], TRUTH, fit["standard_errors"], strict=True):
        assert abs(value - truth) <= 4 * error + 0.02, (value, truth, error)
    # Three distinct records: optimizer termination, identification, numerical stability.
    assert fit["identification"]["status"] == "identified"
    assert fit["identification"]["observability_rank"] == 2
    assert fit["numerics"]["drift_stable"] is True
    assert fit["numerics"]["expm_reliable"] is True
    assert "converged" not in fit["identification"]
    assert "status" not in fit["fitting"]


def test_native_likelihood_equals_the_brute_force_joint_gaussian(full_fit):
    trajectories, _, result = full_fit
    fit = result["fit"]
    reference = joint_gaussian_loglik(fit["model"], OuObservationModel.direct(2), trajectories)
    assert fit["log_likelihood"] == pytest.approx(reference, abs=1e-6)


def test_partial_observation_matches_the_joint_density_and_reports_the_unobserved_state():
    observe = np.array([[1.0, 0.0]])
    trajectories = simulate_exact(21, 3, 60, observe=observe, noise=0.05)
    observation = OuObservationModel((1.0, 0.0), (0.05,))
    model = OuMechanisms(("a", "b"), (-0.9, 0.4, -0.3, -0.9), (0.6, -0.3), (0.5, 0.0, 0.2, 0.4))
    coordinates = [
        {"kind": "drift", "equation": 0, "parent": 0},
        {"kind": "drift", "equation": 0, "parent": 1},
        {"kind": "drift", "equation": 1, "parent": 0},
        {"kind": "drift", "equation": 1, "parent": 1},
    ]
    result = run(
        model.fit_trajectories(
            trajectories,
            coordinates,
            observation=observation,
            learning_rate=0.05,
            score_tolerance=1e-3,
            max_iterations=400,
        )
    )
    fit = result["fit"]
    assert fit["identification"]["unobserved_components"] == ["b"]
    assert fit["identification"]["observability_rank"] == 2
    assert fit["identification"]["status"] != "not_identified"
    assert len(fit["filtered_states"]) == 3
    for state in fit["filtered_states"]:
        assert state["standard_deviation"][1] > state["standard_deviation"][0]
    reference = joint_gaussian_loglik(fit["model"], observation, trajectories)
    assert fit["log_likelihood"] == pytest.approx(reference, abs=1e-6)


def test_held_out_intervention_trajectory_matches_independent_simulation(full_fit):
    _, _, result = full_fit
    prediction = result["predictions"][0]
    rng = np.random.default_rng(99)
    replicates, dt = 4000, 0.004
    mean0 = np.array([0.3, -0.2])
    cov0 = np.array([[0.2, 0.05], [0.05, 0.1]])
    state = rng.multivariate_normal(mean0, cov0, size=replicates)
    baseline = (A, B, L)
    replaced = (np.array([[-1.5, 0.0], A[1]]), np.array([0.4, B[1]]), np.array([[0.3, 0.0], L[1]]))
    observe = {round(t / dt): i for i, t in enumerate(prediction["times"])}
    simulated_mean = np.zeros((len(observe), 2))
    simulated_var = np.zeros((len(observe), 2))
    for step in range(round(4.0 / dt) + 1):
        if step in observe:
            simulated_mean[observe[step]] = state.mean(axis=0)
            simulated_var[observe[step]] = state.var(axis=0)
        drift, forcing, factor = replaced if step >= round(1.0 / dt) else baseline
        noise = rng.standard_normal((replicates, 2)) @ factor.T
        state = state + (state @ drift.T + forcing) * dt + np.sqrt(dt) * noise
    for t in range(len(observe)):
        for c in range(2):
            parameter_var = prediction["state_mean_parameter_covariance"][t][c * 2 + c]
            process_var = prediction["state_covariance"][t][c * 2 + c]
            assert parameter_var > 0.0
            slack = 4 * np.sqrt(parameter_var + process_var / replicates) + 0.02
            assert abs(prediction["state_mean"][t][c] - simulated_mean[t, c]) <= slack
            covariance_var = prediction["state_covariance_parameter_variance"][t][c * 2 + c]
            variance_slack = (
                4 * (np.sqrt(covariance_var) + process_var * np.sqrt(2 / replicates)) + 0.01
            )
            assert abs(process_var - simulated_var[t, c]) <= variance_slack
    # The sustained replacement of `a` is autonomous: its stationary law is closed form.
    sustained = result["stationary"][0]
    assert sustained["failure"] is None
    assert sustained["mean"][0] == pytest.approx(0.4 / 1.5, abs=1e-9)
    assert sustained["covariance"][0] == pytest.approx(0.09 / 3.0, abs=1e-9)


def test_snapshots_with_no_transition_are_refused_as_unidentified():
    snapshots = [
        OuTrajectory(item.id, item.times[:1], item.values[:1]) for item in simulate_exact(31, 6, 3)
    ]
    prepared = start_model().fit_trajectories(snapshots, COORDINATES)
    with pytest.raises(Exception, match="dynamics_not_identified") as refused:
        require_success(prepared.execute())
    assert "no_transition_information" in str(refused.value)


def test_a_coordinate_the_observations_cannot_determine_is_refused_as_unidentified():
    rng = np.random.default_rng(41)
    trajectories = []
    for unit in range(3):
        times = np.cumsum(rng.uniform(0.3, 1.2, size=60))
        phi = np.exp(-1.0 * np.diff(times, prepend=0.0))
        state, rows = 0.5, []
        for p in phi:
            variance = 0.25 / 2.0 * (1.0 - p * p)
            state = 0.5 + p * (state - 0.5) + np.sqrt(variance) * rng.standard_normal()
            rows.append((state,))
        trajectories.append(OuTrajectory(f"u{unit}", tuple(times), tuple(rows)))
    observation = OuObservationModel((1.0, 0.0), (0.0,))
    coordinates = [
        {"kind": "drift", "equation": 0, "parent": 0},
        {"kind": "forcing", "equation": 0},
        {"kind": "drift", "equation": 1, "parent": 1},
    ]
    prepared = OuMechanisms(
        ("a", "b"), (-0.7, 0.0, 0.0, -2.0), (0.3, 0.0), (0.5, 0.0, 0.0, 0.3)
    ).fit_trajectories(trajectories, coordinates, observation=observation, score_tolerance=1e-3)
    with pytest.raises(Exception, match="dynamics_not_identified") as refused:
        require_success(prepared.execute())
    assert "singular_information" in str(refused.value)


def test_evidence_keeps_fitting_identification_and_numerics_and_marks_non_convergence():
    trajectories = simulate_exact(51, 2, 40)
    schema = {"id": "ou", "population": "physical", "variables": [["a", "unit"], ["b", "unit"]]}
    selectors = [{"kind": "likelihood", "candidate": None}]
    bounded = start_model().fit_trajectories(trajectories, COORDINATES, max_iterations=1)
    marked = require_success(bounded.evidence(schema, selectors)).to_dict()["columns"][0]
    assert marked["diagnostics"]["fit_convergence"]["converged"] is False
    assert set(marked["diagnostics"]) >= {"fitting", "identification", "numerics"}
    assert run(bounded)["fit"]["fitting"]["converged"] is False


def test_predictions_are_evidence_selectors_with_separate_sampling_uncertainty(full_fit):
    _, prepared, result = full_fit
    schema = {"id": "ou", "population": "physical", "variables": [["a", "unit"], ["b", "unit"]]}
    selectors = [
        {"kind": "ou_trajectory_state", "design": 0, "time": 2, "variable": 0, "other": None},
        {"kind": "ou_trajectory_state", "design": 0, "time": 2, "variable": 0, "other": 0},
        {"kind": "ou_trajectory_stationary", "world": 0, "variable": 0, "other": None},
    ]
    columns = require_success(prepared.evidence(schema, selectors)).to_dict()["columns"]
    prediction = result["predictions"][0]
    assert columns[0]["samples"] == [pytest.approx(prediction["state_mean"][2][0])]
    assert columns[1]["samples"] == [pytest.approx(prediction["state_covariance"][2][0])]
    assert columns[2]["samples"] == [pytest.approx(result["stationary"][0]["mean"][0])]
    expected = prediction["state_mean_parameter_covariance"][2][0]
    assert columns[0]["diagnostics"]["parameter_variance"] == pytest.approx(expected)
    assert "never added" in columns[0]["diagnostics"]["meaning"]
