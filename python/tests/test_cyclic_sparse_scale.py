# SPDX-License-Identifier: MIT OR Apache-2.0
"""Sparse feedback systems with hundreds of variables against a NumPy/SciPy reference."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from antecedent.cyclic import Equation, EquilibriumModel, SolvePolicy
from antecedent.outcomes import require_success

N = 300


def ring_model(
    predecessor: float, successor: float, own: float
) -> tuple[EquilibriumModel, sp.csr_matrix]:
    rows, cols, vals = [], [], []
    equations = []
    for i in range(N):
        weights = [0.0] * N
        weights[(i - 1) % N] += predecessor
        weights[(i + 1) % N] += successor
        weights[i] += own
        equations.append(Equation(f"v{i}", 0.0, tuple(weights), "identity"))
        for j, w in enumerate(weights):
            if w != 0.0:
                rows.append(i)
                cols.append(j)
                vals.append(w)
    covariance = np.eye(N).ravel().tolist()
    weight_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return EquilibriumModel(tuple(equations), tuple(covariance)), weight_matrix


def noise_row() -> list[float]:
    return [float(np.sin(0.37 * i)) for i in range(N)]


def reference(
    weight_matrix: sp.csr_matrix, noise: list[float], clamps: dict[int, float]
) -> np.ndarray:
    a = (sp.identity(N, format="lil") - weight_matrix.tolil()).tolil()
    rhs = np.array(noise, dtype=float)
    for variable, value in clamps.items():
        a[variable, :] = 0.0
        a[variable, variable] = 1.0
        rhs[variable] = value
    return spla.spsolve(a.tocsc(), rhs)


def test_several_hundred_variable_ring_matches_scipy_with_evidence_and_interventions() -> None:
    model, weights = ring_model(0.35, 0.25, 0.1)
    clamp = {7: 2.0, 123: -1.0}
    prepared = model.prepare(
        [[], [{"variable": v, "value": x} for v, x in clamp.items()]],
        [noise_row()],
    )
    result = require_success(prepared.execute()).to_dict()
    free, intervened = result["worlds"][0][0], result["worlds"][1][0]
    assert np.max(np.abs(np.array(free["values"]) - reference(weights, noise_row(), {}))) < 1e-8
    assert (
        np.max(np.abs(np.array(intervened["values"]) - reference(weights, noise_row(), clamp)))
        < 1e-8
    )
    evidence = free["evidence"]
    assert evidence["free_variables"] == N and evidence["largest_block"] == N
    assert evidence["jacobian_nnz"] == 3 * N
    assert evidence["jacobian_nonsingular_basis"] == "contraction"
    assert evidence["unique_root_basis"] == "contraction"
    assert evidence["contraction_bound"] == pytest.approx(0.7, abs=1e-9)
    assert evidence["admitted_bytes"] > 0 and evidence["budget_checks"] > 0
    assert evidence["ill_conditioned"] is False
    assert free["globally_unique"] is True
    assert free["jacobian_condition_estimate"] > 1.0
    assert intervened["evidence"]["free_variables"] == N - len(clamp)


def test_noncontractive_ring_needs_newton_and_reports_the_affine_certificate() -> None:
    model, weights = ring_model(1.5, 0.0, 0.0)
    with pytest.raises(Exception, match="contractive"):
        require_success(model.prepare([[]], [noise_row()]).execute())
    prepared = model.prepare([[]], [noise_row()], policy=SolvePolicy(require_contraction=False))
    world = require_success(prepared.execute()).to_dict()["worlds"][0][0]
    assert np.max(np.abs(np.array(world["values"]) - reference(weights, noise_row(), {}))) < 1e-8
    assert world["evidence"]["newton_steps"] >= 1
    assert world["evidence"]["unique_root_basis"] == "affine_bijection"


@pytest.mark.parametrize(
    ("policy", "code"),
    [
        (SolvePolicy(max_nnz=10), "solver_jacobian_nnz_budget"),
        (SolvePolicy(max_block_size=50), "solver_block_size_budget"),
        (SolvePolicy(max_bytes=1000), "solver_memory_budget"),
        (SolvePolicy(max_total_iterations=1), "solver_iteration_budget"),
        (SolvePolicy(max_wall_clock_ms=0), "solver_wall_clock_budget"),
    ],
)
def test_exceeded_budgets_are_refused_with_distinct_codes(policy: SolvePolicy, code: str) -> None:
    model, _ = ring_model(0.35, 0.25, 0.1)
    with pytest.raises(Exception, match=code):
        require_success(model.prepare([[]], [noise_row()], policy=policy).execute())


def test_fill_budget_and_ill_conditioning_are_flagged() -> None:
    model, _ = ring_model(1.5, 0.0, 0.0)
    with pytest.raises(Exception, match="solver_factor_fill_budget"):
        require_success(
            model.prepare(
                [[]],
                [noise_row()],
                policy=SolvePolicy(require_contraction=False, max_factor_fill=300),
            ).execute()
        )
    near, _ = ring_model(1.0 + 1e-13, 0.0, 0.0)
    zero = [0.0] * N
    world = require_success(
        near.prepare([[]], [zero], policy=SolvePolicy(require_contraction=False)).execute()
    ).to_dict()["worlds"][0][0]
    assert world["jacobian_condition_estimate"] > 1e9
    assert world["evidence"]["ill_conditioned"] is True
