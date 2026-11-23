# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.1.3", "scipy==1.14.1"]
# ///
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent reference for the exact-transition likelihood of an observed OU process.

For ``dX = (A X + b) dt + L dW`` observed as ``y = H X + noise`` at irregular times, the
stationary process is Gaussian, so the density of all observations is one joint Gaussian
whose covariance blocks are ``H expm(A (t_i - t_j)) S H^T`` (``S`` the stationary covariance
from SciPy's continuous Lyapunov solver) plus the noise on the diagonal. The truth here is
that brute-force joint density: no Kalman recursion, no van Loan transition and no shared
code with the production implementation. Data are drawn by exact discretization with a
seeded NumPy generator and stored, so the Rust consumer evaluates the very same numbers.

    uv run --script conformance/gcm/ou_trajectory_likelihood/reference.py          # write
    uv run --frozen --script conformance/gcm/ou_trajectory_likelihood/reference.py --check  # compare
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import linalg

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected.json"

CASES = (
    {
        "case": "fully_observed_irregular",
        "seed": 11,
        "drift": [[-1.0, 0.5], [-0.4, -0.8]],
        "forcing": [0.6, -0.3],
        "diffusion_factor": [[0.5, 0.0], [0.2, 0.4]],
        "observation": [[1.0, 0.0], [0.0, 1.0]],
        "noise_standard_deviation": [0.0, 0.0],
        "units": 3,
        "per_unit": 25,
    },
    {
        "case": "partially_observed_noisy",
        "seed": 12,
        "drift": [[-1.0, 0.5], [-0.4, -0.8]],
        "forcing": [0.6, -0.3],
        "diffusion_factor": [[0.5, 0.0], [0.2, 0.4]],
        "observation": [[1.0, 0.0]],
        "noise_standard_deviation": [0.05],
        "units": 3,
        "per_unit": 25,
    },
    {
        "case": "oscillatory_combination",
        "seed": 13,
        "drift": [[-0.2, -3.0], [3.0, -0.2]],
        "forcing": [0.1, 0.0],
        "diffusion_factor": [[0.6, 0.0], [0.0, 0.6]],
        "observation": [[1.0, 1.0]],
        "noise_standard_deviation": [0.1],
        "units": 2,
        "per_unit": 30,
    },
)


def joint_loglik(a, b, factor, h, noise_sd, times, values) -> float:
    mean = -np.linalg.solve(a, b)
    stationary = linalg.solve_continuous_lyapunov(a, -(factor @ factor.T))
    m, k = h.shape[0], len(times)
    cov = np.zeros((k * m, k * m))
    for i in range(k):
        for j in range(k):
            if i >= j:
                block = linalg.expm(a * (times[i] - times[j])) @ stationary
            else:
                block = (linalg.expm(a * (times[j] - times[i])) @ stationary).T
            cov[i * m : (i + 1) * m, j * m : (j + 1) * m] = h @ block @ h.T
        cov[i * m : (i + 1) * m, i * m : (i + 1) * m] += np.diag(np.square(noise_sd))
    y = (values - h @ mean).reshape(-1)
    _, logdet = np.linalg.slogdet(cov)
    return float(-0.5 * (len(y) * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(cov, y)))


def solve_case(spec: dict) -> dict:
    a = np.array(spec["drift"])
    b = np.array(spec["forcing"])
    factor = np.array(spec["diffusion_factor"])
    h = np.array(spec["observation"])
    noise_sd = np.array(spec["noise_standard_deviation"])
    rng = np.random.default_rng(spec["seed"])
    mean = -np.linalg.solve(a, b)
    stationary = linalg.solve_continuous_lyapunov(a, -(factor @ factor.T))
    units, total = [], 0.0
    for _ in range(spec["units"]):
        times = np.round(np.cumsum(rng.uniform(0.2, 1.1, size=spec["per_unit"])), 6)
        state = rng.multivariate_normal(mean, stationary)
        previous, rows = 0.0, []
        for t in times:
            phi = linalg.expm(a * (t - previous))
            cov = stationary - phi @ stationary @ phi.T
            state = mean + phi @ (state - mean) + rng.multivariate_normal(np.zeros(2), cov)
            previous = t
            rows.append(h @ state + noise_sd * rng.standard_normal(h.shape[0]))
        values = np.array(rows)
        total += joint_loglik(a, b, factor, h, noise_sd, times, values)
        units.append({"times": times.tolist(), "values": values.tolist()})
    return {**spec, "units_data": units, "log_likelihood": total}


REVIEW_OWNED = ("reference_review", "executing_assertion")


def committed_review() -> dict:
    """The reviewed-artifact record and the assertion path live in the committed fixture."""
    if not EXPECTED.is_file():
        return {}
    oracle = json.loads(EXPECTED.read_text(encoding="utf-8")).get("oracle", {})
    return {k: oracle[k] for k in REVIEW_OWNED if k in oracle}


def build() -> dict:
    document = {
        "oracle": {
            "kind": "independent_reimplementation",
            "method": "brute-force joint Gaussian density of all observation times, with SciPy expm and continuous Lyapunov solutions",
            "independence_basis": (
                "The stationary OU process is jointly Gaussian; its observation covariance "
                "blocks are H expm(A dt) S H^T plus noise, evaluated densely with NumPy/SciPy "
                "in one multivariate-normal density per unit. No Kalman recursion, van Loan "
                "transition or other production code is used. The retained generator and the "
                "executing assertion were read for the review recorded in reference_review."
            ),
            "generator": "conformance/gcm/ou_trajectory_likelihood/reference.py",
            "generator_retained": True,
            "environment_lock": "conformance/gcm/ou_trajectory_likelihood/reference.py.lock",
            "command": "uv run --frozen --script conformance/gcm/ou_trajectory_likelihood/reference.py --check",
            "asserted_outputs": ["cases[*].log_likelihood (summed over the units of each case)"],
            "context_only_outputs": ["seed", "per_unit", "units"],
            "historical_exposure": "Historical author exposure is unrecorded and remains unknown.",
        },
        "schema_version": 1,
        "cases": [solve_case(spec) for spec in CASES],
    }
    document["oracle"].update(committed_review())
    return document


def main() -> int:
    document = build()
    if "--check" in sys.argv[1:]:
        committed = json.loads(EXPECTED.read_text(encoding="utf-8"))
        problems = []
        stored_oracle = {k: v for k, v in committed["oracle"].items() if k not in REVIEW_OWNED}
        fresh_oracle = {k: v for k, v in document["oracle"].items() if k not in REVIEW_OWNED}
        if fresh_oracle != stored_oracle:
            problems.append("oracle record differs")
        if len(document["cases"]) != len(committed["cases"]):
            problems.append("case count")
        for fresh, stored in zip(document["cases"], committed["cases"], strict=False):
            # Stored data are evaluated, not regenerated: compare the likelihood of the stored data.
            a, b = np.array(stored["drift"]), np.array(stored["forcing"])
            factor, h = np.array(stored["diffusion_factor"]), np.array(stored["observation"])
            sd = np.array(stored["noise_standard_deviation"])
            total = sum(
                joint_loglik(a, b, factor, h, sd, np.array(u["times"]), np.array(u["values"]))
                for u in stored["units_data"]
            )
            if not np.isclose(total, stored["log_likelihood"], rtol=1e-9, atol=0):
                problems.append(f"{fresh['case']}.log_likelihood")
        for problem in problems:
            print(f"ou trajectory reference mismatch: {problem}", file=sys.stderr)
        return 1 if problems else 0
    EXPECTED.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
