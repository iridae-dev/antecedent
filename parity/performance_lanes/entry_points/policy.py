#!/usr/bin/env python3
"""First answer for held-out policy evaluation.

One `evaluate_policy_doubly_robust` call scores a fixed binary treat/control recommendation
on randomized held-out subjects with the doubly robust row scores. It returns the policy,
reference and incremental values with their standard errors and 95% intervals, and checks the
training/evaluation ownership of the supplied outcome predictions. The nuisance predictions are
caller inputs fitted on the training half before the timer starts, as is data synthesis.

    python parity/performance_lanes/entry_points/policy.py --role control
    python parity/performance_lanes/entry_points/policy.py --role candidate
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _entry import arguments, digest_arrays, run_entry
from antecedent import policy

DATA_SEED = 13
HALF = 5000
PROPENSITY = 0.5
TREATMENT_COST = 0.2


def design(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def synthesize(half: int) -> dict[str, Any]:
    rng = np.random.default_rng(DATA_SEED)
    n = 2 * half
    x = rng.normal(size=(n, 2))
    assignment = rng.uniform(size=n) < PROPENSITY
    effect = 1.5 * x[:, 1]
    outcome = 1.0 + x[:, 0] + effect * assignment + 0.5 * rng.normal(size=n)
    train, held = slice(0, half), slice(half, n)
    mu = []
    for arm in (False, True):
        rows = np.flatnonzero(assignment[train] == arm)
        beta, *_ = np.linalg.lstsq(design(x[train][rows]), outcome[train][rows], rcond=None)
        mu.append(design(x[held]) @ beta)
    return {
        "outcome": outcome[held],
        "assignment": [bool(a) for a in assignment[held]],
        "mu0": mu[0],
        "mu1": mu[1],
        "recommended": [bool(r) for r in (mu[1] - mu[0]) > TREATMENT_COST],
        "evaluation_ids": [f"e{i}" for i in range(half)],
        "training_ids": [f"t{i}" for i in range(half)],
    }


def main() -> None:
    args = arguments(__doc__ or "")
    inputs = synthesize((args.rows or 2 * HALF) // 2)
    recommended = policy.BinaryPolicy(inputs["recommended"], costs=TREATMENT_COST)
    frame = {"y": inputs["outcome"]}

    def section() -> Any:
        return policy.evaluate_policy_doubly_robust(
            frame,
            outcome="y",
            assignment=inputs["assignment"],
            propensity=PROPENSITY,
            mu0=inputs["mu0"],
            mu1=inputs["mu1"],
            policy=recommended,
            evaluation_subject_ids=inputs["evaluation_ids"],
            training_subject_ids=inputs["training_ids"],
        )

    def fields(result: Any) -> dict[str, Any]:
        return {
            "policy_value": result.policy_value,
            "policy_value_standard_error": result.policy_value_standard_error,
            "policy_value_interval": result.policy_value_interval_95,
            "incremental_value": result.incremental_value,
            "incremental_value_standard_error": result.incremental_value_standard_error,
            "incremental_value_interval": result.incremental_value_interval_95,
            "treatment_rate": result.treatment_rate,
            "prediction_ownership": result.prediction_ownership,
            "propensity_range": (result.propensity_min, result.propensity_max),
            "uncertainty": result.uncertainty,
        }

    run_entry(
        "policy",
        args,
        digest_arrays(
            {
                "y": inputs["outcome"],
                "assignment": np.asarray(inputs["assignment"], dtype=np.float64),
                "mu0": inputs["mu0"],
                "mu1": inputs["mu1"],
                "recommended": np.asarray(inputs["recommended"], dtype=np.float64),
            }
        ),
        section,
        fields,
        required=("policy_value_interval", "incremental_value_interval", "prediction_ownership"),
    )


if __name__ == "__main__":
    main()
