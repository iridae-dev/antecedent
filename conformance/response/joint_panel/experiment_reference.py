"""Independent experiment-to-decision arithmetic, using only the standard library.

The finite prior and paired affine translations are mathematical input, not a
refit. Gaussian future-data integration returns an explicit absolute error bound.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def translated_panel_mean(groups, doses, duration, population_weights, time_weights):
    """Finite-lag structural estimand with independently supplied target mix/window.

    Histories, scenarios, saturation and lag coefficients are fixed known truth.
    This integrates response noise; it does not infer transport from dose ratios.
    """
    if len(groups) != len(population_weights) or duration != len(time_weights):
        raise ValueError("population/time axes do not match the declared target")
    if min(population_weights) < 0 or sum(population_weights) <= 0:
        raise ValueError("invalid target population weights")
    total = 0.0
    for group, mass in zip(groups, population_weights, strict=False):
        a = list(group["initial_a"])
        b = list(group["initial_b"])
        for t in range(duration):
            a.append(doses[0])
            b.append(doses[1])
            ea = 0.6 * a[-1] + 0.3 * a[-2] + 0.1 * a[-3]
            eb = 0.5 * b[-1] + 0.35 * b[-2] + 0.15 * b[-3]
            sa, sb = ea / (ea + 2.0), eb / (eb + 2.0)
            value = (
                2.0
                + group["intercept"]
                + (0.0, 0.2, -0.1, 0.1)[t % 4]
                + 0.7 * group["periods"][t]["z"]
                + 3.0 * sa
                + 2.0 * sb
                + 1.5 * sa * sb
            )
            total += mass * time_weights[t] * value
    return total / sum(population_weights)


def posterior(prior, targets, intercepts, slopes, observed, variance, sample_size):
    """Likelihood ratios expanded independently of the production log-density."""
    if variance <= 0 or sample_size <= 0 or any(s == 0 for s in slopes):
        raise ValueError("invalid Gaussian experiment or noninvertible translation")
    means = [(v - a) / b for v, a, b in zip(targets, intercepts, slopes, strict=False)]
    anchor = min(means, key=lambda m: abs(m - observed))
    mass = [
        p * math.exp((m - anchor) * (2 * observed - m - anchor) * sample_size / (2 * variance))
        for p, m in zip(prior, means, strict=False)
    ]
    normalizer = math.fsum(mass)
    return [p / normalizer for p in mass]


def decision(weights, utilities, feasible):
    """Enumerate all feasible action values; feasibility is not a utility penalty."""
    values = [math.fsum(w * u for w, u in zip(weights, row, strict=False)) for row in utilities]
    admitted = [i for i, ok in enumerate(feasible) if ok]
    if not admitted:
        raise ValueError("no feasible terminal action")
    best = max(values[i] for i in admitted)
    perfect = math.fsum(w * max(utilities[a][i] for a in admitted) for i, w in enumerate(weights))
    return {
        "values": values,
        "chosen": [i for i in admitted if values[i] == best],
        "expected_utility": best,
        "evpi": perfect - best,
    }


def future_value(weights, means, utilities, feasible, variance, sample_size, cells=4096):
    """Nested predictive Gaussian integration with total-variation and Mills bounds.

    Integrate each action's unnormalized posterior utility, then take its maximum.
    Subtracting a state-wise common baseline improves the bound without changing
    EVSI. Every mixture mean is at least eight SDs inside the integration range.
    """
    admitted = [a for a, ok in enumerate(feasible) if ok]
    if not admitted or cells < 1 or variance <= 0 or sample_size <= 0:
        raise ValueError("invalid future study or action set")
    shifted = [
        [utilities[a][i] - utilities[admitted[0]][i] for i in range(len(weights))] for a in admitted
    ]
    sigma = math.sqrt(variance / sample_size)
    lower, upper = min(means) - 8 * sigma, max(means) + 8 * sigma
    step = (upper - lower) / cells
    density_scale = 1 / (sigma * math.sqrt(2 * math.pi))
    integrated = 0.0
    for cell in range(cells):
        y = lower + (cell + 0.5) * step
        likelihood = [density_scale * math.exp(-0.5 * ((y - m) / sigma) ** 2) for m in means]
        integrated += step * max(
            math.fsum(
                w * density * u for w, density, u in zip(weights, likelihood, row, strict=False)
            )
            for row in shifted
        )
    current = max(math.fsum(w * u for w, u in zip(weights, row, strict=False)) for row in shifted)
    absolute_utility = math.fsum(
        max(abs(row[i]) for row in shifted) * w for i, w in enumerate(weights)
    )
    # max is Lipschitz in the infinity norm; sum over absolute mixture derivatives
    # bounds total variation of max_a(sum_i w_i phi_i(y) u_ai).
    variation = absolute_utility * math.sqrt(2 / math.pi) / sigma
    tails = absolute_utility * 2 * math.exp(-32) / (8 * math.sqrt(2 * math.pi))
    return {
        "gross_evsi": integrated - current,
        "absolute_error_bound": step * variation / 2 + tails + 1e-12,
        "cells": cells,
        "tail_standard_deviations": 8,
    }


def reference_artifact():
    prior, targets = [0.2, 0.5, 0.3], [-1.0, 0.5, 2.0]
    intercepts, slopes = [-0.1, 0.0, 0.1], [0.8, 1.0, 1.2]
    weights = posterior(prior, targets, intercepts, slopes, 0.7, 0.6, 3)
    utilities = [[0.0] * 3, [v - 0.6 for v in targets], [v - 0.2 for v in targets]]
    feasible = [True, True, False]
    means = [(v - a) / b for v, a, b in zip(targets, intercepts, slopes, strict=False)]
    future = future_value(weights, means, utilities, feasible, 1.0, 4)
    truth = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    groups = truth["variants"]["interaction"]["latent_truth"]
    estimands = []
    for duration in (2, 12, 24):
        for dose in (1.0, 4.0):
            for mix in ([1.0] * 4, [0.1, 0.2, 0.3, 0.4]):
                value = translated_panel_mean(
                    groups, (dose, dose), duration, mix, [1 / duration] * duration
                )
                estimands.append(
                    {
                        "duration": duration,
                        "dose": dose,
                        "population_weights": mix,
                        "mean": value,
                    }
                )
    return {
        "schema": "antecedent.independent_experiment_reference.v1",
        "semantics": "frozen_modular_finite_law_not_joint_refit",
        "posterior_weights": weights,
        "decision": decision(weights, utilities, feasible),
        "future_study": future,
        "net_values": [future["gross_evsi"] - c for c in (0.01, 0.1)],
        "population_time_dose_estimands": estimands,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(reference_artifact(), indent=2, sort_keys=True) + "\n"
    target = HERE / "experiment_reference.json"
    if args.write:
        target.write_text(rendered, encoding="utf-8")
    elif json.loads(target.read_text(encoding="utf-8")) != json.loads(rendered):
        raise SystemExit("independent experiment reference differs; review and regenerate")
