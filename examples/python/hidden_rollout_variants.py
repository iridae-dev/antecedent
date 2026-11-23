# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare a hidden two-version rollout with a rollout that uses one version."""

from __future__ import annotations

import math
import random

import antecedent as ant
from antecedent import latent_soft_intervention as lsi


def arms(two_versions: bool) -> tuple[list[list[float]], list[list[float]]]:
    rng = random.Random(664)
    controls: list[list[float]] = []
    treated: list[list[float]] = []
    for arm, rows in ((0, controls), (1, treated)):
        for _ in range(1200):
            context = rng.gauss(0, 1)
            first_path = not two_versions or rng.random() < 1 / (1 + math.exp(-1.8 * context))
            first = 0.4 * context + rng.gauss(0, 0.45) + 3 * (arm == 1 and first_path)
            second = -0.3 * context + rng.gauss(0, 0.45) + 3 * (arm == 1 and not first_path)
            rows.append([context, first, second, first + second + rng.gauss(0, 0.6)])
    return controls, treated


def main() -> None:
    for two_versions in (True, False):
        baseline, action = arms(two_versions)
        program = lsi.prepare(
            variables=("context", "setup", "follow_up", "completion"),
            edges=(
                ("context", "setup"),
                ("context", "follow_up"),
                ("setup", "completion"),
                ("follow_up", "completion"),
            ),
            context=("context",),
            candidates=("setup", "follow_up"),
            baseline=baseline,
            action=action,
            population="synthetic rollout",
            randomized=True,
            version_exogenous_given_context=True,
            execution=ant.Execution(seed=73),
        )
        plan = program.inspect()
        assert plan["route"] == "LatentSoftIntervention"
        if not plan["support"]["supported"] and ant.build_flavor() == "release":
            print("Rollout plan prepared. This release build lacks support for the requested fit.")
            continue
        fitted = program.execute()
        discovery = fitted.discovery
        print("Two planted versions" if two_versions else "One planted version")
        print("Evidence and candidate targets", discovery.evidence, discovery.hypotheses)
        if discovery.forecast_admissible:
            forecast = fitted.forecast({"context": 0.0})
            exported = fitted.export({"context": 0.0})
            replay = ant.load(exported, expected=exported.expectation)
            assert replay.value.forecast == forecast
            print("Conditional forecast at context 0", forecast)
        else:
            print("The fit withheld a forecast for this dataset.")


if __name__ == "__main__":
    main()
