# SPDX-License-Identifier: MIT OR Apache-2.0
"""Learn a feedback law, branch intervention targets, and reuse a consumed law.

Synthetic equations: A = 0.4 B + noise_A; B = 0.5 A + noise_B.
A passive sample and two independent randomized clamp experiments identify
both directions under this declared linear recipe. Sampling draws describe fit uncertainty; they
are not a parameter posterior or a calibrated coverage claim.
"""

from __future__ import annotations

import json
import random

import antecedent as ant
from antecedent.cyclic import (
    AsymptoticNormal,
    ExperimentRegime,
    World,
    prepare_experimental_recovery,
)
from antecedent.intervention import Set
from antecedent.outcomes import require_success


def run() -> dict[str, object]:
    generator = random.Random(19)
    clamp_a = [generator.gauss(0.0, 1.0) for _ in range(128)]
    clamp_b = [generator.gauss(0.0, 1.0) for _ in range(128)]
    disturbance = [(generator.gauss(0.0, 0.02), generator.gauss(0.0, 0.02)) for _ in range(128)]
    regimes = [
        ExperimentRegime(
            {
                "A": [(a + 0.4 * b) / 0.8 for a, b in disturbance],
                "B": [(0.5 * a + b) / 0.8 for a, b in disturbance],
            },
            source=0,
        ),
        ExperimentRegime(
            {"A": clamp_a, "B": [0.5 * a + generator.gauss(0.0, 0.02) for a in clamp_a]},
            source=1,
            clamps={"A": 0.0},
        ),
        ExperimentRegime(
            {"A": [0.4 * b + generator.gauss(0.0, 0.02) for b in clamp_b], "B": clamp_b},
            source=2,
            clamps={"B": 0.0},
        ),
    ]
    study = prepare_experimental_recovery(
        ["A", "B"],
        regimes,
        [World(1, Set("A", 1.0))],
        outcome="B",
        uncertainty=AsymptoticNormal(32),
        source_id="synthetic-clamp-experiments",
        certifying_evidence=["declared-independent-linear-generator"],
        sampling_design="independent_subject_trajectories",
        execution=ant.Execution(seed=19, threads=1),
    )
    assert study.inspect()["preparation_evaluations"] == 0
    result = require_success(ant.estimate(study))
    original = require_success(result.interventions())[1].mean()
    assert abs(original - 0.5) < 0.05

    # A new target shares the recovered law and all original draw statuses.
    branch = result.law.prepare([World(3, Set("A", 3.0))], variables=["A", "B"], outcome="B")
    assert branch.inspect()["preparation_evaluations"] == 0
    continued = require_success(ant.estimate(branch))
    changed = require_success(continued.interventions())[3].mean()
    assert abs(changed - 1.5) < 0.05
    assert continued.law.to_dict() == result.law.to_dict()
    assert require_success(result.interventions())[1].mean() == original

    artifact = continued.export()
    loaded = ant.load(bytes(artifact), expected=artifact.expectation)
    assert loaded.study is None
    consumed = require_success(loaded)
    assert consumed.law.to_dict() == result.law.to_dict()
    independent_branch = consumed.law.prepare(
        [World(4, Set("A", 4.0))], variables=["A", "B"], outcome="B"
    )
    independent = require_success(ant.estimate(independent_branch))
    independent_mean = require_success(independent.interventions())[4].mean()
    assert abs(independent_mean - 2.0) < 0.05
    return {
        "original_mean": original,
        "branched_mean": changed,
        "consumed_law_mean": independent_mean,
        "draw_origin": result.law.inspect().details["draw_origin"],
        "acceptance": loaded.acceptance.status,
    }


if __name__ == "__main__":
    # Retained read models are immutable mappings; copy them to plain dictionaries to print.
    print(json.dumps(run(), indent=2, default=dict))
