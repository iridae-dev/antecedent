# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare a coordinated intervention program with its baseline on a simulated panel.

Four geographic groups are observed for 24 weeks while two channels, A and B, are set each
week. The outcome follows a declared mechanism, so the effect of holding both channels at level
4 for the whole 24-week horizon is known exactly: with lag weights that sum to one, saturation
x / (x + 2), channel coefficients 3 and 2 and interaction 1.5, the terminal-week effect of the
joint program over the baseline is 3 s + 2 s + 1.5 s^2 with s = 4 / 6, which is 4. The data are
a synthetic fixture shipped with the repository; the known truth is a fixture truth of the
stated mechanism and not a claim about any real system.

The example fits one law, reads the contrast from the draws every world shares, shows what an
independent combination would have reported instead, and then asks what the assignment design
establishes about the held-out whole program.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from antecedent import Execution, joint_panel
from antecedent.intervention_programs import var
from antecedent.outcomes import require_success

FIXTURE = Path(__file__).resolve().parents[2] / "conformance" / "response" / "joint_panel"
HORIZON = 24
WEEK_NS = 604_800_000_000_000
SEED = 20261021
CHANNELS = ["channel_a", "channel_b"]
ARMS = [("reference", 0.0, 0.0), ("a", 4.0, 0.0), ("b", 0.0, 4.0), ("joint", 4.0, 4.0)]
FITTED = [0, 3]
JOINT_MINUS_BASELINE = {"terms": [(1, 1.0), (0, -1.0)], "reference": 0}
SATURATION = 4.0 / (4.0 + 2.0)
KNOWN_EFFECT = 3.0 * SATURATION + 2.0 * SATURATION + 1.5 * SATURATION * SATURATION


def read_rows(name: str) -> list[dict[str, str]]:
    with (FIXTURE / name).open() as source:
        return list(csv.DictReader(source))


def declare(rows: list[dict[str, str]], arms=ARMS):
    groups, scenarios = {}, []
    for group in range(4):
        selected = [row for row in rows if int(row["group"]) == group]
        data = {
            "channel_a": [float(row["channel_a"]) for row in selected],
            "channel_b": [float(row["channel_b"]) for row in selected],
            "y": [float(row["outcome"]) if row["outcome"] else None for row in selected],
            "z": [float(row["z"]) if row["z"] else None for row in selected],
            "season": [float(row["season"]) for row in selected],
        }
        groups[group] = data
        scenarios.append(
            {
                "group": group,
                "history": {name: data[name][:2] for name in CHANNELS},
                "confounder": data["z"][2:],
                "season": [int(value) for value in data["season"][2:]],
            }
        )
    worlds = [
        {
            "id": name,
            "reference": index == 0,
            "channels": {"channel_a": [a] * HORIZON, "channel_b": [b] * HORIZON},
            "scenarios": scenarios,
        }
        for index, (name, a, b) in enumerate(arms)
    ]
    return groups, worlds


def prepare(rows, *, draws=0, assignment=None, exogenous_context=False, fitted=False):
    arms = [ARMS[index] for index in FITTED] if fitted else ARMS
    groups, worlds = declare(rows, arms)
    return joint_panel.prepare(
        groups,
        channels=CHANNELS,
        outcome="y",
        confounder="z",
        season="season",
        seasons=4,
        interval_ns=WEEK_NS,
        worlds=worlds,
        interactions={"ab": var("channel_a") * var("channel_b")},
        options={
            "optimizer": {"max_iterations": 2000, "gradient_tolerance": 1e-5},
            "inference": {"kind": "frequentist", "draws": draws},
            "initial": None,
        },
        functional=JOINT_MINUS_BASELINE if fitted else None,
        assignment=assignment,
        exogenous_context=exogenous_context,
        execution=Execution(seed=SEED),
    )


def weekly_assignment(rows):
    """Each group-week is randomized given its observed z; the program is never assigned whole."""
    kept = [row for row in rows if int(row["time"]) >= 0]
    arm_of = [
        int(float(row["channel_a"]) > 0) + 2 * int(float(row["channel_b"]) > 0) for row in kept
    ]
    probabilities = []
    for row in kept:
        a = 0.35 if float(row["z"]) < 0 else 0.65
        b = 0.65 if float(row["z"]) < 0 else 0.35
        probabilities.append([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b])
    units = [f"group_{row['group']}.period_{row['time']}" for row in kept]
    return {
        "design": {"kind": "multi_arm"},
        "outcome": 2,
        "realized_assignment": [arm != 0 for arm in arm_of],
        "assignment_probabilities": [p[arm] for p, arm in zip(probabilities, arm_of, strict=True)],
        "assignment_units": units,
        "outcome_units": units,
        "treatment_arms": ["reference", "a"],
        "multi_arm_labels": [name for name, _, _ in ARMS],
        "multi_arm_assignment": arm_of,
        "multi_arm_probabilities": probabilities,
    }


def whole_program_assignment():
    """One independent group per program, each assigned its whole program with probability 1/4."""
    return {
        "design": {"kind": "multi_arm"},
        "outcome": 2,
        "realized_assignment": [False, True, True, True],
        "assignment_probabilities": [0.25] * 4,
        "assignment_units": ["0", "1", "2", "3"],
        "outcome_units": ["0", "1", "2", "3"],
        "treatment_arms": ["reference", "a"],
        "multi_arm_labels": [name for name, _, _ in ARMS],
        "multi_arm_assignment": [0, 1, 2, 3],
        "multi_arm_probabilities": [[0.25] * 4 for _ in range(4)],
    }


def whole_program_attempt(inspection):
    return {a["id"]: a for a in inspection["attempts"]}["randomized.multi_arm_itt.whole_program"]


def run() -> dict[str, object]:
    rows = read_rows("randomized_interaction.csv")
    prepared = prepare(rows, draws=100, fitted=True)
    assert prepared.inspect()["fits_performed"] == 0
    result = require_success(prepared.estimate(execution=Execution(seed=SEED)))
    report = result.report()
    assert report["fits_performed"] == 1
    assert report["diagnostics"]["optimizer_converged"]
    print("calibration:", report["diagnostics"]["calibration"])
    assert report["identified"] is False

    contrast = result.scheduled_contrast("terminal_mean", level=0.95)
    assert contrast["uncertainty"] == "EstimatorSampling"
    point = report["point"]
    assert math.isclose(
        contrast["point"], point[1]["means"][-1] - point[0]["means"][-1], abs_tol=1e-12
    )

    complete = [draw for draw in report["draws"] if draw is not None]
    assert complete
    baseline = [draw[0]["means"][-1] for draw in complete]
    program = [draw[1]["means"][-1] for draw in complete]
    differences = [after - before for after, before in zip(program, baseline, strict=True)]
    shared = [value for value in contrast["draws"] if value is not None]
    assert all(
        math.isclose(left, right, abs_tol=1e-9)
        for left, right in zip(shared, differences, strict=True)
    )
    covariance = statistics.covariance(program, baseline)
    shared_se = statistics.stdev(differences)
    independent_se = math.sqrt(statistics.variance(program) + statistics.variance(baseline))
    assert math.isclose(
        shared_se**2,
        statistics.variance(program) + statistics.variance(baseline) - 2 * covariance,
        rel_tol=1e-9,
    )
    assert shared_se < independent_se
    assert abs(contrast["point"] - KNOWN_EFFECT) <= 3.0 * shared_se
    if contrast["standard_error"] is not None:
        assert math.isclose(contrast["standard_error"], shared_se, rel_tol=1e-9)

    per_week = prepare(rows, assignment=weekly_assignment(rows), exogenous_context=True).inspect()
    weekly = whole_program_attempt(per_week)
    assert per_week["fits_performed"] == 0
    assert per_week["identified"] is False
    assert weekly["outcome"] == "Rejected"
    assert weekly["next_attempt"] is not None

    experiment = read_rows("whole_program_interaction.csv")
    randomized = prepare(
        experiment, assignment=whole_program_assignment(), exogenous_context=True
    ).inspect()
    assert randomized["identified"] is True
    assert whole_program_attempt(randomized)["outcome"] == "Identified"
    undeclared = prepare(
        experiment, assignment=whole_program_assignment(), exogenous_context=False
    ).inspect()
    assert undeclared["identified"] is False
    assert whole_program_attempt(undeclared)["outcome"] == "Rejected"

    replay = require_success(
        joint_panel.consume(
            bytes(result.export(execution=Execution(seed=SEED))), execution=Execution(seed=SEED)
        )
    )
    assert replay["point"] == report["point"]
    assert replay["grants_current_authority"] is False
    return {
        "known_effect": KNOWN_EFFECT,
        "estimated_effect": contrast["point"],
        "shared_fit_standard_error": shared_se,
        "independent_combination_standard_error": independent_se,
        "response_covariance": covariance,
        "failed_draw_slots": len(report["draws"]) - len(complete),
        "weekly_assignment_identifies_program": per_week["identified"],
        "weekly_assignment_detail": weekly["detail"],
        "whole_program_assignment_identifies_program": randomized["identified"],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
