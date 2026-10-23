"""Python declarations share the public Rust generic-panel compiler and executor."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from antecedent import joint_panel
from antecedent.intervention_programs import minimum, var

ROOT = Path(__file__).resolve().parents[2]


def declarations():
    with (ROOT / "conformance/response/joint_panel/randomized_interaction.csv").open() as source:
        rows = list(csv.DictReader(source))
    groups = {}
    scenarios = []
    for group in range(4):
        selected = [row for row in rows if int(row["group"]) == group]
        # CSV names remain independent of source-role binding.
        keys = list(selected[0])
        # Columns in this independent fixture: group,time,season,z,a,b,y.
        data = {
            "a": [float(row[keys[4]]) for row in selected],
            "b": [float(row[keys[5]]) for row in selected],
            "c": [2.0] * len(selected),
            "y": [float(row[keys[6]]) if row[keys[6]] else None for row in selected],
            "z": [float(row[keys[3]]) if row[keys[3]] else None for row in selected],
            "season": [float(row[keys[2]]) for row in selected],
        }
        groups[group] = data
        scenarios.append(
            {
                "group": group,
                "history": {name: data[name][:2] for name in ("a", "b", "c")},
                "confounder": data["z"][2:],
                "season": [int(value) for value in data["season"][2:]],
            }
        )
    worlds = [
        {
            "id": "baseline",
            "reference": True,
            "channels": {"a": [0.0] * 24, "b": [0.0] * 24, "c": [2.0] * 24},
            "scenarios": scenarios,
        },
        {
            "id": "joint",
            "reference": False,
            "channels": {"a": [4.0] * 24, "b": [4.0] * 24, "c": [2.0] * 24},
            "scenarios": scenarios,
        },
    ]
    return groups, worlds


def request(groups, worlds, functional=None):
    return joint_panel.prepare(
        groups,
        channels=["a", "b", "c"],
        outcome="y",
        confounder="z",
        season="season",
        seasons=4,
        interval_ns=604_800_000_000_000,
        worlds=worlds,
        interactions={"programmable": var("a") * var("b") * var("c") + minimum(var("c"), 0.25)},
        options={
            "optimizer": {"max_iterations": 1, "gradient_tolerance": 1e-12},
            "inference": {"kind": "frequentist", "draws": 4},
            "initial": None,
        },
        functional=functional,
        seed=20261021,
    )


def test_generic_python_panel_matches_shared_rust_mechanism_and_retains_failed_slots():
    groups, worlds = declarations()
    prepared = request(groups, worlds)
    assert prepared.inspect()["fits_performed"] == 0
    result = prepared.estimate(seed=20261021)
    report = result.report()
    assert report["fits_performed"] == 1
    assert not report["identified"]
    assert not report["diagnostics"]["optimizer_converged"]
    assert report["draws"] == [None] * 4
    assert report["diagnostics"]["calibration"] == "unmeasured"
    point = report["point_parameters"]
    parameters = point["parameters"]
    for world, response in zip(worlds, report["point"], strict=True):
        for time, actual in enumerate(response["means"]):
            expected = 0.0
            for scenario, group_intercept in zip(
                world["scenarios"], point["group_intercepts"], strict=True
            ):
                saturated = []
                for name, channel in zip(("a", "b", "c"), parameters["channels"], strict=True):
                    exposure = sum(
                        weight
                        * (
                            world["channels"][name][time - lag]
                            if lag <= time
                            else scenario["history"][name][2 + time - lag]
                        )
                        for lag, weight in enumerate(channel["lag"])
                    )
                    saturated.append(exposure / (exposure + parameters["half_saturation"]))
                feature = saturated[0] * saturated[1] * saturated[2] + min(saturated[2], 0.25)
                expected += (
                    parameters["intercept"]
                    + group_intercept
                    + sum(
                        channel["coefficient"] * value
                        for channel, value in zip(parameters["channels"], saturated, strict=True)
                    )
                    + parameters["interactions"][0]["coefficient"] * feature
                    + parameters["confounder"] * scenario["confounder"][time]
                    + parameters["season_effects"][scenario["season"][time]]
                ) / 4
            assert actual == pytest.approx(expected, abs=1e-12)
    retargeted = prepared.retarget(worlds, seed=20261021)
    del prepared
    assert retargeted.estimate(seed=20261021).report()["point"] == report["point"]
    assert retargeted.inspect()["fits_performed"] == 1
    replay = joint_panel.consume(result.export(seed=20261021), seed=20261021)
    assert replay["point"] == report["point"]
    assert replay["draws"] == [None] * 4
    assert replay["grants_current_authority"] is False


def test_python_panel_refuses_foreign_expression_read_without_fitting():
    groups, worlds = declarations()
    with pytest.raises(Exception, match="unknown variable|unknown source variable"):
        joint_panel.prepare(
            groups,
            channels=["a", "b", "c"],
            outcome="y",
            confounder="z",
            season="season",
            seasons=4,
            interval_ns=604_800_000_000_000,
            worlds=worlds,
            interactions={"wrong": var("absent")},
            options={
                "optimizer": {"max_iterations": 1, "gradient_tolerance": 1e-5},
                "inference": {"kind": "frequentist", "draws": 0},
                "initial": None,
            },
        )


def test_scheduled_contrast_assembles_on_the_shared_draws_and_refuses_separate_worlds():
    groups, worlds = declarations()
    separate = request(groups, worlds).estimate(seed=20261021)
    with pytest.raises(Exception, match="contrast target"):
        separate.scheduled_contrast()
    prepared = request(groups, worlds, functional={"terms": [(1, 1.0), (0, -1.0)], "reference": 0})
    result = prepared.estimate(seed=20261021)
    report = result.report()
    contrast = result.scheduled_contrast("terminal_mean", level=0.9)
    assert contrast["point"] == pytest.approx(
        report["point"][1]["means"][-1] - report["point"][0]["means"][-1], abs=1e-12
    )
    # Unconverged fit: failed slots stay failed and no interval is invented.
    assert contrast["draws"] == [None] * 4
    assert contrast["interval"] is None
    assert contrast["level"] == 0.9
    with pytest.raises(Exception, match="level"):
        result.scheduled_contrast("terminal_mean", level=1.5)
