"""Categorical panel channels are dummy-coded by the Rust checked conversion."""

from __future__ import annotations

import pytest
from antecedent import Execution, ResourceLimits, joint_panel, state
from antecedent.errors import (
    CausalCancelledError,
    CausalError,
    CausalResourceError,
    CausalTypeError,
    CausalValueError,
)
from antecedent.joint_panel import PanelCategoricalChannel, PanelDummyChannel
from antecedent.outcomes import require_success

from test_joint_panel import declarations

INTERVAL_NS = 604_800_000_000_000
LEVELS = ("none", "low", "high")
ROLES = {"outcome": "y", "confounder": "z", "season": "season"}


def label(value, row):
    """No dose is ``none``; a dose is ``low`` on an even row and ``high`` on an odd one."""
    if value == 0:
        return "none"
    return "low" if row % 2 == 0 else "high"


def labelled():
    groups, _ = declarations()
    coded = {
        group: {**columns, "a": [label(v, row) for row, v in enumerate(columns["a"])]}
        for group, columns in groups.items()
    }
    return coded, groups


def declared(**changes):
    values = {"channel": "a", "levels": LEVELS, "reference": "none"}
    return PanelCategoricalChannel(**{**values, **changes})


def code(groups, categorical=None, **changes):
    return joint_panel.dummy_code_panel(
        groups,
        channels=changes.pop("channels", ["a", "b", "c"]),
        interval_ns=INTERVAL_NS,
        categorical=[declared()] if categorical is None else categorical,
        **ROLES,
        **changes,
    )


def test_dummy_coding_replaces_the_channel_by_an_indicator_per_non_reference_level():
    groups, numeric = labelled()
    coded = code(groups)
    assert coded.channels == ("a=low", "a=high", "b", "c")
    assert coded.indicators == (
        PanelDummyChannel(name="a=low", source="a", level="low"),
        PanelDummyChannel(name="a=high", source="a", level="high"),
    )
    assert (coded.outcome, coded.confounder, coded.season) == ("y", "z", "season")
    for group, columns in coded.groups.items():
        assert set(columns) == {"a=low", "a=high", "b", "c", "y", "z", "season"}
        labels = groups[group]["a"]
        assert columns["a=low"] == tuple(1.0 if v == "low" else 0.0 for v in labels)
        assert columns["a=high"] == tuple(1.0 if v == "high" else 0.0 for v in labels)
        # The numeric channel, the outcome (with its missing prehistory) and the covariates
        # carry over unchanged.
        for name in ("b", "c", "y", "z", "season"):
            assert columns[name] == tuple(numeric[group][name])
    assert coded.groups[0]["y"][:2] == (None, None)


def test_a_missing_label_stays_missing_in_every_indicator_of_its_channel():
    groups, _ = labelled()
    groups[0]["a"][5] = None
    coded = code(groups)
    assert coded.groups[0]["a=low"][5] is None
    assert coded.groups[0]["a=high"][5] is None
    assert coded.groups[0]["a=low"][4] is not None


def test_coded_panel_prepares_exactly_as_its_labels_do():
    groups, _ = labelled()
    coded = code(groups)
    channels = list(coded.channels)

    def worlds(names):
        scenarios = [
            {
                "group": group,
                "history": {name: list(columns[name][:2]) for name in names},
                "confounder": list(columns["z"][2:]),
                "season": [int(v) for v in columns["season"][2:]],
            }
            for group, columns in coded.groups.items()
        ]
        levels = {"a=low": 0.0, "a=high": 0.0, "b": 0.0, "c": 2.0}
        joint = {"a=low": 0.0, "a=high": 1.0, "b": 4.0, "c": 2.0}
        return [
            {
                "id": identifier,
                "reference": identifier == "baseline",
                "channels": {name: [setting[name]] * 24 for name in names},
                "scenarios": scenarios,
            }
            for identifier, setting in (("baseline", levels), ("joint", joint))
        ]

    options = {
        "optimizer": {"max_iterations": 1, "gradient_tolerance": 1e-12},
        "inference": {"kind": "frequentist", "draws": 4},
        "initial": None,
    }
    shared = {
        "channels": channels,
        "interval_ns": INTERVAL_NS,
        "seasons": 4,
        "worlds": worlds(channels),
        "interactions": {},
        "options": options,
        **ROLES,
    }
    by_labels = joint_panel.prepare(
        groups,
        **{**shared, "channels": ["a", "b", "c"]},
        categorical=[declared()],
        execution=Execution(seed=11),
    )
    by_hand = joint_panel.prepare(coded.groups, **shared, execution=Execution(seed=11))
    assert by_labels.inspect()["indicators"] == [
        {"name": "a=low", "source": "a", "level": "low"},
        {"name": "a=high", "source": "a", "level": "high"},
    ]
    assert by_hand.inspect()["indicators"] == []
    point = [
        require_success(panel.estimate(execution=Execution(seed=11))).report()["point"]
        for panel in (by_labels, by_hand)
    ]
    assert point[0] == point[1]
    # The original channel name no longer exists: worlds name its indicators.
    renamed = worlds(channels)
    for world in renamed:
        world["channels"]["a"] = world["channels"].pop("a=low")
    with pytest.raises(CausalValueError, match="unknown source variable"):
        joint_panel.prepare(
            groups,
            **{**shared, "channels": ["a", "b", "c"], "worlds": renamed},
            categorical=[declared()],
        )


@pytest.mark.parametrize(
    ("categorical", "message"),
    [
        (declared(levels=("none", "low")), "differs from the domain"),
        (declared(levels=(*LEVELS, "mid")), "differs from the domain"),
        (declared(reference="mid"), "one of them the reference"),
        (declared(levels=("none",)), "at least two distinct declared levels"),
        (declared(levels=("none", "none", "low", "high")), "at least two distinct declared levels"),
        (declared(channel="absent"), "unknown source variable"),
        (declared(channel="y"), "category labels|not one of the declared channels"),
    ],
)
def test_a_declaration_the_data_cannot_support_is_refused_by_name(categorical, message):
    groups, _ = labelled()
    with pytest.raises(CausalValueError, match=message):
        code(groups, [categorical])


def test_numbers_where_labels_are_declared_and_a_twice_declared_channel_are_refused():
    groups, numeric = labelled()
    with pytest.raises(CausalValueError, match="category labels, not numbers"):
        code(numeric)
    with pytest.raises(CausalValueError, match="declared twice"):
        code(groups, [declared(), declared()])
    # A label column that is not declared categorical is not read as a dose.
    with pytest.raises(CausalValueError, match="not a declared categorical channel"):
        code(groups, [])


def test_a_label_channel_with_no_observed_level_is_refused():
    groups, _ = labelled()
    for columns in groups.values():
        columns["a"] = [None] * len(columns["a"])
    with pytest.raises(CausalValueError, match="no observed level"):
        code(groups)


def test_declarations_are_typed_and_the_conversion_runs_under_the_callers_execution():
    with pytest.raises(CausalTypeError):
        PanelCategoricalChannel(channel="a", levels="none", reference="none")
    with pytest.raises(CausalTypeError):
        PanelCategoricalChannel(channel="a", levels=("none", 1), reference="none")
    groups, _ = labelled()
    with pytest.raises(CausalTypeError):
        code(groups, [{"channel": "a"}])
    with pytest.raises(CausalResourceError):
        code(groups, execution=Execution(limits=ResourceLimits(memory_bytes=1)))
    token = state.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        code(groups, execution=Execution(cancel=token))
    with pytest.raises(CausalError):
        code(groups, channels=["a", "b"])
