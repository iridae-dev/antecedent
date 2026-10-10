"""2.3A X5 temporal extensions: uncertain initial state, new-period refresh, closed interval.

The truth is the enumerated two-step dynamic SCM the Rust core tests use: for the sequence
``(0, 0)`` the response given ``s0`` is ``0.33`` at ``s0 = 0`` and ``0.46`` at ``s0 = 1``; the
target law ``P(s0 = 1) = 0.7`` gives the marginalized value ``0.3 * 0.33 + 0.7 * 0.46 = 0.421``,
while fixing the state at its mode gives ``0.46``. A replacement panel whose every cell mean is
``0.1`` higher re-evaluates to ``0.521``.
"""

from __future__ import annotations

import pytest
from antecedent.errors import CausalSerializationError, CausalUnsupportedError
from antecedent.transport import advanced
from antecedent.transport._temporal_extensions import _temporal_dependent_interval_candidate
from antecedent.transport.advanced import (
    InitialStateLaw,
    TemporalExtensionRefusal,
    TemporalPremises,
    TemporalUnitPanel,
    TemporalWindow,
    consume_temporal_initial_state_artifact,
    consume_temporal_refresh_artifact,
    temporal_dependent_interval,
    temporal_initial_state,
    temporal_new_period_refresh,
)

from _refusal import assert_registered_refusal

# P(l = 1 | s0, a1) in tenths, indexed [s0][a1].
P_L1 = [[3, 7], [2, 8]]
# Outcome mean in tenths, indexed [s0][a1][l][a2].
M10 = [
    [[[3, 4], [4, 6]], [[3, 5], [5, 7]]],
    [[[4, 3], [7, 6]], [[6, 3], [5, 7]]],
]
SEQUENCE = (0, 0)


def make_panel(snapshot: str, *, first_unit: int = 0, start: int = 0, lift: int = 0):
    """Two identical units holding every (s0, a1) block in the SCM's proportions."""
    units = []
    for unit in range(2):
        rows = []
        for s0 in range(2):
            for a1 in range(2):
                n_one = 20 * P_L1[s0][a1]
                for level in range(2):
                    n_level = n_one if level == 1 else 200 - n_one
                    for a2 in range(2):
                        n_cell = n_level // 2
                        ones = n_cell * (M10[s0][a1][level][a2] + lift) // 10
                        for k in range(n_cell):
                            time_id = start + len(rows)
                            y = 1.0 if k < ones else 0.0
                            rows.append((first_unit + unit, time_id, s0, a1, level, a2, y))
        units.extend(rows)
    return TemporalUnitPanel.from_rows(snapshot, units)


def premises(**overrides) -> TemporalPremises:
    fields = {
        "initial_state_variable": "s0",
        "time_order": ["s0", "a1", "l2", "a2", "y"],
        "source_regime": "source",
        "target_regime": "target",
        "graph_id": "two_slice_graph_v1",
        "proof_id": "proof_1",
    }
    fields.update(overrides)
    return TemporalPremises(**fields)


def window(period, **overrides) -> TemporalWindow:
    fields = {
        "lag_alignment": {"s0": 1, "a1": 1, "l2": 2, "a2": 2, "y": 2},
        "intervention_history": ["a1=0", "a2=0"],
        "selection_targets": ["s_t1", "s_t2"],
    }
    fields.update(overrides)
    return TemporalWindow(period=period, **fields)


def target_law(p1: float = 0.7, snapshot: str = "target-state") -> InitialStateLaw:
    return InitialStateLaw(snapshot, {0: 1.0 - p1, 1: p1})


def held(*, with_window: bool = True, fixed_state: int | None = 1):
    return temporal_initial_state(
        make_panel("snap_old"),
        sequence=SEQUENCE,
        target_law=target_law(),
        premises=premises(),
        fixed_state=fixed_state,
        window=window((0, 1000)) if with_window else None,
    )


def new_panel(snapshot: str = "snap_new", start: int = 1000):
    return make_panel(snapshot, first_unit=10, start=start, lift=1)


def refusal_of(call) -> TemporalExtensionRefusal:
    with pytest.raises(TemporalExtensionRefusal) as caught:
        call()
    error = caught.value
    assert isinstance(error, CausalUnsupportedError)
    assert_registered_refusal(error)
    return error


# --- uncertain initial state -------------------------------------------------------------


def test_x5_initial_state_marginalizes_over_the_target_law_not_the_fixed_mode():
    result = held()
    assert result.value == pytest.approx(0.421, abs=1e-12)
    assert result.label == "marginalized_initial_state"
    assert result.inference_claim == "point_only"
    assert result.interval is None
    assert result.sequence == SEQUENCE
    assert [c.state for c in result.contributions] == [0, 1]
    assert result.contributions[0].response == pytest.approx(0.33, abs=1e-12)
    assert result.contributions[1].response == pytest.approx(0.46, abs=1e-12)
    assert result.state_snapshot_id == "target-state"
    assert result.panel_snapshot_id == "snap_old"
    # Fixing the state at the law's mode is a different estimand with its own label.
    assert result.fixed is not None
    assert result.fixed.label == "fixed_initial_state"
    assert result.fixed.value == pytest.approx(0.46, abs=1e-12)
    assert abs(result.value - result.fixed.value) > 0.03


def test_x5_initial_state_fixed_and_marginalized_are_never_relabeled():
    result = held()
    assert result.label != result.fixed.label
    assert result.label == advanced_label("MARGINALIZED_LABEL")
    assert result.fixed.label == advanced_label("FIXED_LABEL")
    # No fixed-state request, no fixed result: the marginalized value is not offered as one.
    alone = held(fixed_state=None)
    assert alone.fixed is None
    assert alone.value == pytest.approx(result.value, abs=0.0)
    consumed = consume_temporal_initial_state_artifact(result.export())
    assert consumed.label == "marginalized_initial_state"
    assert consumed.fixed.label == "fixed_initial_state"
    assert consumed.value != consumed.fixed.value


def advanced_label(name: str) -> str:
    from antecedent.transport import _temporal_extensions

    return getattr(_temporal_extensions, name)


def test_x5_initial_state_follows_the_law_and_its_snapshot():
    even = temporal_initial_state(
        make_panel("snap_old"),
        sequence=SEQUENCE,
        target_law=target_law(0.5, "target-state-v2"),
        premises=premises(),
    )
    assert even.value == pytest.approx(0.395, abs=1e-12)
    assert even.state_snapshot_id == "target-state-v2"
    assert even.state_law_digest != held().state_law_digest


def test_x5_initial_state_export_consumes_independently():
    result = held()
    blob = result.export()
    assert isinstance(blob, bytes)
    consumed = consume_temporal_initial_state_artifact(blob)
    assert consumed.value == pytest.approx(0.421, abs=1e-12)
    assert consumed.replayed_value == consumed.value
    assert consumed.replayed_fixed_value == consumed.fixed.value
    assert consumed.fixed.state == 1
    assert consumed.law_population == "target"
    assert consumed.law_snapshot_id == "target-state"
    assert consumed.panel_snapshot_id == "snap_old"
    assert consumed.sequence == SEQUENCE
    assert consumed.inference_claim == "point_only"
    assert consumed.premises["initial_state_variable"] == "s0"
    assert consumed.premises["time_order"][0] == "s0"
    assert consumed.premises["graph_id"] == "two_slice_graph_v1"
    assert [c.state for c in consumed.contributions] == [0, 1]


def test_x5_initial_state_consumer_refuses_tampered_or_foreign_bytes():
    blob = bytearray(held().export())
    blob[-3] ^= 0x55
    with pytest.raises(CausalSerializationError):
        consume_temporal_initial_state_artifact(bytes(blob))
    with pytest.raises(CausalSerializationError):
        consume_temporal_initial_state_artifact(b"not an artifact")
    refreshed = temporal_new_period_refresh(held(), new_panel(), window=window((1000, 2000)))
    with pytest.raises(CausalSerializationError):
        consume_temporal_initial_state_artifact(refreshed.export())


def test_x5_source_state_only_refuses_a_source_law_and_a_point_state():
    source = InitialStateLaw("source-state", {0: 0.5, 1: 0.5}, population="source")
    for law in (source, 1):
        error = refusal_of(
            lambda law=law: temporal_initial_state(
                make_panel("p"), sequence=SEQUENCE, target_law=law, premises=premises()
            )
        )
        assert error.reason_code == "transport_missing_evidence"
        assert error.detail == "initial_state.target_law_missing"
        assert "initial_state.target_law_missing" in str(error)


def test_x5_source_state_only_refuses_support_gaps_and_bad_premises():
    gap = InitialStateLaw("target-state", {0: 0.5, 7: 0.5})
    error = refusal_of(
        lambda: temporal_initial_state(
            make_panel("p"), sequence=SEQUENCE, target_law=gap, premises=premises()
        )
    )
    assert error.reason_code == "transport_support_failure"
    assert error.detail == "initial_state.support_gap"

    error = refusal_of(
        lambda: temporal_initial_state(
            make_panel("p"),
            sequence=SEQUENCE,
            target_law=target_law(),
            premises=premises(time_order=["a1", "s0"]),
        )
    )
    assert error.reason_code == "invalid_argument"
    assert error.detail == "initial_state.invalid_premises"

    error = refusal_of(
        lambda: temporal_initial_state(
            TemporalUnitPanel("p", None),
            sequence=SEQUENCE,
            target_law=target_law(),
            premises=premises(),
        )
    )
    assert error.detail == "temporal_interval.unknown_units"
    assert error.reason_code == "route_not_supported"


# --- new-period refresh ------------------------------------------------------------------


def test_x5_period_replace_re_evaluates_and_a_stale_interval_never_survives():
    prior = held()
    refreshed = temporal_new_period_refresh(
        prior, new_panel(), window=window((1000, 2000)), interval_existed=True
    )
    assert refreshed.value == pytest.approx(0.521, abs=1e-12)
    assert refreshed.previous_value == pytest.approx(0.421, abs=1e-12)
    assert refreshed.value != refreshed.previous_value
    assert refreshed.decision == "reusable"
    assert refreshed.label == "marginalized_initial_state"
    # The held interval was invalidated; the refreshed result has none.
    assert refreshed.interval is None
    assert refreshed.result.interval is None
    assert refreshed.interval_invalidated is True
    assert refreshed.receipt.interval_invalidated is True
    assert refreshed.receipt.inference_claim == "point_only"
    assert refreshed.receipt.old_period == (0, 1000)
    assert refreshed.receipt.new_period == (1000, 2000)
    assert refreshed.receipt.old_snapshot_id == "snap_old"
    assert refreshed.receipt.new_snapshot_id == "snap_new"
    assert refreshed.receipt.proof_id == "proof_1"
    assert refreshed.result.panel_snapshot_id == "snap_new"
    # The prior result is unchanged.
    assert prior.value == pytest.approx(0.421, abs=1e-12)
    # The fixed-state companion is re-evaluated too, under its own label.
    assert refreshed.result.fixed.label == "fixed_initial_state"
    assert refreshed.result.fixed.value == pytest.approx(0.56, abs=1e-12)


def test_x5_period_replace_without_a_prior_interval_records_none():
    refreshed = temporal_new_period_refresh(held(), new_panel(), window=window((1000, 2000)))
    assert refreshed.interval_invalidated is False
    assert refreshed.receipt.interval_invalidated is False


def test_x5_period_replace_chains_and_exports_an_independent_artifact():
    first = temporal_new_period_refresh(held(), new_panel(), window=window((1000, 2000)))
    second = temporal_new_period_refresh(
        first,
        make_panel("snap_third", first_unit=20, start=2000, lift=2),
        window=window((2000, 3000)),
    )
    assert second.previous_value == pytest.approx(first.value, abs=0.0)
    assert second.value == pytest.approx(0.621, abs=1e-12)
    blob = first.export()
    consumed = consume_temporal_refresh_artifact(blob)
    assert consumed.decision == "reusable"
    assert consumed.invalidation is None
    assert consumed.old_value == pytest.approx(0.421, abs=1e-12)
    assert consumed.new_value == pytest.approx(0.521, abs=1e-12)
    assert consumed.interval_present is False
    assert consumed.label == "marginalized_initial_state"
    assert consumed.receipt.digest == first.receipt.digest
    assert consumed.receipt.new_period == (1000, 2000)
    assert consumed.new_window["snapshot_id"] == "snap_new"
    assert consumed.old_window["snapshot_id"] == "snap_old"
    assert consumed.inference_claim == "point_only"


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({"horizon": 3}, "temporal_refresh.horizon_changed"),
        (
            {"lag_alignment": {"s0": 1, "a1": 2, "l2": 2, "a2": 2, "y": 2}},
            "temporal_refresh.lag_alignment_changed",
        ),
        (
            {"intervention_history": ["a1=0", "a2=1"]},
            "temporal_refresh.intervention_history_changed",
        ),
        ({"proof_id": "proof_2"}, "temporal_refresh.premises_changed"),
        ({"graph_id": "three_slice_graph"}, "temporal_refresh.premises_changed"),
        ({"selection_targets": ["s_t1"]}, "temporal_refresh.premises_changed"),
    ],
)
def test_x5_third_period_invalidates_with_a_typed_refusal(overrides, detail):
    error = refusal_of(
        lambda: temporal_new_period_refresh(
            held(), new_panel(), window=window((1000, 2000), **overrides), interval_existed=True
        )
    )
    assert error.reason_code == "route_not_supported"
    assert error.detail == detail
    assert detail in str(error)


def test_x5_third_period_a_replacement_that_is_not_new_is_stale():
    same_snapshot = new_panel(snapshot="snap_old")
    error = refusal_of(
        lambda: temporal_new_period_refresh(held(), same_snapshot, window=window((1000, 2000)))
    )
    assert error.reason_code == "route_not_supported"
    assert error.detail == "temporal_refresh.stale_snapshot"
    same_period = make_panel("snap_new", first_unit=10, start=0, lift=1)
    error = refusal_of(
        lambda: temporal_new_period_refresh(held(), same_period, window=window((0, 1000)))
    )
    assert error.detail == "temporal_refresh.stale_snapshot"


def test_x5_third_period_invalid_replacements_and_missing_windows_refuse():
    # A period that does not cover the panel's time ids.
    error = refusal_of(
        lambda: temporal_new_period_refresh(held(), new_panel(), window=window((1000, 1500)))
    )
    assert error.reason_code == "invalid_argument"
    assert error.detail == "temporal_refresh.invalid_replacement"
    error = refusal_of(
        lambda: temporal_new_period_refresh(held(), new_panel(), window=window((2000, 2000)))
    )
    assert error.detail == "temporal_refresh.invalid_replacement"
    # A held result that never declared its window has nothing to refresh.
    error = refusal_of(
        lambda: temporal_new_period_refresh(
            held(with_window=False), new_panel(), window=window((1000, 2000))
        )
    )
    assert error.reason_code == "invalid_argument"
    assert error.detail == "temporal_refresh.window_missing"
    # A result that was not produced by a refresh has no refresh artifact.
    error = refusal_of(held().export_refresh)
    assert error.detail == "temporal_refresh.no_refresh"


def test_x5_period_reseal_consumer_refuses_tampered_refresh_bytes():
    blob = bytearray(
        temporal_new_period_refresh(held(), new_panel(), window=window((1000, 2000))).export()
    )
    blob[-3] ^= 0x55
    with pytest.raises(CausalSerializationError):
        consume_temporal_refresh_artifact(bytes(blob))
    with pytest.raises(CausalSerializationError):
        consume_temporal_refresh_artifact(held().export())


# --- closed dependent interval -----------------------------------------------------------


def interval_panel(units: int = 24) -> TemporalUnitPanel:
    rows = []
    for unit in range(units):
        time_id = 0
        for s0 in range(2):
            for level in range(2):
                for a2 in range(2):
                    y = ((unit * 7 + s0 * 3 + level * 5 + a2) % 10) / 10.0
                    rows.append((unit, time_id, s0, 0, level, a2, y))
                    time_id += 1
    return TemporalUnitPanel.from_rows("interval-panel", rows)


def test_x5_internal_candidate_route_is_closed_after_real_validation():
    error = refusal_of(
        lambda: _temporal_dependent_interval_candidate(
            interval_panel(), sequence=SEQUENCE, replicates=40
        )
    )
    assert error.reason_code == "cell_not_licensed"
    assert error.detail == "temporal_interval.route_frozen"
    assert "temporal_interval.route_frozen" in str(error)
    # The internal candidate diagnostic remains closed for every estimand.
    for kwargs in (
        {"estimand": "fixed_initial_state", "fixed_state": 1},
        {"estimand": "marginalized_initial_state", "target_law": target_law()},
    ):
        error = refusal_of(
            lambda kwargs=kwargs: _temporal_dependent_interval_candidate(
                interval_panel(), sequence=SEQUENCE, replicates=40, **kwargs
            )
        )
        assert error.detail == "temporal_interval.route_frozen"


def test_x5_dependent_interval_validates_its_arguments_before_closing():
    error = refusal_of(
        lambda: temporal_dependent_interval(
            TemporalUnitPanel("interval-panel", None),
            sequence=SEQUENCE,
            target_law=target_law(),
            replicates=40,
        )
    )
    assert error.reason_code == "route_not_supported"
    assert error.detail == "temporal_interval.unknown_units"

    error = refusal_of(
        lambda: temporal_dependent_interval(
            interval_panel(5), sequence=SEQUENCE, target_law=target_law(), replicates=40
        )
    )
    assert error.reason_code == "too_few_clusters"
    assert error.detail == "temporal_interval.too_few_units"

    error = refusal_of(
        lambda: temporal_dependent_interval(
            interval_panel(), sequence=SEQUENCE, target_law=target_law(), replicates=5000
        )
    )
    assert error.reason_code == "route_not_supported"
    assert error.detail == "temporal_interval.too_many_replicates"

    error = refusal_of(
        lambda: temporal_dependent_interval(
            interval_panel(), sequence=SEQUENCE, target_law=target_law(), replicates=10
        )
    )
    assert error.reason_code == "invalid_argument"

    error = refusal_of(
        lambda: temporal_dependent_interval(
            interval_panel(),
            sequence=SEQUENCE,
            estimand="fixed_initial_state",
            fixed_state=9,
            replicates=40,
        )
    )
    assert error.reason_code == "route_not_supported"
    assert error.detail == "temporal_interval.unsupported_history"


def test_x5_the_two_step_sequence_route_is_unchanged():
    # The 2.2 point-only sequence route keeps its own symbols (and its interval refusal).
    assert callable(advanced.prepare_temporal_transport_sequence)
    assert callable(advanced.consume_temporal_transport_artifact)
    assert advanced.TemporalSequenceSpec.__name__ == "TemporalSequenceSpec"
