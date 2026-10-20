"""Temporal extensions: uncertain initial state, new-period refresh, closed interval.

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
    PeriodPrior,
    PeriodRefresh,
    PeriodWindow,
    consume_period_refresh_artifact,
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


def test_initial_state_marginalizes_over_the_target_law_not_the_fixed_mode():
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


def test_initial_state_fixed_and_marginalized_are_never_relabeled():
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


def test_initial_state_follows_the_law_and_its_snapshot():
    even = temporal_initial_state(
        make_panel("snap_old"),
        sequence=SEQUENCE,
        target_law=target_law(0.5, "target-state-v2"),
        premises=premises(),
    )
    assert even.value == pytest.approx(0.395, abs=1e-12)
    assert even.state_snapshot_id == "target-state-v2"
    assert even.state_law_digest != held().state_law_digest


def test_initial_state_export_consumes_independently():
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


def test_initial_state_consumer_refuses_tampered_or_foreign_bytes():
    blob = bytearray(held().export())
    blob[-3] ^= 0x55
    with pytest.raises(CausalSerializationError):
        consume_temporal_initial_state_artifact(bytes(blob))
    with pytest.raises(CausalSerializationError):
        consume_temporal_initial_state_artifact(b"not an artifact")
    refreshed = temporal_new_period_refresh(held(), new_panel(), window=window((1000, 2000)))
    with pytest.raises(CausalSerializationError):
        consume_temporal_initial_state_artifact(refreshed.export())


def test_source_state_only_refuses_a_source_law_and_a_point_state():
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


def test_source_state_only_refuses_support_gaps_and_bad_premises():
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


def test_period_replace_re_evaluates_and_a_stale_interval_never_survives():
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


def test_period_replace_without_a_prior_interval_records_none():
    refreshed = temporal_new_period_refresh(held(), new_panel(), window=window((1000, 2000)))
    assert refreshed.interval_invalidated is False
    assert refreshed.receipt.interval_invalidated is False


def test_period_replace_chains_and_exports_an_independent_artifact():
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
def test_third_period_invalidates_with_a_typed_refusal(overrides, detail):
    error = refusal_of(
        lambda: temporal_new_period_refresh(
            held(), new_panel(), window=window((1000, 2000), **overrides), interval_existed=True
        )
    )
    assert error.reason_code == "route_not_supported"
    assert error.detail == detail
    assert detail in str(error)


def test_third_period_a_replacement_that_is_not_new_is_stale():
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


def test_third_period_invalid_replacements_and_missing_windows_refuse():
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


def test_period_reseal_consumer_refuses_tampered_refresh_bytes():
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


def test_internal_candidate_route_is_closed_after_real_validation():
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


def test_dependent_interval_validates_its_arguments_before_closing():
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


def test_the_two_step_sequence_route_is_unchanged():
    # The point-only sequence route keeps its own symbols (and its interval refusal).
    assert callable(advanced.prepare_temporal_transport_sequence)
    assert callable(advanced.consume_temporal_transport_artifact)
    assert advanced.TemporalSequenceSpec.__name__ == "TemporalSequenceSpec"


# --- period refresh of a linear Gaussian mechanism ----------------------------------------------
#
# The autoregressive mechanism v_t = c + phi_1 v_{t-1} + phi_2 v_{t-2} + eps_t of one series,
# least squares and the normal-inverse-gamma posterior (m0 = 0, Lambda0 = diag(0.5, 0.4, 0.3),
# InvGamma(3, 2)) on the same rows. The expected values are numpy fits of the rows a fresh fit
# would use, so they do not restate the implementation.

SERIES = [
    0.3, 0.9, 0.4, 1.1, 0.8, 1.5, 1.0, 1.6, 1.2, 1.9, 1.1, 1.7, 1.5, 2.0, 1.4, 2.2, 1.6, 2.1,
]  # fmt: skip


def stated_prior() -> PeriodPrior:
    return PeriodPrior(
        [0.0, 0.0, 0.0], [0.5, 0, 0, 0, 0.4, 0, 0, 0, 0.3], shape=3.0, rate=2.0
    )


def period_window(start, end, snapshot, units=("u1", "u2")) -> PeriodWindow:
    return PeriodWindow(
        period=(start, end),
        snapshot_id=snapshot,
        unit_ids=units,
        lag_alignment={"v": 1},
        intervention_history=["a1=1", "a2=0"],
        selection_targets=["s_t1"],
        regimes=["observational"],
        graph_id="two_slice_graph",
        proof_id="proof_1",
        initial_law="law_1",
    )


def started(*, invariant: bool) -> PeriodRefresh:
    return PeriodRefresh.start(
        SERIES[0:6],
        series_id="v",
        lags=2,
        prior=stated_prior(),
        window=period_window(0, 6, "snap_1"),
        mechanism_invariant=invariant,
    )


def assert_fit(refresh, n, beta, posterior_mean, shape, rate):
    fit = refresh.least_squares
    assert fit.n == n
    assert fit.beta == pytest.approx(beta, abs=1e-9)
    posterior = refresh.posterior
    assert posterior.n == n
    assert posterior.mean == pytest.approx(posterior_mean, abs=1e-9)
    assert posterior.shape == pytest.approx(shape, abs=1e-12)
    assert posterior.rate == pytest.approx(rate, abs=1e-9)
    assert refresh.check_against_fresh() is True


def solve(a, b):
    """Gaussian elimination, independent of the library."""
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(a[i][col]))
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            a[row] = [x - factor * y for x, y in zip(a[row], a[col], strict=True)]
            b[row] -= factor * b[col]
    x = [0.0] * n
    for row in reversed(range(n)):
        tail = sum(a[row][k] * x[k] for k in range(row + 1, n))
        x[row] = (b[row] - tail) / a[row][row]
    return x


def posterior_mean(rows, precision0):
    """Posterior mean ``(Lambda0 + X'X)^-1 X'y`` for a zero prior mean."""
    p = len(rows[0][0])
    gram = [
        [precision0[i][j] + sum(x[i] * x[j] for x, _ in rows) for j in range(p)] for i in range(p)
    ]
    return solve(gram, [sum(x[i] * y for x, y in rows) for i in range(p)])


def lag_rows(first, last):
    return [([1.0, SERIES[t - 1], SERIES[t - 2]], SERIES[t]) for t in range(first, last)]


def test_period_refresh_first_period_matches_the_reference_fit():
    refresh = started(invariant=True)
    assert_fit(
        refresh,
        4,
        [-0.3917647058823548, 0.5058823529411782, 1.3882352941176475],
        [0.2272942518191619, 0.16164821678711794, 0.8370005826277753],
        5.0,
        2.1763865921233685,
    )
    assert [e.outcome for e in refresh.events] == ["start"]


def test_period_refresh_extends_a_declared_invariant_mechanism_like_one_fit():
    refresh = started(invariant=True)
    outcome = refresh.refresh(SERIES[6:12], period_window(6, 12, "snap_2"))
    assert (outcome.outcome, outcome.contiguous, outcome.invalidated) == ("extended", True, ())
    assert_fit(
        refresh,
        10,
        [0.47578572103621536, -0.12081675084793765, 0.8347229368587971],
        [0.3515871568250203, 0.004469720612162134, 0.7997133706820022],
        8.0,
        2.265297812709253,
    )
    # The same posterior mean from the pooled rows, solved here.
    pooled = posterior_mean(lag_rows(2, 12), [[0.5, 0, 0], [0, 0.4, 0], [0, 0, 0.3]])
    assert refresh.posterior.mean == pytest.approx(pooled, abs=1e-9)


def test_period_refresh_gap_forms_no_spanning_rows():
    refresh = started(invariant=True)
    outcome = refresh.refresh(SERIES[8:14], period_window(8, 14, "snap_later"))
    assert (outcome.outcome, outcome.contiguous) == ("extended", False)
    assert_fit(
        refresh,
        8,
        [0.3670513885592226, -0.006186209352005272, 0.8397371231901244],
        [0.2970587733457107, 0.09556683194501768, 0.7763572332881739],
        7.0,
        2.2907118116577543,
    )


def test_period_refresh_restarts_an_undeclared_mechanism_under_the_stated_prior():
    refresh = started(invariant=False)
    outcome = refresh.refresh(SERIES[6:12], period_window(6, 12, "snap_2"))
    assert outcome.outcome == "restarted"
    assert {"model", "design", "evidence"} <= set(outcome.invalidated)
    assert "identification" not in outcome.invalidated
    assert_fit(
        refresh,
        4,
        [2.4679746835440985, -0.8291139240505575, 0.14683544303804172],
        [0.24998303743073655, 0.08424742734366153, 0.7520072373628859],
        5.0,
        2.188748727807305,
    )


def test_period_refresh_population_drift_restarts_even_when_declared_invariant():
    refresh = started(invariant=True)
    outcome = refresh.refresh(SERIES[12:18], period_window(12, 18, "snap_drift", ("u1", "u3")))
    assert outcome.outcome == "restarted"
    assert "model" in outcome.invalidated
    assert refresh.least_squares.n == 4


def test_period_refresh_overlap_reuse_and_prior_change_are_typed_refusals():
    refresh = started(invariant=True)
    before = refresh.posterior
    error = refusal_of(lambda: refresh.refresh(SERIES[4:10], period_window(4, 10, "snap_overlap")))
    assert error.reason_code == "invalid_argument"
    assert error.detail == "period_refresh.overlap"
    assert "supply a later period of a new snapshot" in str(error)
    error = refusal_of(lambda: refresh.refresh(SERIES[6:12], period_window(6, 12, "snap_1")))
    assert error.detail == "period_refresh.overlap"
    assert refresh.posterior == before  # a refused period changes nothing

    other = PeriodPrior.isotropic(3, 1.0, 4.0, 1.5)
    error = refusal_of(lambda: refresh.summary_under(other))
    assert error.detail == "period_refresh.prior_changed"
    assert "rebuild from every retained block under the new prior" in str(error)
    assert refresh.summary_under(stated_prior()) == 4
    # The named next attempt: rebuild from the retained blocks under the new prior.
    refresh.reprior(other)
    rebuilt = posterior_mean(lag_rows(2, 6), [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    assert refresh.posterior.mean == pytest.approx(rebuilt, abs=1e-9)
    assert refresh.posterior.shape == pytest.approx(4.0 + 4 / 2)

    narrow = PeriodPrior.isotropic(2, 1.0, 3.0, 2.0)
    error = refusal_of(
        lambda: PeriodRefresh.start(
            SERIES[0:6],
            series_id="v",
            lags=2,
            prior=narrow,
            window=period_window(0, 6, "snap_1"),
        )
    )
    assert error.detail == "period_refresh.start"


def test_period_refresh_eviction_rebuilds_both_summaries():
    refresh = started(invariant=True)
    assert refresh.evict_row(2) == "recomputed"
    kept = lag_rows(3, 6)
    expected = posterior_mean(kept, [[0.5, 0, 0], [0, 0.4, 0], [0, 0, 0.3]])
    assert refresh.posterior.n == 3
    assert refresh.posterior.mean == pytest.approx(expected, abs=1e-9)
    assert refresh.check_against_fresh() is True
    error = refusal_of(lambda: refresh.evict_row(2))
    assert error.detail == "period_refresh.evict"


def test_period_refresh_ledger_is_recomputed_by_an_independent_consumer():
    refresh = started(invariant=True)
    refresh.refresh(SERIES[6:12], period_window(6, 12, "snap_2"))
    consumed = consume_period_refresh_artifact(refresh.export())
    assert consumed.standing == "declared_invariant"
    assert consumed.rows_used == 10
    assert consumed.live_snapshots == ("snap_1", "snap_2")
    assert [e.outcome for e in consumed.events] == ["start", "extended"]
    assert consumed.events[1].period == (6, 12)
    assert consumed.posterior.mean == pytest.approx(refresh.posterior.mean, abs=1e-9)
    assert consumed.posterior.mean == pytest.approx(
        [0.3515871568250203, 0.004469720612162134, 0.7997133706820022], abs=1e-9
    )
    assert consumed.least_squares.beta == pytest.approx(
        [0.47578572103621536, -0.12081675084793765, 0.8347229368587971], abs=1e-9
    )
    assert consumed.prior_digest == refresh.prior_digest

    restarted = started(invariant=False)
    restarted.refresh(SERIES[6:12], period_window(6, 12, "snap_2"))
    consumed = consume_period_refresh_artifact(restarted.export())
    assert [e.outcome for e in consumed.events] == ["start", "restarted"]
    assert consumed.live_snapshots == ("snap_2",)
    assert consumed.rows_used == 4


def test_period_refresh_tampered_ledger_is_refused_by_the_consumer():
    refresh = started(invariant=True)
    refresh.refresh(SERIES[6:12], period_window(6, 12, "snap_2"))
    blob = refresh.export()
    for fraction in (0.4, 0.6, 0.8):
        index = int(len(blob) * fraction)
        forged = blob[:index] + bytes([blob[index] ^ 0x01]) + blob[index + 1 :]
        with pytest.raises((CausalSerializationError, TemporalExtensionRefusal)):
            consume_period_refresh_artifact(forged)
    with pytest.raises(CausalSerializationError):
        consume_period_refresh_artifact(blob[:-5])
    with pytest.raises(CausalSerializationError):
        consume_period_refresh_artifact(b"not a ledger")
