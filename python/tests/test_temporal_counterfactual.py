"""X8 fixed-population temporal counterfactual (2.3A) and the closed transported route.

The two-slice SCM is known in closed form::

    L0 = 1.0 + u0
    L1 = 0.5 + 0.8 L0 + 1.5 A0 + u1
    Y  = -1.0 + 0.5 L0 + 0.7 A0 + 2.0 L1 + 1.2 A1 + u2

so each unit's exogenous history is recovered exactly from its factual trajectory and the
counterfactual outcome of either action history is hand-computable. The shared noise cancels
in every unit's contrast: ``0.7 + 1.2 + 2.0 * 1.5 = 4.9`` for histories ``(1, 1)`` versus
``(0, 0)``. The transported path-specific route is CLOSED: the tests assert the live refusal
and the exact missing gates, never an evaluation.
"""

from __future__ import annotations

import pytest
from antecedent.errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.temporal import TemporalRefusal
from antecedent.temporal_counterfactual import (
    ActionHistory,
    NodeMechanism,
    TemporalMechanisms,
    TransportedPrerequisites,
    UnitHistory,
    consume_temporal_counterfactual_artifact,
    temporal_fixed_population,
    transported_path_specific,
)

from _refusal import assert_registered_refusal

NOISE = [[0.3, -0.2, 0.5], [-0.4, 0.6, -0.1], [0.0, 0.1, 0.9], [1.1, -0.7, -0.3]]
FACTUAL_ACTIONS = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
PLUS = (1.0, 1.0)
MINUS = (0.0, 0.0)


def simulate(noise, actions):
    l0 = 1.0 + noise[0]
    l1 = 0.5 + 0.8 * l0 + 1.5 * actions[0] + noise[1]
    y = -1.0 + 0.5 * l0 + 0.7 * actions[0] + 2.0 * l1 + 1.2 * actions[1] + noise[2]
    return [l0, actions[0], l1, actions[1], y]


def histories(noise=NOISE, **kwargs):
    rows = []
    for i, (u, a) in enumerate(zip(noise, FACTUAL_ACTIONS, strict=True)):
        l0, a0, l1, a1, y = simulate(u, a)
        rows.append(UnitHistory(f"unit-{i}", l0, a0, l1, a1, y, history=f"hist-{i}", **kwargs))
    return rows


def mechanisms(fit_id="fit-known-coefficients", halfwidth=None):
    return TemporalMechanisms(
        fit_id,
        (
            NodeMechanism("covariate_0", 1.0, {}, halfwidth),
            NodeMechanism("covariate_1", 0.5, {"covariate_0": 0.8, "action_0": 1.5}, halfwidth),
            NodeMechanism(
                "outcome",
                -1.0,
                {"covariate_0": 0.5, "action_0": 0.7, "covariate_1": 2.0, "action_1": 1.2},
                halfwidth,
            ),
        ),
    )


def run(**overrides):
    arguments = {
        "plus": ActionHistory("always_treat", PLUS),
        "minus": ActionHistory("never_treat", MINUS),
        "mechanisms": mechanisms(),
        "snapshot": "snapshot-1",
    }
    units = overrides.pop("histories", None)
    arguments.update(overrides)
    return temporal_fixed_population(units if units is not None else histories(), **arguments)


def refusal_of(**overrides) -> TemporalRefusal:
    with pytest.raises(TemporalRefusal) as raised:
        run(**overrides)
    assert isinstance(raised.value, CausalUnsupportedError)
    assert_registered_refusal(raised.value)
    return raised.value


def test_x8_shared_history_matches_hand_abduction_and_replay() -> None:
    result = run()
    assert [u.unit for u in result.units] == [f"unit-{i}" for i in range(4)]
    plus_sum = minus_sum = 0.0
    for i, row in enumerate(result.units):
        truth_plus = simulate(NOISE[i], PLUS)[4]
        truth_minus = simulate(NOISE[i], MINUS)[4]
        assert row.plus_outcome == pytest.approx(truth_plus, abs=1e-9), i
        assert row.minus_outcome == pytest.approx(truth_minus, abs=1e-9), i
        assert row.factual_outcome == pytest.approx(
            simulate(NOISE[i], FACTUAL_ACTIONS[i])[4], abs=1e-9
        )
        assert row.contrast == pytest.approx(4.9, abs=1e-9)
        plus_sum += truth_plus
        minus_sum += truth_minus
    # Unit 0 by hand: L0 = 1.3, L1 = 0.5 + 1.04 + 1.5 - 0.2 = 2.84,
    # Y(1,1) = -1 + 0.65 + 0.7 + 5.68 + 1.2 + 0.5 = 7.73, Y(0,0) = -1 + 0.65 + 2.68 + 0.5 = 2.83.
    assert result.units[0].plus_outcome == pytest.approx(7.73, abs=1e-9)
    assert result.units[0].minus_outcome == pytest.approx(2.83, abs=1e-9)
    assert result.mean_plus == pytest.approx(plus_sum / 4.0, abs=1e-9)
    assert result.mean_minus == pytest.approx(minus_sum / 4.0, abs=1e-9)
    assert result.contrast == pytest.approx(4.9, abs=1e-9)
    assert (result.plus_name, result.minus_name) == ("always_treat", "never_treat")
    assert result.inference_claim == "point_only"
    assert result.mechanism_class == "linear_gaussian_additive_noise"
    assert result.snapshot == "snapshot-1"


def test_x8_receipt_records_one_shared_draw_per_unit() -> None:
    receipt = run().receipt
    assert receipt.shared_by_both_worlds
    assert (receipt.n_units, receipt.n_worlds, receipt.horizon) == (4, 2, 2)
    assert [d[0] for d in receipt.unit_draws] == [f"unit-{i}" for i in range(4)]
    assert [d[1] for d in receipt.unit_draws] == [f"hist-{i}" for i in range(4)]
    assert len({d[2] for d in receipt.unit_draws}) == 4
    assert receipt.snapshot == "snapshot-1"
    assert receipt.plus_digest != receipt.minus_digest
    assert len(receipt.digest) == 32


def test_x8_independent_noise_would_answer_a_different_question() -> None:
    result = run()
    for i, row in enumerate(result.units):
        fresh_minus = simulate(NOISE[(i + 1) % 4], MINUS)[4]
        # The shared-draw contrast is constant (4.9); a fresh draw in the other world is not.
        assert abs((row.plus_outcome - fresh_minus) - 4.9) > 1e-3, i
    independent_mean = (
        sum(row.plus_outcome - simulate([0.9, 0.9, 0.9], MINUS)[4] for row in result.units) / 4.0
    )
    assert abs(independent_mean - result.contrast) > 1e-3


def test_x8_unit_order_does_not_change_the_answer() -> None:
    base = run()
    again = run(histories=list(reversed(histories())))
    assert again.units == base.units
    assert again.receipt == base.receipt
    assert again.identity == base.identity


def test_x8_unpaired_histories_refuse_with_the_offending_unit_as_witness() -> None:
    error = refusal_of(
        plus=ActionHistory("always_treat", PLUS, units=("unit-0", "unit-1", "unit-2"))
    )
    assert error.reason_code == "route_not_supported"
    assert error.detail == "temporal_counterfactual.unpaired_histories"
    assert error.offending == "unit-3"
    assert error.witness["unit"] == "unit-3"
    assert error.witness["world"] == "always_treat"
    assert error.witness["history"] == "hist-3"

    ghost = refusal_of(
        minus=ActionHistory(
            "never_treat", MINUS, units=tuple(f"unit-{i}" for i in range(4)) + ("ghost",)
        )
    )
    assert ghost.detail == "temporal_counterfactual.unpaired_histories"
    assert ghost.witness["unit"] == "ghost"
    assert ghost.witness["world"] == "never_treat"
    assert ghost.witness["history"] is None

    duplicated = histories() + [histories()[0]]
    assert refusal_of(histories=duplicated).detail == "temporal_counterfactual.unpaired_histories"


def test_x8_misaligned_times_and_missing_histories_refuse() -> None:
    error = refusal_of(plus=ActionHistory("always_treat", PLUS, times=(0, 2)))
    assert error.reason_code == "route_not_supported"
    assert error.detail == "temporal_counterfactual.time_misaligned"
    assert error.witness["unit"] == "unit-0"
    assert error.witness["world"] == "always_treat"
    empty = refusal_of(
        histories=[],
        plus=ActionHistory("always_treat", PLUS, units=()),
        minus=ActionHistory("never_treat", MINUS, units=()),
    )
    assert empty.detail == "temporal_counterfactual.shared_history_missing"


def test_x8_refuting_history_retains_node_residual_and_bound() -> None:
    # Unit 0's outcome noise is 0.5, outside the declared half-width 0.45.
    error = refusal_of(mechanisms=mechanisms(halfwidth=0.45))
    assert error.reason_code == "route_not_supported"
    assert error.detail == "temporal_counterfactual.refuting_history"
    assert error.witness["unit"] == "unit-0"
    assert error.witness["history"] == "hist-0"
    assert error.witness["node"] == "outcome"
    assert error.witness["residual"] == pytest.approx(0.5, abs=1e-9)
    assert error.witness["bound"] == 0.45
    # A bound wide enough for every noise term admits every history.
    assert run(mechanisms=mechanisms(halfwidth=2.0)).contrast == pytest.approx(4.9, abs=1e-9)


def test_x8_structure_outside_the_theorem_refuses() -> None:
    error = refusal_of(latent_confounding=True)
    assert (error.reason_code, error.detail) == (
        "route_not_supported",
        "temporal_counterfactual.latent_confounding",
    )
    error = refusal_of(horizon=3)
    assert (error.reason_code, error.detail) == (
        "invalid_argument",
        "temporal_counterfactual.horizon_exceeded",
    )
    partial = TemporalMechanisms("partial-fit", mechanisms().mechanisms[:2])
    error = refusal_of(mechanisms=partial)
    assert (error.reason_code, error.detail) == (
        "invalid_argument",
        "temporal_counterfactual.fit_mismatch",
    )
    same_name = refusal_of(minus=ActionHistory("always_treat", MINUS))
    assert same_name.detail == "temporal_counterfactual.history_name_invalid"
    unknown = TemporalMechanisms(
        "unknown-node", (*mechanisms().mechanisms[:2], NodeMechanism("revenue", 0.0, {}))
    )
    assert refusal_of(mechanisms=unknown).detail == "temporal_counterfactual.invalid_graph"


def test_x8_non_finite_inputs_refuse_and_bad_types_raise() -> None:
    with pytest.raises(TemporalRefusal) as raised:
        UnitHistory("u", float("nan"), 0.0, 0.0, 0.0, 0.0)
    assert raised.value.detail == "temporal_counterfactual.non_finite_history"
    assert raised.value.reason_code == "invalid_argument"
    with pytest.raises(TemporalRefusal) as raised:
        ActionHistory("w", (float("inf"), 0.0))
    assert raised.value.detail == "temporal_counterfactual.non_finite_history"
    with pytest.raises(CausalTypeError):
        run(plus="always_treat")
    with pytest.raises(CausalTypeError):
        temporal_fixed_population(
            "units",  # type: ignore[arg-type]
            plus=ActionHistory("p", PLUS),
            minus=ActionHistory("m", MINUS),
            mechanisms=mechanisms(),
            snapshot="s",
        )
    with pytest.raises(CausalValueError):
        UnitHistory("u", 0.0, 0.0, 0.0, 0.0, 0.0, times=(-1, 0))


def test_x8_artifact_round_trip_replays_both_worlds() -> None:
    result = run()
    artifact = result.export()
    assert isinstance(artifact, bytes)
    fresh = consume_temporal_counterfactual_artifact(artifact, expected=result.identity)
    assert fresh.units == result.units
    assert fresh.contrast == result.contrast
    assert fresh.receipt == result.receipt
    assert fresh.identity == result.identity
    assert fresh.export() == artifact
    assert consume_temporal_counterfactual_artifact(artifact).identity == result.identity
    assert consume_temporal_counterfactual_artifact(artifact, expected=result.identity._wire())


def test_x8_resealed_mutations_are_refused_against_the_retained_identity() -> None:
    original = run()

    def refused(mutated, field: str) -> None:
        # Alone the mutated artifact is internally consistent and replays ...
        assert consume_temporal_counterfactual_artifact(mutated.export()).units == mutated.units
        # ... but against the identity the consumer retained it is refused.
        with pytest.raises(TemporalRefusal) as raised:
            consume_temporal_counterfactual_artifact(mutated.export(), expected=original.identity)
        assert raised.value.reason_code == "route_not_supported"
        assert raised.value.detail == "temporal_counterfactual.artifact_changed"
        assert raised.value.offending == field, field

    # A different but self-consistent trajectory for unit 2.
    changed_noise = [list(n) for n in NOISE]
    changed_noise[2] = [0.4, 0.4, 0.4]
    refused(run(histories=histories(changed_noise)), "unit_history")
    # Every time moved consistently: the unit-history digest binds the observation times.
    refused(
        run(
            histories=histories(times=(0, 2)),
            plus=ActionHistory("always_treat", PLUS, times=(0, 2)),
            minus=ActionHistory("never_treat", MINUS, times=(0, 2)),
        ),
        "unit_history",
    )
    refused(run(plus=ActionHistory("always_treat", (1.0, 0.0))), "action_history")
    refused(run(plus=ActionHistory("treat_early", PLUS)), "action_history")
    refused(run(snapshot="snapshot-2"), "snapshot")
    refused(run(mechanisms=mechanisms(fit_id="fit-refit")), "mechanism_fit")


def test_x8_corrupt_truncated_and_foreign_artifacts_refuse() -> None:
    artifact = run().export()
    corrupt = bytearray(artifact)
    corrupt[len(corrupt) // 2] ^= 0xFF
    with pytest.raises(CausalSerializationError):
        consume_temporal_counterfactual_artifact(bytes(corrupt))
    with pytest.raises(CausalSerializationError):
        consume_temporal_counterfactual_artifact(artifact[:-5])
    with pytest.raises(CausalSerializationError):
        consume_temporal_counterfactual_artifact(b"not an artifact")
    with pytest.raises(CausalTypeError):
        consume_temporal_counterfactual_artifact("text")  # type: ignore[arg-type]


# --- The transported route is closed ---------------------------------------------------------


def transported(**kwargs) -> TemporalRefusal:
    with pytest.raises(TemporalRefusal) as raised:
        transported_path_specific(**kwargs)
    assert isinstance(raised.value, CausalUnsupportedError)
    assert_registered_refusal(raised.value)
    return raised.value


GATES = (
    "transport_license_missing",
    "fixed_population_temporal_license_missing",
    "cross_population_assumptions_missing",
)


def test_x8_transported_route_is_closed_with_every_missing_gate() -> None:
    error = transported(required_factors=["source:rct", "target:field"])
    assert error.reason_code == "cell_not_licensed"
    assert error.detail == "transported_counterfactual.route_frozen"
    assert error.offending == "transport_license_missing"
    assert error.missing_gates == GATES
    assert error.missing_factors == ("source:rct", "target:field")
    assert error.stage == "identify"
    assert error.remedy is not None
    assert "transported_counterfactual.route_frozen" in str(error)


def test_x8_transported_route_reports_only_the_gates_still_missing() -> None:
    error = transported(prerequisites=TransportedPrerequisites(transport_license=True))
    assert error.detail == "transported_counterfactual.route_frozen"
    assert error.offending == "fixed_population_temporal_license_missing"
    assert error.missing_gates == GATES[1:]
    error = transported(
        prerequisites=TransportedPrerequisites(
            transport_license=True, fixed_population_license=True
        )
    )
    assert error.offending == "cross_population_assumptions_missing"
    assert error.missing_gates == GATES[2:]


def test_x8_transported_route_with_a_missing_gate_never_reports_factors_as_the_cause() -> None:
    error = transported(
        prerequisites=TransportedPrerequisites(transport_license=True),
        required_factors=["source:rct", "target:field"],
        supplied_factors={},
    )
    assert error.reason_code == "cell_not_licensed"
    assert error.detail == "transported_counterfactual.route_frozen"
    assert error.missing_factors == ("source:rct", "target:field")


ALL_PASSED = TransportedPrerequisites(True, True, True)


def test_x8_transported_route_names_the_missing_regime_factor() -> None:
    error = transported(
        prerequisites=ALL_PASSED,
        required_factors=[("target", "field"), ("source", "rct")],
        supplied_factors={("target", "field"): "evidence-field"},
    )
    assert error.reason_code == "transport_missing_evidence"
    assert error.detail == "transported_counterfactual.factor_missing"
    assert error.offending == "source:rct"
    assert error.missing_factors == ("source:rct",)
    assert error.missing_gates == ()


def test_x8_transported_route_stays_closed_even_when_every_gate_and_factor_is_present() -> None:
    error = transported(
        prerequisites=ALL_PASSED,
        required_factors=["source:rct", "target:field"],
        supplied_factors={"source:rct": "e1", "target:field": "e2"},
    )
    assert error.reason_code == "cell_not_licensed"
    assert error.detail == "transported_counterfactual.route_frozen"
    assert error.offending is None
    assert error.missing_gates == ()
    assert error.missing_factors == ()


def test_x8_transported_route_rejects_malformed_factors() -> None:
    with pytest.raises(CausalValueError):
        transported_path_specific(required_factors=["population:rct"])
    with pytest.raises(CausalValueError):
        transported_path_specific(required_factors=["source:"])
    with pytest.raises(CausalTypeError):
        transported_path_specific(required_factors=[1])  # type: ignore[list-item]
