"""F17: an assumption-sensitivity surface consumed by a decision contract.

Every expected value is derived by hand from the declared surfaces; the fixtures mirror
`crates/antecedent-design/tests/sensitivity_decision.rs`:

* frozen surface (record ``2.3B.F17.sensitivity_decision_composition``): ``gamma`` in
  ``{0, 1, 2}``; action A has utility ``2 - gamma`` (2, 1, 0) and action B has utility 1,
  so A leads at 0, they tie at 1 and B leads at 2;
* invariant surface: A is ``5 - gamma`` (5, 4, 3) against B = 1, so A leads everywhere;
* closed-form tipping: grid ``{0, 0.5, 1.5, 2}``, A = ``2 - gamma``, B = 1, so the
  difference ``1 - gamma`` crosses zero at ``gamma = 1``, between the grid points;
* no robust action: an effect whose range is ``[-1, 2]`` at every ``gamma`` against a status
  quo of 0, so the best action depends on where in the range the surface lies.
"""

from __future__ import annotations

import pytest
from antecedent import Admg
from antecedent import sensitivity_decision as sd
from antecedent.decision import Contract
from antecedent.errors import CausalTypeError, CausalUnsupportedError
from antecedent.joint_distribution import ScientificQuantity
from antecedent.transport import advanced as transport


def _quantity(name: str, units: str = "utils") -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=name,
        variable_name=name,
        role="outcome",
        units=units,
        population_id="target",
        regime_id="do(a=1)",
        horizon=0,
        functional_id="sensitivity_surface",
    )


def _provenance() -> sd.SurfaceProvenance:
    return sd.SurfaceProvenance(
        source_kind="supplied_surface",
        query_binding="f17-test",
        provider_snapshot="snapshot-1",
        source_regime="regime:1",
        method="hand-derived surface",
        causal_contract_id="checked-contract",
    )


def _artifact(
    grid: list[float],
    quantities: list[sd.SurfaceQuantity],
    actions: list[sd.Action],
    **kwargs: object,
) -> sd.SensitivityArtifact:
    coordinate = sd.AssumptionCoordinate(
        "gamma", "sensitivity_parameter", "dimensionless", 0.0, max(grid)
    )
    return sd.SensitivityArtifact.from_surface(
        coordinate=coordinate,
        grid=grid,
        quantities=quantities,
        actions=actions,
        provenance=_provenance(),
        **kwargs,  # type: ignore[arg-type]
    )


def _point(name: str, values: list[float]) -> sd.SurfaceQuantity:
    return sd.SurfaceQuantity(_quantity(name), tuple(values))


def _frozen() -> sd.SensitivityArtifact:
    return _artifact(
        [0.0, 1.0, 2.0],
        [_point("ua", [2.0, 1.0, 0.0]), _point("ub", [1.0, 1.0, 1.0])],
        [sd.Action("A", sd.quantity("ua")), sd.Action("B", sd.quantity("ub"))],
    )


def test_f17_frozen_surface_is_an_assumption_dependent_switch_with_its_tipping_coordinate():
    artifact = _frozen()
    result = sd.decide(artifact.contract(), artifact)
    assert result.kind == "assumption_dependent"
    assert not result.robust
    assert result.invariant_action is None
    switch = result.switch
    assert switch is not None
    # A wins at gamma=0, they tie exactly at 1, B wins at 2: one switch, an exact tie at 1.
    assert (switch.from_actions, switch.to_actions) == (("A",), ("B",))
    assert switch.exact
    assert switch.bracket == (1.0, 1.0)
    assert switch.tipping_coordinate == 1.0
    assert result.coordinates == (0.0, 1.0, 2.0)
    assert result.coverage == "point_surface"
    # Per grid point leaders by hand: {A}, {A, B}, {B}.
    leaders = [tuple(sorted(atom.leaders)) for atom in result.atoms]
    assert leaders == [("A",), ("A", "B"), ("B",)]
    assert result.structural_verdict["kind"] == "no_invariant_best"
    # The classification is also recomputable from the artifact alone, and agrees.
    assert artifact.outcome == result.outcome
    assert artifact.outcome_over() == result.outcome
    assert "changes from 'A' to 'B'" in result.explain()
    assert "not a probability law" in result.interpretation


def test_f17_invariant_action_across_the_whole_assumption_range():
    artifact = _artifact(
        [0.0, 1.0, 2.0],
        [_point("ua", [5.0, 4.0, 3.0]), _point("ub", [1.0, 1.0, 1.0])],
        [sd.Action("A", sd.quantity("ua")), sd.Action("B", sd.quantity("ub"))],
    )
    result = sd.decide(artifact.contract(), artifact)
    assert result.kind == "invariant_action"
    assert result.robust
    assert result.invariant_action == "A"
    assert result.switch is None
    assert result.structural_verdict == {"kind": "invariant_best", "action": "A"}
    # A coordinate sub-range is evaluated on its own.
    sub = sd.decide(artifact.contract(), artifact, range=(1.0, 2.0))
    assert sub.coordinates == (1.0, 2.0)
    assert sub.invariant_action == "A"


def test_f17_tipping_coordinate_is_the_labelled_interpolated_crossing():
    # Difference 1 - gamma is zero at gamma = 1, between the grid points 0.5 and 1.5.
    artifact = _artifact(
        [0.0, 0.5, 1.5, 2.0],
        [_point("ua", [2.0, 1.5, 0.5, 0.0]), _point("ub", [1.0, 1.0, 1.0, 1.0])],
        [sd.Action("A", sd.quantity("ua")), sd.Action("B", sd.quantity("ub"))],
    )
    result = sd.decide(artifact.contract(), artifact)
    switch = result.switch
    assert result.kind == "assumption_dependent"
    assert switch is not None
    assert not switch.exact
    assert switch.bracket == (0.5, 1.5)
    assert switch.interpolated == pytest.approx(1.0, abs=1e-12)
    assert switch.tipping_coordinate == switch.interpolated
    assert "interpolated crossing" in result.explain()


def test_f17_no_robust_action_when_the_range_straddles_at_every_point():
    artifact = _artifact(
        [0.0, 1.0, 2.0],
        [
            sd.SurfaceQuantity(_quantity("effect"), (-1.0,) * 3, (2.0,) * 3),
            _point("status_quo", [0.0, 0.0, 0.0]),
        ],
        [sd.Action("A", sd.quantity("effect")), sd.Action("B", sd.quantity("status_quo"))],
    )
    result = sd.decide(artifact.contract(), artifact)
    assert result.kind == "no_robust_action"
    assert result.outcome.reason == "mixed_within_range"
    assert result.outcome.coordinates == (0.0, 1.0, 2.0)
    assert not result.robust
    # Two vertex scenarios per grid point; the utilities are multilinear, so the vertices
    # certify the whole range at each grid point.
    assert len(result.atoms) == 6
    assert result.coverage == "vertex_certified"
    assert artifact.quantities[0].ranged
    assert result.structural_verdict["kind"] == "no_invariant_best"


def test_f17_range_is_distinct_from_a_sampling_interval_and_never_composed():
    def ranged(sampling: object = None) -> sd.SensitivityArtifact:
        return _artifact(
            [0.0, 1.0, 2.0],
            [
                sd.SurfaceQuantity(_quantity("effect"), (0.5, 0.4, 0.3), (1.5, 1.4, 1.3)),
                _point("status_quo", [0.0, 0.0, 0.0]),
            ],
            [sd.Action("A", sd.quantity("effect")), sd.Action("B", sd.quantity("status_quo"))],
            sampling=sampling,
        )

    plain = ranged()
    interval = sd.SamplingInterval(
        "effect", 0.95, "percentile_bootstrap", lower=(-2.0,) * 3, upper=(3.0,) * 3
    )
    reported = ranged(interval)
    without = sd.decide(plain.contract(), plain)
    with_interval = sd.decide(reported.contract(), reported)
    # A leads under the assumption range; the far wider sampling interval would flip the
    # decision if it were merged into the range, so it must not be.
    assert without.invariant_action == "A"
    assert with_interval.outcome == without.outcome
    assert len(with_interval.atoms) == len(without.atoms)
    assert plain.identity["digest"] != reported.identity["digest"]
    assert without.sampling.status == "withheld"
    report = with_interval.sampling
    assert report.status == "separate_not_composed"
    assert not report.composed
    assert (report.level, report.method) == (0.95, "percentile_bootstrap")
    assert report.lower == (-2.0,) * 3
    assert report.upper == (3.0,) * 3
    assert "not composed" in with_interval.explain()

    # The three uncertainty kinds stay in three fields.
    uncertainty = reported.uncertainty
    assert "not a probability" in uncertainty.assumption_range
    assert uncertainty.identified_bound is None
    assert isinstance(uncertainty.sampling, sd.SamplingInterval)

    # Asking for the composition refuses: no method is licensed.
    with pytest.raises(sd.SensitivityRefusal) as composed:
        sd.decide(
            reported.contract(),
            reported,
            sampling_composition="conservative_endpoint_percentile_bootstrap",
        )
    assert isinstance(composed.value, CausalUnsupportedError)
    assert composed.value.reason_code == "cell_not_licensed"
    assert composed.value.detail == "sensitivity_decision_composition.composition_not_licensed"

    # An artifact that claims a composition is refused at construction.
    claimed = sd.SamplingInterval(
        "ua", 0.95, "percentile_bootstrap", lower=(0.0,) * 3, upper=(1.0,) * 3, composed="x"
    )
    with pytest.raises(sd.SensitivityRefusal) as at_construction:
        _artifact(
            [0.0, 1.0, 2.0],
            [_point("ua", [2.0, 1.0, 0.0]), _point("ub", [1.0, 1.0, 1.0])],
            [sd.Action("A", sd.quantity("ua")), sd.Action("B", sd.quantity("ub"))],
            sampling=claimed,
        )
    assert at_construction.value.reason_code == "cell_not_licensed"

    # An assumption range is never a probability: weights on a ranged surface refuse.
    with pytest.raises(sd.SensitivityRefusal) as weights:
        sd.decide(plain.contract(), plain, weights=[0.2, 0.3, 0.5])
    assert weights.value.reason_code == "decision_contract_unsatisfied"
    assert weights.value.detail == "sensitivity_decision_composition.wrong_contract"


def test_f17_unsupported_coordinate_refuses_and_a_subrange_avoids_it():
    artifact = _artifact(
        [0.0, 1.0, 2.0],
        [_point("ua", [5.0, 4.0, 3.0]), _point("ub", [1.0, 1.0, 1.0])],
        [sd.Action("A", sd.quantity("ua")), sd.Action("B", sd.quantity("ub"))],
        support=["supported", "unsupported", "supported"],
    )
    assert artifact.outcome is None
    with pytest.raises(sd.SensitivityRefusal) as unsupported:
        sd.decide(artifact.contract(), artifact)
    assert unsupported.value.reason_code == "quantity_semantics_mismatch"
    assert unsupported.value.detail == "sensitivity_decision_composition.unsupported_coordinate"
    avoided = sd.decide(artifact.contract(), artifact, range=(0.0, 0.0))
    assert avoided.invariant_action == "A"

    unevaluated = _artifact(
        [0.0, 1.0, 2.0],
        [_point("ua", [5.0, 4.0, 3.0]), _point("ub", [1.0, 1.0, 1.0])],
        [sd.Action("A", sd.quantity("ua")), sd.Action("B", sd.quantity("ub"))],
        support=["supported", "unevaluated", "supported"],
    )
    result = sd.decide(unevaluated.contract(), unevaluated)
    # An unevaluated point leaves the decision unresolved; invariance is not assumed.
    assert result.kind == "unresolved"
    assert result.outcome.unresolved_coordinate == 1.0


def test_f17_mixed_estimands_and_foreign_inputs_refuse():
    with pytest.raises(sd.SensitivityRefusal) as mixed:
        _artifact(
            [0.0, 1.0],
            [_point("ua", [1.0, 2.0]), sd.SurfaceQuantity(_quantity("kg", "kg"), (1.0, 2.0))],
            [
                sd.Action("A", sd.quantity("ua") + sd.quantity("kg")),
                sd.Action("B", sd.quantity("ua")),
            ],
        )
    assert mixed.value.detail == "sensitivity_decision_composition.wrong_contract"
    assert mixed.value.reason_code == "decision_contract_unsatisfied"

    artifact = _frozen()
    own = artifact.contract()
    foreign = Contract(
        actions=own.actions,
        utility_units="other_units",
        criterion=own.criterion,
        target_population="target",
    )
    with pytest.raises(sd.SensitivityRefusal) as units:
        sd.decide(foreign, artifact)
    assert units.value.detail == "sensitivity_decision_composition.wrong_contract"
    with pytest.raises(CausalTypeError):
        sd.decide("not a contract", artifact)  # type: ignore[arg-type]


def test_f17_artifact_is_exported_consumed_and_refuses_a_resealed_mutation():
    artifact = _frozen()
    data = artifact.export()
    consumed = sd.SensitivityArtifact.consume(data, expected_identity=artifact.identity)
    # Premises and data digests are kept apart and survive the round trip.
    assert consumed.identity == artifact.identity
    assert set(artifact.identity) == {"premises_digest", "data_digest", "digest"}
    assert consumed.outcome == artifact.outcome
    assert consumed.grid == artifact.grid == (0.0, 1.0, 2.0)
    assert consumed.coordinate == artifact.coordinate
    assert consumed.provenance == artifact.provenance
    assert [a.id for a in consumed.actions] == ["A", "B"]
    assert sd.decide(consumed.contract(), consumed).outcome == artifact.outcome

    # A different surface is a self-consistent artifact, but not under the identity the
    # consumer retained: the resealed change is refused.
    other = _artifact(
        [0.0, 1.0, 2.0],
        [_point("ua", [5.0, 4.0, 3.0]), _point("ub", [1.0, 1.0, 1.0])],
        [sd.Action("A", sd.quantity("ua")), sd.Action("B", sd.quantity("ub"))],
    )
    resealed = other.export()
    sd.SensitivityArtifact.consume(resealed)
    with pytest.raises(sd.SensitivityRefusal) as changed:
        sd.SensitivityArtifact.consume(resealed, expected_identity=artifact.identity)
    assert changed.value.detail == "sensitivity_decision_composition.wrong_contract"
    assert "retained identity" in str(changed.value)
    assert other.identity["premises_digest"] == artifact.identity["premises_digest"]
    assert other.identity["data_digest"] != artifact.identity["data_digest"]

    # A byte flipped inside the container is caught by its checksums.
    corrupted = bytearray(data)
    corrupted[len(corrupted) // 2] ^= 0xFF
    with pytest.raises(sd.SensitivityRefusal):
        sd.SensitivityArtifact.consume(bytes(corrupted))
    with pytest.raises(CausalTypeError):
        sd.SensitivityArtifact.consume("not bytes")  # type: ignore[arg-type]


# --- a 2.2 joint-sensitivity result feeds a decision contract ----------------------

P_W = {0: 0.65, 1: 0.35}
P_Y1 = {(0, 0): 0.2, (0, 1): 0.5, (1, 0): 0.3, (1, 1): 0.8}


def _prepared_stage() -> object:
    names = ["w", "z", "x", "y"]
    graph = Admg.from_edges(
        names,
        [("w", "z"), ("z", "x"), ("x", "y"), ("w", "y")],
        [("w", "y"), ("z", "y"), ("z", "x")],
    )
    query = transport.ZTransportQuery(
        transport.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
        controllable=["z"],
        experiment_assignment={"z": 0.0},
    )
    builder = transport.identify_z_transport(graph=graph, query=query)
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    catalog = transport.EvidenceCatalog(
        environments=(transport.Environment("source", coordinates),),
        regimes=(
            transport.EvidenceRegime(
                "do_z_0",
                "source",
                kind="experimental",
                interventions=["z"],
                intervention_values={"z": 0.0},
                measured=names,
            ),
        ),
        bindings=(
            transport.RegimeBinding(
                "do_z_0",
                "snapshot_z0",
                schema_names=names,
                sampling="independent",
                dependence="independent_studies",
            ),
        ),
    )
    probabilities = []
    for w in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                p_y = P_Y1[(w, x)] if y else 1.0 - P_Y1[(w, x)]
                probabilities.append(P_W[w] * (0.4 if x else 0.6) * p_y)
    law = transport.ExactDiscreteLaw(
        "source",
        "do_z_0",
        (("w", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        tuple(probabilities),
        "snapshot_z0",
        interventions=(("z", 0.0),),
    )
    return builder.prepare_exact(catalog, (law,), {"x": 1.0})


def test_f17_a_2_2_joint_sensitivity_result_feeds_a_decision_contract():
    stage = _prepared_stage()
    stage.estimate()  # type: ignore[attr-defined]
    deviation = transport.JointDeviation({"outcome_kernel": 0.2})
    effect = _quantity("effect", "outcome_units")
    artifact = sd.SensitivityArtifact.from_joint_sensitivity(
        stage,
        deviation,
        effect=effect,
        actions=[
            sd.Action("adopt", sd.quantity("effect")),
            sd.Action("status_quo", sd.const(0.45)),
        ],
        causal_contract_id="checked-contract",
    )
    # The surface is the exact 2.2 range scaled linearly in the contamination fraction.
    baseline = 0.65 * 0.3 + 0.35 * 0.5
    delta = {w: P_Y1[(w, 1)] - P_Y1[(w, 0)] for w in (0, 1)}
    kernel = 0.2
    upper = sum(P_W[w] * ((1 - kernel) * delta[w] + kernel) for w in (0, 1))
    lower = sum(P_W[w] * ((1 - kernel) * delta[w] - kernel) for w in (0, 1))
    surface = artifact.quantities[0]
    assert artifact.coordinate.id == "outcome_kernel"
    assert artifact.coordinate.scale == "contamination_fraction"
    assert artifact.grid[0] == 0.0
    assert artifact.grid[-1] == 0.2
    assert len(artifact.grid) == 9
    assert surface.lower[0] == pytest.approx(baseline, abs=1e-12)
    assert surface.upper[0] == pytest.approx(baseline, abs=1e-12)
    assert surface.lower[-1] == pytest.approx(lower, abs=1e-12)
    assert surface.upper[-1] == pytest.approx(upper, abs=1e-12)
    # The 2.2 withheld sampling status is carried unchanged: still an assumption range.
    assert isinstance(artifact.uncertainty.sampling, sd.SamplingWithheld)
    assert artifact.uncertainty.sampling.reason_code == "cell_not_licensed"
    assert artifact.provenance.source_kind == "joint_mechanism_sensitivity_2_2"
    # Adopting is best only if the effect clears 0.45. The unperturbed baseline (0.37) does
    # not, and the range's upper end (0.37 + 0.63 t) clears it for t > 0.127, so the best
    # action depends on where in the range the surface lies at fractions 0.15, 0.175, 0.2.
    result = sd.decide(artifact.contract(), artifact)
    assert result.kind == "no_robust_action"
    assert result.outcome.reason == "mixed_within_range"
    assert result.outcome.coordinates == pytest.approx((0.15, 0.175, 0.2), abs=1e-12)
    # Restricting to the fractions where the range stays below 0.45 gives an invariant action.
    cautious = sd.decide(artifact.contract(), artifact, range=(0.0, 0.13))
    assert cautious.invariant_action == "status_quo"
    # And the whole artifact is durable.
    consumed = sd.SensitivityArtifact.consume(
        artifact.export(), expected_identity=artifact.identity
    )
    assert sd.decide(consumed.contract(), consumed).outcome == result.outcome
