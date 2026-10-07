"""C1: claims bound to the identified program, and native responses as decision inputs.

Hand example. Program: outcome ``y`` (mmHg) under ``do(a = d)`` for the grid
``d in {1, 2}`` in the target population. Native means: ``E[y | do(1)] = 3`` and
``E[y | do(2)] = 5``. Action A reads the dose-1 mean (utility ``m1``); action B reads
the dose-2 mean (utility ``m2 - 1.5``). Expected utility is A = 3.0 and B = 3.5, so B
is uniquely optimal. ``P(utility_B <= 3) = 1/2`` needs aligned draws of both
coordinates; a response view retains none, so Python must refuse it (the Rust test
``crates/antecedent/tests/native_claims.rs`` asserts the 0.5 on retained draws).
"""

from __future__ import annotations

import dataclasses

import antecedent as ac
import pytest
from antecedent import decision, external, program_claims
from antecedent import inverse_query as iq
from antecedent.errors import CausalUnsupportedError
from antecedent.extensibility import ProviderTrust
from antecedent.external import ExternalRefusal
from antecedent.joint_distribution import ScientificQuantity
from antecedent.results._views import IdentificationView
from antecedent.results.response import (
    CausalResponseView,
    ResponseUncertainty,
    ResponseView,
    SupportReport,
)

GRID = [1.0, 2.0]
EDGES = [("x", "a"), ("x", "y"), ("a", "y")]
NAMES = ["x", "a", "y"]


def _ident(grid=GRID):
    return ac.identify(graph=EDGES, names=NAMES, query=ac.ResponseCurve("a", "y", grid=grid))


def _spec(**kwargs):
    return external.response(
        _ident(), outcome_units="mmHg", dose_units="mg", population="target", **kwargs
    )


def _program(**kwargs):
    return program_claims.ProgramBinding.from_identification(
        _ident(), outcome_units="mmHg", dose_units="mg", **kwargs
    )


def _provider():
    return external.ProviderObject(
        provider_id="lab",
        object_id="curve",
        version="v3",
        snapshot="snap-9",
        request="req-1",
        meaning="interventional_predictive",
        capabilities=("mean",),
    )


def _response():
    return external.Response(provider=_provider(), values=[3.0, 5.0], attested_by="lab")


def _q(dose: str) -> ScientificQuantity:
    """The coordinate written out by hand, independent of the production derivation."""
    return ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="mmHg",
        population_id="target",
        regime_id=f"do(a={dose})",
        horizon=0,
        functional_id="mean",
        conditioning=(),
        transform_id="identity",
    )


def _view(
    *,
    grid=(1.0, 2.0),
    means=(3.0, 5.0),
    point_status=("supported", "supported"),
    support="supported",
    identification="NonparametricallyIdentified",
    snapshot="snap-1",
    uncertainty="none",
) -> CausalResponseView:
    return CausalResponseView(
        estimand=ac.ResponseCurve("a", "y", grid=list(grid)),
        response=ResponseView(
            treatments=["a"],
            outcomes=["y"],
            points=[[g] for g in grid],
            values=[[m] for m in means],
        ),
        estimate=None,
        uncertainty=ResponseUncertainty(kind=uncertainty),
        support=SupportReport(
            status=support,
            query_region={"a": (min(grid), max(grid))},
            point_status=point_status,
        ),
        identification=IdentificationView(
            status=identification,
            method="response.backdoor",
            adjustment_set=["x"],
            assumption_count=0,
            derivation_step_count=0,
        ),
        provenance={"operation_id": "op-1"},
        data_snapshot_id=snapshot,
    )


def _contract(criterion: decision.Criterion) -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action("A", inputs=(_q("1"),), utility=decision.x(0)),
            decision.Action("B", inputs=(_q("2"),), utility=decision.x(0) - 1.5),
        ),
        utility_units="util",
        criterion=criterion,
        target_population="target",
    )


def _refusal(call) -> ExternalRefusal:
    with pytest.raises(ExternalRefusal) as caught:
        call()
    assert isinstance(caught.value, CausalUnsupportedError)
    return caught.value


# ----------------------------------------------------------------- program identity


def test_c1_default_spec_identity_is_the_programs_not_a_graph_only_hash():
    spec, program = _spec(), _program()
    assert spec.contract_id == program.identity
    assert len(program.identity) == 64
    assert spec.contract_id != "contract:" + spec.graph_id.removeprefix("graph:")
    assert not spec.contract_id.startswith(("graph:", "contract:"))
    # The program is the same however it is derived.
    assert program_claims.ProgramBinding.from_spec(spec) == program


def test_c1_identity_changes_when_any_field_of_the_program_changes():
    program = _program()
    changes = {
        "graph_id": "graph:other",
        "contract_id": "contract:other",
        "treatment_id": "x",
        "outcome_id": "z",
        "population_id": "elsewhere",
        "intervention_kind": "soft",
        "horizon": 1,
        "dose_grid": (1.0, 3.0),
        "dose_units": "g",
        "outcome_units": "kPa",
        "functional_id": "median",
        "transform_id": "log",
    }
    assert set(changes) == {field.name for field in dataclasses.fields(program)}
    seen = {program.identity}
    for name, value in changes.items():
        other = dataclasses.replace(program, **{name: value}).identity
        assert other not in seen, name
        seen.add(other)
    # And the spec's own identity follows its request, units and contract premises.
    assert _spec().contract_id != _spec(require_evidence=("factor:z",)).contract_id
    assert (
        _spec().contract_id
        != external.response(_ident(), outcome_units="mmHg", dose_units="g").contract_id
    )
    assert (
        _spec().contract_id
        != external.response(_ident([1.0, 3.0]), outcome_units="mmHg", dose_units="mg").contract_id
    )


def test_c1_an_invalid_program_refuses_with_its_own_detail():
    refusal = _refusal(lambda: dataclasses.replace(_program(), treatment_id=" ").identity)
    assert refusal.detail == "program_binding.invalid_binding"
    assert refusal.offending == "treatment_id"
    refusal = _refusal(lambda: dataclasses.replace(_program(), dose_grid=(2.0, 1.0)).identity)
    assert (refusal.detail, refusal.offending) == ("program_binding.invalid_binding", "dose_grid")


# ------------------------------------------------------------ external bound to program


def test_c1_a_faithful_spec_binds_to_its_program_and_stays_external():
    spec, program = _spec(), _program()
    checked = program_claims.check_external_program(spec, program)
    assert (checked.identity, checked.coordinates) == (program.identity, 2)
    bound = program_claims.bind_to_program(spec, program)
    assert bound.identity == program.identity
    claim = bound.bind(_response())
    assert claim.native is False
    assert claim.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert claim.identity["causal_contract_id"] == program.identity
    assert list(claim.values) == [3.0, 5.0]
    assert claim.quantities == (_q("1"), _q("2"))


def test_c1_substituted_treatment_or_outcome_is_refused_at_bind_time():
    spec, program = _spec(), _program()
    for field, wrong in (("treatment_id", "x"), ("outcome_id", "z")):
        other = dataclasses.replace(program, **{field: wrong})
        for call in (
            lambda other=other: program_claims.check_external_program(spec, other),
            lambda other=other: spec.bind(_response(), program=other),
        ):
            refusal = _refusal(call)
            assert refusal.detail == "program_binding.treatment_outcome_substitution"
            assert refusal.reason_code == "external_binding_mismatch"
            assert (refusal.offending, refusal.expected, refusal.supplied) == (
                field,
                wrong,
                "a" if field == "treatment_id" else "y",
            )


def test_c1_target_population_mismatch_is_refused():
    refusal = _refusal(
        lambda: program_claims.bind_to_program(
            _spec(), dataclasses.replace(_program(), population_id="other")
        )
    )
    assert refusal.detail == "program_binding.population_mismatch"
    assert refusal.reason_code == "quantity_semantics_mismatch"
    assert (refusal.offending, refusal.expected, refusal.supplied) == (
        "population_id",
        "other",
        "target",
    )


def test_c1_dose_grid_or_dose_unit_change_is_refused():
    spec, program = _spec(), _program()
    grid = _refusal(
        lambda: program_claims.bind_to_program(
            spec, dataclasses.replace(program, dose_grid=(1.0, 3.0))
        )
    )
    assert grid.detail == "program_binding.dose_grid_changed"
    assert (grid.offending, grid.expected, grid.supplied) == ("doses", "1,3", "1,2")
    units = _refusal(
        lambda: program_claims.bind_to_program(spec, dataclasses.replace(program, dose_units="g"))
    )
    assert units.detail == "program_binding.dose_grid_changed"
    assert (units.offending, units.expected, units.supplied) == ("dose_units", "g", "mg")
    # Dose units the caller never named are not a license to guess them.
    unnamed = external.response(_ident(), outcome_units="mmHg")
    refusal = _refusal(lambda: program_claims.bind_to_program(unnamed, program))
    assert (refusal.detail, refusal.offending) == (
        "program_binding.dose_grid_changed",
        "dose_units",
    )
    assert refusal.supplied == external.UNSPECIFIED_UNITS


def test_c1_an_incompatible_quantities_override_is_refused_each_way():
    program = _program()
    base = _spec().quantities

    def override(index: int, **change):
        return _spec(
            quantities=tuple(
                dataclasses.replace(q, **change) if i == index else q for i, q in enumerate(base)
            )
        )

    units = _refusal(lambda: program_claims.bind_to_program(override(1, units="kPa"), program))
    assert units.detail == "program_binding.quantities_override_mismatch"
    assert units.reason_code == "quantity_semantics_mismatch"
    assert units.offending == "coordinate[1]"
    other_outcome = _refusal(
        lambda: program_claims.bind_to_program(override(0, variable_id="z"), program)
    )
    assert other_outcome.detail == "program_binding.treatment_outcome_substitution"
    assert other_outcome.offending == "coordinate[0]"
    other_population = _refusal(
        lambda: program_claims.bind_to_program(override(0, population_id="other"), program)
    )
    assert other_population.detail == "program_binding.population_mismatch"
    other_dose = _refusal(
        lambda: program_claims.bind_to_program(override(1, regime_id="do(a=9)"), program)
    )
    assert other_dose.detail == "program_binding.dose_grid_changed"
    other_treatment = _refusal(
        lambda: program_claims.bind_to_program(override(1, regime_id="do(x=2)"), program)
    )
    assert other_treatment.detail == "program_binding.treatment_outcome_substitution"
    short = _spec(quantities=base[:1])
    count = _refusal(lambda: program_claims.bind_to_program(short, program))
    assert count.detail == "program_binding.dose_grid_changed"
    assert (count.offending, count.expected, count.supplied) == ("quantities", "2", "1")
    # An override never shares the identity of the request it does not exactly answer.
    assert override(1, units="kPa").contract_id != program.identity
    # A compatible override (equal coordinates) answers the request exactly, so it binds.
    same = _spec(quantities=base)
    assert same.contract_id == program.identity
    assert program_claims.bind_to_program(same, program).bind(_response()).native is False


def test_c1_graph_only_declared_identity_is_refused():
    spec, program = _spec(), _program()
    bare = spec.graph_id.removeprefix("graph:")
    for declared in (f"contract:{bare}", spec.graph_id, program.graph_id, "graph:anything"):
        refusal = _refusal(
            lambda declared=declared: program_claims.bind_to_program(
                dataclasses.replace(spec, contract_id=declared), program
            )
        )
        assert refusal.detail == "program_binding.graph_only_identity", declared
        assert refusal.offending == "declared_identity"
        assert refusal.expected == program.identity


def test_c1_a_changed_graph_or_contract_premise_is_refused():
    spec, program = _spec(), _program()
    other_graph = _refusal(
        lambda: program_claims.bind_to_program(
            dataclasses.replace(spec, graph_id="graph:o"), program
        )
    )
    assert other_graph.detail == "program_binding.contract_identity_mismatch"
    assert other_graph.offending == "graph_id"
    premises = _spec(require_evidence=("factor:z",))
    changed = _refusal(lambda: program_claims.bind_to_program(premises, program))
    assert changed.detail == "program_binding.contract_identity_mismatch"
    assert changed.offending == "contract_id"
    tampered = _refusal(
        lambda: program_claims.bind_to_program(
            dataclasses.replace(spec, contract_id="0" * 64), program
        )
    )
    assert (tampered.detail, tampered.offending) == (
        "program_binding.contract_identity_mismatch",
        "declared_identity",
    )


def test_c1_explicit_contract_id_still_binds_unchanged():
    spec = external.response(
        _ident(),
        outcome_units="mmHg",
        population="target",
        graph_id="graph-1",
        contract_id="checked-contract",
    )
    assert spec.contract_id == "checked-contract"
    claim = spec.bind(_response())
    assert claim.identity["causal_contract_id"] == "checked-contract"
    assert claim.identity["graph_id"] == "graph-1"
    # It names no program, so it cannot be bound to one.
    program = dataclasses.replace(_program(), graph_id="graph-1")
    refusal = _refusal(lambda: program_claims.bind_to_program(spec, program))
    assert refusal.detail == "program_binding.contract_identity_mismatch"
    # A spec with no identified dose-grid request has no program to check against.
    ident = ac.identify(graph=EDGES, names=NAMES, query=ac.AverageEffect("a", "y"))
    ate = external.response(ident, outcome_units="mmHg")
    assert ate.request is None
    assert ate.contract_id.startswith("contract:")
    assert ate.contract_id != "contract:" + ate.graph_id.removeprefix("graph:")
    none = _refusal(lambda: program_claims.bind_to_program(ate, _program()))
    assert none.detail == "program_binding.no_identified_request"


# ----------------------------------------------------------------------- native claims


def test_c1_native_claim_exposes_coordinates_support_and_native_trust():
    program = _program()
    claim = program_claims.native_claim(_view(), program)
    assert claim.coordinates == (_q("1"), _q("2"))
    assert claim.means == (3.0, 5.0)
    assert claim.support == ("supported", "supported")
    assert claim.support_status == "supported"
    assert claim.trust is ProviderTrust.NATIVE_LICENSED
    assert claim.calibration == "point_only"
    assert claim.program_identity == program.identity
    assert claim.provenance_id == "op-1"
    assert claim.has_joint_law is False
    uncertain = _view(uncertainty="posterior")
    assert program_claims.native_claim(uncertain, program).calibration == "unmeasured"


def test_c1_native_claim_refuses_a_response_for_another_request():
    program = _program()
    grid = _refusal(lambda: program_claims.native_claim(_view(grid=(1.0, 3.0)), program))
    assert grid.detail == "program_binding.dose_grid_changed"
    outcome = _refusal(
        lambda: program_claims.native_claim(_view(), dataclasses.replace(program, outcome_id="z"))
    )
    assert (outcome.detail, outcome.offending) == (
        "program_binding.treatment_outcome_substitution",
        "outcome_id",
    )
    # A native response carries no population of its own: the claim's population is taken from
    # the program binding, so a population mismatch is only detectable for external claims
    # (see the external population test above), not here.


def test_c1_native_claim_refuses_unlicensed_or_unlabeled_responses():
    program = _program()
    partial = _refusal(
        lambda: program_claims.native_claim(_view(identification="PartiallyIdentified"), program)
    )
    assert partial.detail == "native_claims.response_not_point_identified"
    assert partial.reason_code == "effect_not_identified"
    unlabeled = _refusal(lambda: program_claims.native_claim(_view(point_status=None), program))
    assert unlabeled.detail == "native_claims.point_support_missing"
    hidden = _refusal(
        lambda: program_claims.native_claim(
            _view(point_status=("supported", "weak_overlap"), support="supported"), program
        )
    )
    assert hidden.detail == "coordinate_support.summary_not_worst_label"
    no_snapshot = _refusal(lambda: program_claims.native_claim(_view(snapshot=None), program))
    assert no_snapshot.detail == "native_claims.invalid_context"
    # A calibration status is never asserted by the caller.
    measured = _refusal(
        lambda: program_claims.native_claim(
            _view(),
            program,
            calibration="measured",  # type: ignore[arg-type]
        )
    )
    assert measured.detail == "native_claims.calibration_not_licensed"
    assert (measured.expected, measured.supplied) == ("unmeasured,point_only", "measured")


def test_c1_native_mean_response_feeds_decision_and_inverse_query_on_hand_values():
    claim = program_claims.native_claim(_view(), _program())
    contract = _contract(decision.Criterion.expected_utility())
    source = claim.as_decision_source(contract)
    assert source.representation == "mean"
    assert source.trust is ProviderTrust.NATIVE_LICENSED
    assert source.withheld == ()
    assert source.coordinates == (_q("1"), _q("2"))
    assert source.source.means == (3.0, 5.0)
    assert source.source.provider_id == "native:op-1"
    assert source.program_identity == claim.program_identity
    result = contract.evaluate(source.source)
    by_id = {action.id: action for action in result.actions}
    assert by_id["A"].expected_utility == pytest.approx(3.0, abs=1e-12)
    assert by_id["B"].expected_utility == pytest.approx(3.5, abs=1e-12)
    assert result.verdict.kind == "uniquely_optimal"
    assert result.selected == ("B",)
    with pytest.raises(CausalUnsupportedError) as export:
        result.export()
    assert export.value.detail == "decision_evaluation.mean_source_not_replayable"

    query = iq.InverseQuery(contract, ("A", "B"), (iq.target_mean(3.2),))
    answered = query.evaluate(source.mean_claim())
    assert answered.feasible_actions == ("B",)
    assert answered.selected == "B"
    # P(utility_B <= 3) = 0.5 needs aligned draws; a mean never answers it.
    for constraint in (
        iq.probability_threshold(3.0, 0.5, tail="lower", direction="at_most"),
        iq.target_quantile(0.5, 3.0),
    ):
        refused = iq.InverseQuery(contract, ("A", "B"), (constraint,))
        with pytest.raises(iq.InverseQueryRefusal) as caught:
            refused.evaluate(source.mean_claim())
        assert caught.value.detail == "decision_evaluation.mean_source_insufficient"


def test_c1_native_mean_only_result_is_refused_for_probability_and_quantile_sources():
    claim = program_claims.native_claim(_view(), _program())
    for criterion in (
        decision.Criterion.quantile(0.5),
        decision.Criterion.threshold_probability(3.0),
    ):
        refusal = _refusal(
            lambda criterion=criterion: claim.as_decision_source(_contract(criterion))
        )
        assert refusal.reason_code == "decision_contract_unsatisfied"
        assert refusal.detail == "native_claims.source_not_supplied"
        assert refusal.expected == "joint_draws,marginal_draws"
        assert refusal.supplied == "mean"
    # A nonlinear utility under an expectation is a distribution question, not a mean one.
    nonlinear = decision.Contract(
        actions=(
            decision.Action("A", inputs=(_q("1"), _q("2")), utility=decision.x(0) * decision.x(1)),
            decision.Action("B", inputs=(_q("1"),), utility=decision.x(0)),
        ),
        utility_units="util",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    assert _refusal(lambda: claim.as_decision_source(nonlinear)).detail == (
        "native_claims.source_not_supplied"
    )


def test_c1_unsupported_coordinates_are_withheld_not_silently_used():
    program = _program()
    outside = "outside_empirical_support"
    partial = program_claims.native_claim(
        _view(point_status=("supported", outside), support=outside), program
    )
    contract = _contract(decision.Criterion.expected_utility())
    source = partial.as_decision_source(contract)
    assert source.coordinates == (_q("1"),)
    assert source.point_status == ("supported",)
    assert [(w.index, w.coordinate, w.status) for w in source.withheld] == [(1, _q("2"), outside)]
    # The action that reads the withheld coordinate cannot be evaluated on this source.
    with pytest.raises(CausalUnsupportedError):
        contract.evaluate(source.source)
    nothing = program_claims.native_claim(
        _view(point_status=("missing_evidence", "missing_evidence"), support="missing_evidence"),
        program,
    )
    assert _refusal(lambda: nothing.as_decision_source(contract)).detail == (
        "native_claims.no_supported_coordinate"
    )
