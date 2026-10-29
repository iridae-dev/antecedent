"""Contracts-first: identities and transformation preview."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.outcomes import require_success
from antecedent.query import AverageEffect


def _data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    z = rng.normal(size=80)
    t = (rng.uniform(size=80) < 1 / (1 + np.exp(-0.4 + 0.9 * z))).astype(float)
    y = 2.0 * t + z + rng.normal(scale=0.4, size=80)
    return {"t": t, "y": y, "z": z}


GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def _prepare(data, *, query=None, graph=GRAPH, **kwargs):
    """One canonical preparation: a point-only frequentist study, nothing executed."""
    return ant.prepare(
        data,
        graph=graph,
        query=AverageEffect("t", "y") if query is None else query,
        refute="none",
        inference=ant.Frequentist(bootstrap=0),
        **kwargs,
    )


def _hex(identity) -> str:
    """The canonical hex of a retained contract identity (a 32-byte digest)."""
    return identity if isinstance(identity, str) else bytes(identity).hex()


_IDENTITIES = (
    "target_id",
    "identification_id",
    "identification_product_id",
    "program_id",
    "inference_binding_id",
    "observation_id",
    "data_snapshot_id",
)


def test_prepared_contract_has_domain_separated_identities() -> None:
    prepared = _prepare(_data())
    report = prepared.inspect()
    assert report.stage == "prepared"
    # The data snapshot identity is not materialized by preparation: inspection
    # reports it unavailable instead of hashing the observations.
    assert report.data_snapshot_id is None
    assert report.details["snapshot_unavailable"] == "prepared_snapshot_identity_not_materialized"
    identities = [getattr(report, name) for name in _IDENTITIES if name != "data_snapshot_id"]
    assert all(isinstance(identity, str) and identity for identity in identities)
    # Every layer is a separate domain: no two identities coincide.
    assert len(set(identities)) == len(identities)
    assert report.claim_id is None
    assert report.details["accepted_version"] == 1
    # Nothing has executed: empirical support is not yet available.
    assert report.support.payload["empirical"]["available"] is False
    assert report.support.payload["empirical"]["reason"] == "not_evaluated"
    outcome = ant.estimate(prepared)
    executed = outcome.inspect()
    contract = executed.contract
    assert contract["format"] == 2
    assert _hex(contract["identities"]["target"]) != _hex(contract["identities"]["program"])
    assert "identification_product" in contract["identities"]
    assert executed.identification.summary == "nonparametrically_identified"
    assert executed.support.payload["matrix_coordinate"].startswith(
        "AverageEffect:Dag:explicit:Frequentist:"
    )
    # The executed contract binds the same prepared question, identification and
    # inference; the snapshot identity is materialized by execution, and the program
    # identity is the one the execution sealed (its checked lowering is bound only once
    # the operation has run), which the executed contract and report agree on.
    for name in _IDENTITIES:
        if name not in {"data_snapshot_id", "program_id"}:
            assert getattr(executed, name) == getattr(report, name), name
    assert _hex(contract["identities"]["program"]) == executed.program_id
    assert _hex(contract["identities"]["target"]) == report.target_id
    assert isinstance(executed.data_snapshot_id, str) and executed.data_snapshot_id
    assert _hex(contract["identities"]["data_snapshot"]) == executed.data_snapshot_id
    assert executed.assumptions.available
    refused = prepared.preview_transform("average_unweighted_class")
    assert refused["refused"] == "true"
    preview = prepared.preview_transform("compatible_data_replace")
    assert preview["intent"] == "compatible_data_replace"
    assert preview["refused"] == "false"


def test_same_shape_rebind_changes_only_data_identity() -> None:
    data = _data()
    prepared = _prepare(data)
    before = prepared.inspect()
    preview = prepared.preview_transform("compatible_data_replace")
    # Preparation does not materialize the snapshot identity; the preview does.
    assert before.data_snapshot_id is None
    assert isinstance(preview["input_data_snapshot"], str) and preview["input_data_snapshot"]
    branch = ant.rebind(prepared, data={**data, "y": data["y"] + data["t"]})
    after = branch.inspect()
    for layer in _IDENTITIES:
        if layer == "data_snapshot_id":
            continue
        assert getattr(before, layer) == getattr(after, layer), layer
    assert after.data_snapshot_id is None
    assert preview["input_program"] == after.program_id
    # Only the data identity differs, observed through the branches' snapshot previews
    # and the executions' materialized snapshot identities.
    branch_preview = branch.preview_transform("compatible_data_replace")
    assert branch_preview["input_data_snapshot"] != preview["input_data_snapshot"]
    assert branch_preview["input_program"] == preview["input_program"]
    first = ant.estimate(prepared).inspect()
    second = ant.estimate(branch).inspect()
    assert first.data_snapshot_id != second.data_snapshot_id
    for layer in _IDENTITIES:
        if layer != "data_snapshot_id":
            assert getattr(first, layer) == getattr(second, layer), layer
    # The source preparation is unchanged by its branch.
    assert prepared.inspect().data_snapshot_id is None
    assert prepared.preview_transform("compatible_data_replace") == preview


def test_contracted_linear_adjustment_replays_from_fit_statistics() -> None:
    from antecedent import artifacts

    data = _data()
    outcome = ant.estimate(_prepare(data))
    require_success(outcome)
    contract = outcome.inspect().contract
    encoded = bytes(outcome.export())
    loaded = artifacts.loads(encoded)
    assert loaded.payload_kind == "analysis_result"
    assert loaded.contract is not None
    accepted = artifacts.accept(encoded)
    assert accepted["accepts_as_verified_program"] == "true"
    assert accepted["program"] == _hex(contract["identities"]["program"])
    assert accepted["target"] == _hex(contract["identities"]["target"])
    query = outcome.inspect().target["query"]
    assert accepted["treatment"] == str(query["treatment_id"])
    assert accepted["outcome"] == str(query["outcome_id"])
    assert accepted["treatment"] == "0"
    assert accepted["outcome"] == "1"
    assert accepted["control"] == "set:0=0"
    assert accepted["active"] == "set:0=1"
    assert accepted["population"] == "all_observed"
    assert accepted["temporal_coordinates"] == "none"
    assert accepted["variable_names"] == "t,y,z"


def test_conditional_effect_contracted_artifact_names_missing_checked_operation() -> None:
    from antecedent import artifacts
    from antecedent.query import ConditionalEffect

    n = 120
    t = np.array([float(i % 2) for i in range(n)])
    w = np.array([float(i % 5) for i in range(n)])
    y = 1.0 + 2.0 * t + 0.5 * t * w
    data = {"t": t, "y": y, "w": w}
    outcome = ant.estimate(
        _prepare(data, graph=[("t", "y"), ("w", "y")], query=ConditionalEffect("t", "y", "w"))
    )
    result = require_success(outcome)
    contract = outcome.inspect().contract
    accepted = artifacts.accept(bytes(outcome.export()))
    assert accepted["accepts_as_verified_program"] == "false"
    assert "dependencies.checked_conditional_effect_operation" in accepted["unresolved"]
    assert accepted["program"] == _hex(contract["identities"]["program"])
    assert accepted["target"] == _hex(contract["identities"]["target"])
    assert accepted["query_kind"] == "conditional_effect"
    query = outcome.inspect().target["query"]
    assert accepted["treatment"] == str(query["treatment_id"])
    assert accepted["outcome"] == str(query["outcome_id"])
    assert accepted["treatment"] == "0"
    assert accepted["outcome"] == "1"
    assert accepted["population"] == "all_observed"
    assert outcome.inspect().support.payload["matrix_coordinate"].startswith("ConditionalEffect:")
    assert abs(result.estimate.ate - 3.0) < 1e-6


def test_mediation_contracted_artifact_names_missing_checked_operation() -> None:
    from antecedent import artifacts
    from antecedent.query import MediationEffect

    a = np.sin(np.arange(200) * 0.71)
    m = 2.0 * a + np.cos(np.arange(200) * 1.13)
    y = 3.0 * a + 4.0 * m + 0.1 * np.sin(np.arange(200) * 0.31)
    data = {"a": a, "m": m, "y": y}
    outcome = ant.estimate(
        _prepare(
            data,
            graph=[("a", "m"), ("a", "y"), ("m", "y")],
            query=MediationEffect("a", "y", mediators=["m"], contrast="direct"),
        )
    )
    result = require_success(outcome)
    contract = outcome.inspect().contract
    accepted = artifacts.accept(bytes(outcome.export()))
    assert accepted["accepts_as_verified_program"] == "false"
    assert "dependencies.checked_mediation_operation" in accepted["unresolved"]
    assert accepted["program"] == _hex(contract["identities"]["program"])
    assert accepted["target"] == _hex(contract["identities"]["target"])
    assert accepted["query_kind"] == "mediation"
    assert accepted["treatment"] == "0"
    assert accepted["outcome"] == "2"
    assert outcome.inspect().support.payload["matrix_coordinate"].startswith("MediationEffect:")
    # The direct effect of a with the mediator held at its natural value is the structural
    # coefficient 3 (y = 3a + 4m + small noise); the indirect path 4 * 2 is not in it.
    assert abs(result.estimate.ate - 3.0) < 0.01, result.estimate.ate


def test_response_curve_contracted_artifact_names_missing_checked_grid() -> None:
    from antecedent import artifacts
    from antecedent.query import ResponseCurve

    n = 240
    z = np.sin(np.arange(n) / 17.0)
    treatment = z + np.cos(np.arange(n) / 11.0) + (np.arange(n) % 7) * 0.03
    outcome = 1.0 + 2.0 * treatment + 0.8 * z
    data = {"treatment": treatment, "outcome": outcome, "confounder": z}
    outcome = ant.estimate(
        _prepare(
            data,
            graph=[
                ("confounder", "treatment"),
                ("confounder", "outcome"),
                ("treatment", "outcome"),
            ],
            query=ResponseCurve("treatment", "outcome", grid=[-0.5, 0.0, 0.5]),
        )
    )
    result = require_success(outcome)
    # Deterministic linear outcome 1 + 2 treatment + 0.8 z: the curve at g is 1 + 2 g + 0.8 mean(z).
    grid = np.array([-0.5, 0.0, 0.5])
    np.testing.assert_allclose(
        np.asarray(result.response.values).ravel(), 1.0 + 2.0 * grid + 0.8 * z.mean(), atol=0.05
    )
    contract = outcome.inspect().contract
    accepted = artifacts.accept(bytes(outcome.export()))
    assert accepted["accepts_as_verified_program"] == "false"
    assert "dependencies.checked_response_grid_operation" in accepted["unresolved"]
    assert accepted["program"] == _hex(contract["identities"]["program"])
    assert accepted["query_kind"] == "response"
    assert accepted["temporal_coordinates"] == "none"
    assert outcome.inspect().support.payload["matrix_coordinate"].startswith("ResponseCurve:")


def test_intervention_response_contracted_artifact_names_missing_checked_operation() -> None:
    from antecedent import artifacts
    from antecedent.intervention import Set
    from antecedent.query import InterventionResponse

    n = 240
    z = np.sin(np.arange(n) / 17.0)
    treatment = z + np.cos(np.arange(n) / 11.0)
    outcome = 1.0 + 2.0 * treatment + 0.8 * z
    data = {"treatment": treatment, "outcome": outcome, "confounder": z}
    outcome = ant.estimate(
        _prepare(
            data,
            graph=[
                ("confounder", "treatment"),
                ("confounder", "outcome"),
                ("treatment", "outcome"),
            ],
            query=InterventionResponse("outcome", intervention=Set("treatment", 0.25)),
        )
    )
    result = require_success(outcome)
    contract = outcome.inspect().contract
    accepted = artifacts.accept(bytes(outcome.export()))
    assert accepted["accepts_as_verified_program"] == "false"
    assert "dependencies.checked_intervention_response_operation" in accepted["unresolved"]
    assert accepted["program"] == _hex(contract["identities"]["program"])
    assert accepted["query_kind"] == "response"
    assert accepted["temporal_coordinates"] == "none"
    assert (
        outcome.inspect().support.payload["matrix_coordinate"].startswith("InterventionResponse:")
    )
    # Outcome 1 + 2 treatment + 0.8 z under do(treatment = 0.25): 1.5 + 0.8 mean(z), exactly the
    # structural value up to the estimator's smoothing.
    expected = 1.5 + 0.8 * z.mean()
    assert abs(float(np.asarray(result.response.values).ravel()[0]) - expected) < 0.01


def test_four_slots_agree_between_inspect_and_prepared_contract() -> None:
    prepared = _prepare(_data())
    inspected = prepared.inspect()
    assert inspected.stage == "prepared"
    assert inspected.contract is None
    assert inspected.claim_id is None
    assert inspected.identification.available is True
    assert inspected.rendering_limitation() is None
    assert prepared.preflight().identification.available is False
    report = inspected.to_dict()
    assert report["calibration"]["status"] == "unavailable"
    assert report["calibration"]["reason"] == "not_executed"
    outcome = ant.estimate(prepared)
    result = require_success(outcome)
    contract = outcome.inspect().contract
    assert contract is not None
    # The prepared question, identification and inference are the executed contract's;
    # the executed program identity is the one the execution sealed.
    assert inspected.target_id == _hex(contract["identities"]["target"])
    assert inspected.identification_id == _hex(contract["identities"]["identification"])
    assert inspected.inference_binding_id == _hex(contract["identities"]["inference_binding"])
    # Preparation does not materialize the snapshot identity; execution does.
    assert inspected.data_snapshot_id is None
    assert outcome.inspect().data_snapshot_id == _hex(contract["identities"]["data_snapshot"])
    assert "claim_id" not in contract["identities"]
    assert result.reasoning is not None
    assert result.program_id == _hex(contract["identities"]["program"])
    assert result.claim_id is not None
    assert result.claim_id != result.program_id
    assert result.answer.kind == "point" and result.answer.value == result.effect
    # One preview API, one vocabulary: the Rust transformation intents.
    display = prepared.preview_transform("display_precision")
    assert display["intent"] == "display_precision"
    assert display["refused"] == "false"
    scientific = prepared.preview_transform("new_conditional_query")
    assert scientific["intent"] == "new_conditional_query"
    assert scientific["input_program"] == inspected.program_id


def test_four_slots_refuse_identified_atom_mean_for_partial_claim() -> None:
    from antecedent.results import (
        AnalysisResult,
        EstimateView,
        IdentificationView,
        PerformanceView,
        PosteriorView,
        ValidationView,
    )

    result = AnalysisResult(
        identification=IdentificationView(
            status="GraphDependent",
            method="mixture",
            adjustment_set=[],
            assumption_count=0,
            derivation_step_count=0,
        ),
        estimate=EstimateView(
            ate=2.29,
            se_analytic=0.03,
            se_bootstrap=None,
            estimator_id="ols",
            method="mixture",
        ),
        posterior=PosteriorView(
            effect_mean=2.29,
            effect_sd=0.03,
            q025=2.2,
            q975=2.4,
            n_draws=200,
            p_below_zero=0.0,
            backend="conjugate",
            unidentified_mass=0.2,
        ),
        validation=ValidationView(passed=True, ran=False, count=0),
        performance=PerformanceView(),
        diagnostics=[],
        provenance={},
        structural_unidentified_mass=0.2,
    )
    assert result.rendering_limitation() == "unidentified_mass"
    assert result.answer.kind == "partial"
    assert result.answer.value is None
    assert result.answer.detail == "unidentified_mass"
    html = result._repr_html_()
    assert "no point mean" in html and "(unidentified_mass)" in html
    assert "2.290" not in html
    assert "2.290" not in repr(result.estimate) and "2.290" not in repr(result.posterior)


def test_calibration_reads_from_contract() -> None:
    result = require_success(ant.estimate(_prepare(_data())))
    slot = result.calibration
    assert slot.status in {"calibrated", "scope_not_assessed", "unavailable"}
    if slot.status == "unavailable":
        assert slot.reason


def test_calibration_prepared_is_not_executed() -> None:
    calibration = _prepare(_data()).inspect().calibration
    assert calibration.status == "unavailable"
    assert calibration.reason == "not_executed"


def test_custom_validator_is_attested_and_covered_by_claim_id() -> None:
    def always_pass(*, ate, **_kwargs):
        return {"passed": True, "refuted_ate": ate, "comparison": 0.0}

    def always_fail(*, ate, **_kwargs):
        return {"passed": False, "refuted_ate": ate, "comparison": 1.0}

    def run(validators):
        return require_success(ant.estimate(_prepare(_data(), validators=validators)))

    passing, failing, unchecked = run([always_pass]), run([always_fail]), run(None)
    attested = require_success(ant.load(bytes(passing.export()))).decoded_artifact.contract[
        "claim"
    ]["attested"]
    assert attested, "a custom validator result must be attested in the claim"
    assert (
        require_success(ant.load(bytes(failing.export()))).decoded_artifact.contract["claim"][
            "attested"
        ]
        != attested
    )
    assert len({passing.claim_id, failing.claim_id, unchecked.claim_id}) == 3


def _aipw_study(data):
    return _prepare(data, estimator="aipw")


def test_retarget_export_reload_and_reexecute_on_snapshot() -> None:
    from antecedent import artifacts

    data = _data()
    estimated = ant.estimate(_aipw_study(data))
    require_success(estimated)
    weights = np.exp(data["z"] / 3)
    retargeted = estimated.study.retarget(weights, depends_on=["z"])
    require_success(retargeted)
    encoded = bytes(retargeted.export())
    loaded = artifacts.loads(encoded)
    assert loaded.payload_kind == "analysis_result"
    acceptance = artifacts.accept(encoded)
    assert acceptance["accepts_as_verified_program"] == "false"
    assert "program.checked_aipw_lowering" in acceptance["unresolved"]
    section = loaded.contract
    carried = section["target_weights"]
    assert np.array_equal(np.asarray(carried["values"], dtype=float), weights)
    assert (
        bytes(section["identities"]["target_weights"]).hex()
        == retargeted.inspect().target_weights_id
    )
    assert carried["identity"]["data_snapshot"] == section["identities"]["data_snapshot"]


def test_constant_retarget_keeps_the_prepared_target() -> None:
    data = _data()
    estimated = ant.estimate(_aipw_study(data))
    constant = estimated.study.retarget(np.full(len(data["t"]), 7.0), depends_on=[])
    require_success(constant)
    assert constant.inspect().target_weights_id is None
    assert constant.inspect().target_id == estimated.inspect().target_id
    from antecedent import artifacts

    assert artifacts.accept(bytes(constant.export()))["accepts_as_verified_program"] == "true"


def test_retarget_uses_the_scores_of_the_execution_it_follows() -> None:
    from antecedent import artifacts

    def draw(seed: int, shift: float) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        z = rng.normal(size=300)
        t = (rng.uniform(size=300) < 1 / (1 + np.exp(-0.8 * z))).astype(float)
        y = (2.0 + shift * z) * t + z + rng.normal(scale=0.5, size=300)
        return {"t": t, "y": y, "z": z}

    d1, d2 = draw(1, 0.0), draw(2, 3.0)
    weights = np.exp(d2["z"])
    # Preparing on d1 and branching onto d2 retargets the scores of the d2 execution.
    after_estimate = ant.estimate(ant.rebind(_aipw_study(d1), data=d2)).study.retarget(
        weights, depends_on=["z"]
    )
    on_d2 = ant.estimate(_aipw_study(d2)).study.retarget(weights, depends_on=["z"])
    on_d1 = ant.estimate(_aipw_study(d1)).study.retarget(weights, depends_on=["z"])
    assert abs(require_success(on_d2).ate - require_success(on_d1).ate) > 0.5, (
        "fixture must separate the two snapshots"
    )
    assert require_success(after_estimate).ate == require_success(on_d2).ate
    exported = artifacts.loads(bytes(after_estimate.export())).contract
    reference = artifacts.loads(bytes(on_d2.export())).contract
    assert exported["identities"]["data_snapshot"] == reference["identities"]["data_snapshot"]
    assert exported["identities"]["target_weights"] == reference["identities"]["target_weights"]


def test_retarget_refuses_new_data_with_code() -> None:
    """Row weights re-execute on their own snapshot, and refuse another one.

    The binding is a property of the retargeted result, not of the plan: the
    plan is still an AllObserved study and re-estimates on new data.
    """
    data = _data()
    estimated = ant.estimate(_aipw_study(data))
    require_success(estimated)
    weights = np.exp(data["z"] / 3)
    retargeted = estimated.study.retarget(weights, depends_on=["z"])
    carried = bytes(retargeted.export())

    # The plan itself is not bound: a plain estimate on a rebound branch still works.
    moved = {**data, "y": data["y"] + 1.0}
    assert (
        require_success(ant.estimate(ant.rebind(estimated.study, data=moved))).ate
        == require_success(ant.estimate(ant.rebind(estimated.study, data=moved))).ate
    )

    # Re-executing the carried weights on their own snapshot reproduces them.
    same = ant.estimate(_aipw_study(data))
    again = same.study.reexecute_retarget(carried)
    assert require_success(again).ate == require_success(retargeted).ate
    assert again.inspect().target_weights_id == retargeted.inspect().target_weights_id

    # On a different snapshot of the same shape it refuses with the code: a valid scientific
    # question that cannot be answered returns a typed refusal outcome.
    other = ant.estimate(_aipw_study(moved))
    refused = other.study.reexecute_retarget(carried)
    assert not refused.accepted
    assert refused.refusal is not None
    assert refused.refusal.code == "row_weights_bound_to_snapshot"
    with pytest.raises(CausalUnsupportedError) as raised:
        require_success(refused)
    assert getattr(raised.value, "reason_code", None) == "row_weights_bound_to_snapshot"
