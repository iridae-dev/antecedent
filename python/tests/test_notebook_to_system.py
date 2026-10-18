"""Notebook-to-system contracts for retained Python interfaces."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import antecedent as ant
import numpy as np
import pytest
from antecedent import adaptive_inference as ai
from antecedent import joint_law_propagation as jlp
from antecedent import mechanism_composition as mc
from antecedent import predictive_assessment as pa
from antecedent import provider_callbacks as pc
from antecedent import scientific_bridge as sb
from antecedent.artifacts import ArtifactExpectation
from antecedent.results import MeanSource


def _study():
    rng = np.random.default_rng(314)
    z = rng.normal(size=160)
    t = rng.binomial(1, 0.5, len(z)).astype(float)
    data = {"z": z, "t": t, "y": 2 * t + z + rng.normal(size=len(z))}
    study = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        bootstrap=0,
        refute=False,
    )
    return data, study


def test_prepared_identification_and_inspection_are_retained():
    _, study = _study()
    assert study.last_result is None
    assert study.inspect().stage == "prepared"
    identification = study.identification
    assert identification.verdict == "identified"
    assert identification.adjustment_set == ("z",)
    assert identification.derivation_statements
    assert identification.assumption_statements
    assert study.identification is identification
    result = study.estimate()
    assert study.last_result is result
    assert study.inspect().stage == "executed"
    assert study.inspect().answer.value == result.answer.value
    assert study.inspect_plan().answer.detail == "not_executed"
    assert not hasattr(identification, "validate")
    assert "prepared" in study.inspect_plan()._repr_html_()


def test_original_result_refutation_survives_refresh_and_failed_clicks():
    data, study = _study()
    result = study.estimate()
    updated = study.refresh({**data, "y": data["y"] + data["t"]})
    checked = result.refute(suite="cheap")
    assert checked.answer.value == result.answer.value
    assert checked.estimate.estimator_id == result.estimate.estimator_id
    assert updated.answer.value == pytest.approx(result.answer.value + 1)
    with pytest.raises(ant.CausalError):
        study.refresh({"missing": np.ones(160)})
    assert study.last_result is updated
    assert study.reprepare(data).estimate().answer.value == result.answer.value


def test_loaded_verified_values_and_acceptance_are_immutable():
    _, study = _study()
    original = study.estimate()
    artifact = original.artifact()
    loaded = ant.load(artifact, expected=artifact.expectation)
    assert loaded.acceptance.verified and loaded.acceptance.identity_bound
    with pytest.raises(TypeError, match="does not support item assignment"):
        loaded.decoded_artifact.contract["claim"]["value_bits"] = 0
    with pytest.raises(TypeError, match="does not support item assignment"):
        loaded.acceptance.details["recognized"] = "false"
    copy = loaded.decoded_artifact.to_dict()
    copy["contract"]["claim"]["value_bits"] = 0
    assert loaded.answer.value == original.answer.value
    assert loaded.export() == original.export()
    with pytest.raises(ant.errors.CausalSerializationError, match="expected claim_id"):
        ant.load(artifact, expected={"claim_id": "0" * 64})


def test_scientific_arrays_are_detached_and_cannot_be_made_writable():
    mean = np.array([1.0, 2.0])
    covariance = np.eye(2)
    law = mc.JointGaussianLaw(("x", "y"), mean, covariance)
    mean[0] = 99
    covariance[0, 0] = 99
    assert law.mean[0] == 1 and law.covariance[0, 0] == 1
    with pytest.raises(ValueError, match="WRITEABLE"):
        law.mean.setflags(write=True)
    editable = law.to_dict()
    editable["mean"][0] = 42
    assert law.mean[0] == 1


def test_callback_ownership_and_tampered_handles_are_refused():
    from test_provider_callbacks import exported, gaussian

    owner, declaration, execution, artifact = exported()
    other = pc.CallbackSession()
    other.execute(gaussian(), declaration, pc.Request(pc.Query.mean()))
    with pytest.raises(pc.CallbackRefusal) as caught:
        other.export(execution, "wrong-session")
    assert caught.value.detail == "provider_callbacks.foreign_execution"
    with pytest.raises(pc.CallbackRefusal) as caught:
        owner.export(replace(execution, output_digest="0" * 64), "tampered")
    assert caught.value.detail == "provider_callbacks.execution_identity_mismatch"
    assert isinstance(execution.export(), bytes)
    loaded = ant.load(artifact, expected=artifact.expectation)
    assert not loaded.acceptance.replayable and loaded.acceptance.resumable
    assert loaded.export() == artifact.data
    resumed = other.resume(execution.export(), provider=gaussian())
    assert resumed.kind == "answered"
    assert other.export(resumed.execution, "resumed")


@pytest.mark.parametrize("value", [0.9, True, np.bool_(False), np.nan, np.inf, 2**64])
def test_integer_fields_refuse_lossy_values(value):
    with pytest.raises(ant.errors.CausalValidateError):
        ai.BayesianPosterior(value)._wire()
    with pytest.raises(ant.errors.CausalValidateError):
        pa.ValidationDesign.rolling_origin([0, 1], value)


def test_numpy_and_integral_numeric_inputs_are_lossless():
    assert ai.BayesianPosterior(np.int64(1))._wire()["look"] == 1
    assert ai.BayesianPosterior(1.0)._wire()["look"] == 1
    assert pa.ValidationDesign.blocked(np.arange(3)).wire["blocks"] == [0, 1, 2]
    assert pa.Dependence.serial(np.arange(3), 1.0).wire["truncation_lag"] == 1
    assert pa.keys("rows", np.arange(3)) == pa.keys("rows", [0, 1, 2])
    query = ant.ResponseCurve("t", "y", grid=[0, 1], horizons=np.arange(1, 3))
    assert tuple(query.horizons) == (1, 2)


def test_adaptive_results_have_named_method_specific_records():
    from test_adaptive_inference import group_sequential_design, thompson_design

    session = ai.AdaptiveInferenceSession.prepare(thompson_design())
    result = session.analyze(ai.BayesianPosterior(1))
    assert isinstance(result.payload, ai.PosteriorSummary)
    assert 0 <= result.payload.arms[0].mean <= 1
    assert result.payload.arms[0].alpha >= 1
    assert not hasattr(result, "values")
    assert result.to_rows()[0]["claim"] == "point_only"
    sequential = ai.AdaptiveInferenceSession.prepare(group_sequential_design()).analyze(
        ai.GroupSequential(2)
    )
    assert isinstance(sequential.payload, ai.SequentialSummary)
    assert sequential.payload.stopped_at is None or sequential.payload.stopped_at >= 0


def test_bridge_is_a_complete_checked_mean_producer():
    from antecedent import decision, inverse_query

    from test_scientific_bridge import surrogate_contract, surrogate_source

    claim = sb.prepare_scientific_bridge(surrogate_contract()).derive(surrogate_source())
    source = claim.as_mean_source()
    assert isinstance(source, MeanSource)
    assert decision.MeanSource is inverse_query.MeanClaim is MeanSource
    assert source.means == (claim.value,)
    assert source.coordinates == (claim.quantity,)
    assert source.provider_id == claim.claim_id
    contract = decision.Contract(
        actions=[
            decision.Action("use", inputs=(claim.quantity,), utility=decision.x(0)),
            decision.Action("wait", inputs=(claim.quantity,), utility=-decision.x(0)),
        ],
        utility_units=claim.quantity.units,
        target_population=claim.quantity.population_id,
        criterion=decision.Criterion.expected_utility(),
    )
    result = contract.evaluate(claim)
    assert result.source.means == source.means
    with pytest.raises(ant.errors.CausalValueError, match="differs"):
        claim.as_mean_source(quantity=replace(claim.quantity, units="another_unit"))


def test_specialized_artifact_dispatch_and_fresh_process_replay(tmp_path):
    from test_scientific_bridge import surrogate_contract, surrogate_source

    claim = sb.prepare_scientific_bridge(surrogate_contract()).derive(surrogate_source())
    envelope = claim.artifact()
    assert claim.export() == envelope.data
    assert ant.artifacts.inspect(envelope).acceptance.status == "unavailable"
    loaded = ant.load(envelope, expected=envelope.expectation)
    assert isinstance(loaded, sb.ConsumedBridge)
    assert loaded.claim.value == claim.value
    assert loaded.acceptance.identity_bound and loaded.acceptance.replayable
    path = tmp_path / "bridge.bin"
    path.write_bytes(envelope.data)
    script = """import json,sys; import antecedent as a
from pathlib import Path
r=a.load(Path(sys.argv[1]).read_bytes(), expected=sys.argv[2])
print(json.dumps({'value':r.claim.value,'bound':r.acceptance.identity_bound}))
"""
    output = subprocess.run(
        [sys.executable, "-c", script, str(path), envelope.expectation.identity],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.stdout) == {"value": claim.value, "bound": True}
    with pytest.raises(ant.CausalError):
        ant.load(envelope.data, expected="0" * 64)
    with pytest.raises(ant.errors.CausalValueError, match="kind"):
        ant.load(envelope, expected=ArtifactExpectation("wrong", envelope.expectation.identity))


def test_propagation_reading_remains_mean_only_and_immutable():
    claim = jlp.mixture([jlp.MixtureComponent("first", "evidence", 1.0, 2.0, 3.0)])
    with pytest.raises(TypeError, match="does not support item assignment"):
        claim.values["mean"] = 99
    quantity = ant.joint_distribution.ScientificQuantity.of(
        "outcome", "y", units="units", population="target", regime="declared_model"
    )
    source = claim.as_mean_source(quantity=quantity)
    assert source.means == (claim.value("mean"),)
    with pytest.raises(jlp.JointLawPropagationRefusal):
        claim.interval()
    loaded = ant.load(claim.artifact(), expected=claim.identity)
    assert loaded.values == claim.values
    assert loaded.export() == claim.export()


@pytest.mark.parametrize("operation", ["estimate", "refresh", "study_refute", "result_refute"])
@pytest.mark.parametrize("change", ["assignment", "ids", "row_order"])
def test_every_design_rebinding_path_is_guarded_and_atomic(operation, change):
    from test_column_name_inputs import _bernoulli, _randomized_named_query

    assignment, outcome, ids = _bernoulli(n=80)
    data = {"y": outcome, "arm": assignment, "id": ids}
    study = ant.prepare(data, query=_randomized_named_query(), refute=False)
    original = study.estimate()
    before = original.export()
    changed = dict(data)
    if change == "assignment":
        changed["arm"] = ~assignment
        column = "arm"
    elif change == "ids":
        changed["id"] = np.array([f"new-{key}" for key in ids])
        column = "id"
    else:
        changed = {key: value[::-1] for key, value in data.items()}
        column = "arm|id"
    call = {
        "estimate": lambda: study.estimate(changed),
        "refresh": lambda: study.refresh(changed),
        "study_refute": lambda: study.refute(changed, suite="cheap"),
        "result_refute": lambda: original.refute(changed, suite="cheap"),
    }[operation]
    with pytest.raises(ant.CausalError, match=f"({column}).*reprepare"):
        call()
    assert study.last_result is original
    assert original.export() == before
    assert study.estimate().answer.value == original.answer.value


def test_named_design_normalization_and_input_detachment():
    from test_column_name_inputs import _bernoulli, _randomized_named_query

    assignment, outcome, ids = _bernoulli(n=80)
    study = ant.prepare(
        {"y": outcome, "arm": assignment, "id": ids}, query=_randomized_named_query(), refute=False
    )
    original = study.estimate()
    assert (
        study.estimate({"y": outcome, "arm": assignment.astype(float), "id": ids}).answer.value
        == original.answer.value
    )
    assignment[:] = ~assignment
    with pytest.raises(ant.CausalError, match="arm.*reprepare"):
        study.refresh({"y": outcome, "arm": assignment})
    assert study.refresh({"y": outcome}).answer.value == original.answer.value
    revised = study.reprepare({"y": outcome, "arm": assignment, "id": ids})
    assert revised.estimate().answer.value == pytest.approx(-original.answer.value)


def test_exact_numeric_integer_normalization_without_float_rounding():
    from decimal import Decimal
    from fractions import Fraction

    from antecedent._validate import count

    assert count(Fraction(6, 2), "count") == 3
    assert count(Decimal("3.00"), "count") == 3
    assert count(np.longdouble("1.0"), "count") == 1
    for value in (Fraction(3, 2), Decimal("3.01"), Decimal("NaN"), Decimal("Infinity")):
        with pytest.raises(ant.errors.CausalTypeError, match="count"):
            count(value, "count")
    assert count(2**64 - 1, "count", maximum=2**64 - 1) == 2**64 - 1
    with pytest.raises(ant.errors.CausalValueError, match="count"):
        count(2**64, "count", maximum=2**64 - 1)


def test_design_integer_columns_never_overflow_numpy_signed_casts():
    from antecedent._columns import _cast_int

    assert _cast_int(np.array([float(2**63)]), "fold_ids", "fold") == (2**63,)
    assert _cast_int(np.array([np.uint64(2**64 - 1)]), "fold_ids", "fold") == (2**64 - 1,)
    with pytest.raises(ant.errors.CausalValueError, match=r"fold_ids='fold'\[1\]"):
        _cast_int(np.array([0.0, float(2**64)]), "fold_ids", "fold")
