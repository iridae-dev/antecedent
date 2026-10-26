"""Checked mean handoffs and shared dispatch across specialized scientific producers."""

from dataclasses import replace

import antecedent as ant
import pytest
from antecedent import decision
from antecedent import evidence_ledger as el
from antecedent import mechanism_composition as mc
from antecedent import observation_query as oq
from antecedent.artifacts import ExportedArtifact
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import Outcome, require_success
from antecedent.results import MeanSource


def quantity(variable="Y", units="u", population="pop", regime="declared_model", functional="mean"):
    return ScientificQuantity.of(
        "outcome",
        variable,
        units=units,
        population=population,
        regime=regime,
        functional=functional,
    )


def test_measurement_adapter_retains_data_binding_and_refuses_scope_changes():
    from test_measurement_models import prepare_classical, simulate

    estimate = require_success(prepare_classical(0.6, 0.0, simulate(150, 34)).estimate())
    q = quantity("outcome", "unit", "cohort", "observational")
    source = estimate.as_mean_source(quantity=q)
    assert source.means == (estimate.point,)
    assert source.provider_id == estimate.model_identity
    assert source.snapshot_id == estimate.data_identity
    assert source.trust != "native"
    assert source.lineage[0]["identity"] == estimate.model_identity
    with pytest.raises(ant.errors.CausalValueError, match="population"):
        estimate.as_mean_source(quantity=replace(q, population_id="other"))
    envelope = estimate.export()
    assert envelope.expectation.identity == estimate.model_identity
    with pytest.raises(ant.errors.CausalValueError, match="requires expected"):
        require_success(ant.load(envelope))
    consumed = require_success(ant.load(envelope, expected=envelope.expectation))
    assert consumed.point == estimate.point
    assert bytes(consumed.export()) == bytes(estimate.export())
    assert consumed.as_mean_source(quantity=q).snapshot_id == source.snapshot_id
    with pytest.raises(ant.CausalError):
        require_success(ant.load(envelope, expected="0" * 64))


def test_composed_mean_and_consumed_mean_handoffs_require_declarations():
    from test_mechanism_composition import chain, compose

    model = compose(chain())
    q = quantity()
    contrast = mc.MeanContrast.ate("Y", "X", 2.0, 1.0)
    with pytest.raises(ant.errors.CausalValueError, match="explicit scientific quantity"):
        model.as_mean_source(contrast=contrast)
    with pytest.raises(ant.errors.CausalValueError, match="contrast="):
        model.as_mean_source(quantity=q)
    source = model.as_mean_source(quantity=q, contrast=contrast)
    assert source.means == pytest.approx((6.5,))
    artifact = model.export({"X": 2.0}, outcome="Y")
    consumed = require_success(ant.load(artifact, expected=artifact.expectation))
    assert consumed.result.mean == pytest.approx(13.5)
    assert consumed.as_mean_source(quantity=q).means == pytest.approx((13.5,))
    assert bytes(consumed.export()) == artifact.data


def test_ledger_mean_keeps_evidence_and_remains_point_only():
    from test_evidence_ledger import data, summary

    ledger = el.EvidenceLedger()
    first = ledger.add_source(data("first", 0, 10))
    second = ledger.add_source(data("second", 10, 20))
    outcome = require_success(
        ledger.pool_summaries("pooled", [summary(first, 1.0, 1.0), summary(second, 3.0, 4.0)])
    )
    source = outcome.as_mean_source(quantity=quantity())
    assert source.means == pytest.approx((1.4,))
    assert set(source.evidence) == {first, second}
    assert source.provider_id == outcome.source_identity
    assert source.lineage[0]["evidence"] == source.evidence
    artifact = ledger.export(outcome.result)
    consumed = require_success(ant.load(artifact, expected=artifact.expectation))
    assert bytes(consumed.export()) == artifact.data
    assert consumed.acceptance.identity_bound
    with pytest.raises(el.EvidenceLedgerRefusal):
        ledger.pooled_interval(outcome.result)


def test_transported_counterfactual_identity_and_point_only_reading_survive_dispatch():
    from test_transported_counterfactual import effect

    claim = effect()
    source = claim.as_mean_source(quantity=quantity("y"))
    assert source.means == (claim.target_contrast,)
    assert source.lineage
    artifact = claim.export()
    consumed = require_success(ant.load(artifact, expected=artifact.expectation))
    assert consumed.identity == claim.identity
    assert consumed.inference_claim == "point_only"
    assert consumed.target_contrast == claim.target_contrast
    assert bytes(consumed.export()) == artifact.data


def test_observation_probability_preserves_its_functional_semantics():
    from test_observation_query import confounded

    claim = oq.threshold_crossing(confounded(), "Y", "1", "S", threshold=0.5)
    q = quantity(units="probability", functional="probability")
    with pytest.raises(ant.errors.CausalValueError, match="field="):
        claim.as_mean_source(quantity=q)
    source = claim.as_mean_source(quantity=q, field="crossing_probability")
    assert source.means == pytest.approx((0.5,))
    with pytest.raises(ant.errors.CausalValueError, match="probability functional"):
        claim.as_mean_source(
            quantity=replace(q, functional_id="mean"), field="crossing_probability"
        )
    artifact = claim.export()
    consumed = require_success(ant.load(artifact, expected=artifact.expectation))
    assert consumed.claim.crossing_probability == claim.crossing_probability
    assert consumed.as_mean_source(quantity=q, field="crossing_probability").means == source.means
    assert bytes(consumed.export()) == artifact.data


def test_mean_adapter_does_not_license_nonlinear_expected_utilities():
    from test_mechanism_composition import chain, compose

    q = quantity()
    source = compose(chain()).as_mean_source(quantity=q, contrast=mc.MeanContrast.mean_at("Y", {}))
    assert isinstance(source, MeanSource)
    contract = decision.Contract(
        actions=[
            decision.Action("use", inputs=(q,), utility=decision.x(0) * decision.x(0)),
            decision.Action("wait", inputs=(q,), utility=-decision.x(0)),
        ],
        utility_units="u",
        target_population="pop",
        criterion=decision.Criterion.expected_utility(),
    )
    with pytest.raises(decision.DecisionRefusal) as refused:
        require_success(contract.evaluate(source))
    assert refused.value.reason_code == "decision_contract_unsatisfied"


def test_compact_and_cohort_artifacts_have_independent_frozen_metadata():
    from antecedent import cohort_decision as cd

    from test_cohort_decision import cohort, compact_model

    model = compact_model()
    envelope = model.export()
    consumed = require_success(ant.load(envelope, expected=envelope.expectation))
    assert consumed.identity == model.identity
    assert bytes(consumed.export()) == envelope.data
    with pytest.raises(TypeError, match="does not support item assignment"):
        consumed.body["scope"] = "relabelled"
    editable = consumed.to_dict()
    editable["body"]["coefficients"][0] = 999
    assert consumed.to_dict()["body"]["coefficients"][0] != 999
    source = cd.decision_source_from_compact_export(model, cohort=cohort(), outcome_units="mmHg")
    artifact = source.export()
    loaded = require_success(ant.load(artifact, expected=artifact.expectation))
    assert loaded.mean_contrast == source.mean_contrast
    assert not loaded.is_native
    assert bytes(loaded.export()) == artifact.data
    with pytest.raises(TypeError, match="does not support item assignment"):
        loaded.summary["mean_contrast"] = 999


def test_constancy_dispatch_preserves_uncalibrated_boundary():
    from antecedent import temporal

    claim = temporal.effect_constancy(
        [("a", 1.0, 0.5), ("b", 2.0, 0.5)],
        estimand=temporal.EffectEstimand("ate_difference", "u", "treated-control", "h2"),
    )
    envelope = claim.export()
    loaded = require_success(ant.load(envelope, expected=envelope.expectation))
    assert loaded.statistic == claim.statistic
    assert loaded.calibration == "unmeasured"
    assert bytes(loaded.export()) == envelope.data


def _cbor_length(major, length):
    if length < 24:
        return bytes([(major << 5) | length])
    if length <= 255:
        return bytes([(major << 5) | 24, length])
    if length <= 65535:
        return bytes([(major << 5) | 25]) + length.to_bytes(2, "big")
    return bytes([(major << 5) | 26]) + length.to_bytes(4, "big")


def test_unknown_framed_format_is_checked_and_forwardable_without_consumption():
    from test_cohort_decision import compact_model

    inner = bytes(compact_model().export())
    # Rust's named artifact framing is a CBOR pair (schema names, inner bytes).
    frame = b"ANTECEDENT-FUTURE\x01\x82\x80" + _cbor_length(4, len(inner))
    frame += b"".join(_cbor_length(0, byte) for byte in inner)
    reading = ant.artifacts.inspect(frame)
    assert reading.kind == "ANTECEDENT-FUTURE"
    assert not reading.acceptance.replayable
    assert bytes(reading.export()) == frame
    loaded = require_success(ant.load(frame))
    assert isinstance(loaded, ant.artifacts.ArtifactInspection)
    assert bytes(loaded.export()) == frame
    assert loaded.acceptance.status == "unavailable"
    with pytest.raises(ant.errors.CausalSerializationError):
        ant.artifacts.inspect(frame[:-1])
    with pytest.raises(ant.errors.CausalSerializationError):
        ant.artifacts.inspect(frame + b"trailing")


@pytest.mark.parametrize(
    "data", [b"ANTECEDENT-garbage", b"ANTECEDENT-FUTURE\x01", b"ANTECEDENT-FUTURE\x01garbage"]
)
def test_malformed_frames_are_errors_even_without_scientific_consumption(data):
    with pytest.raises(ant.errors.CausalSerializationError):
        ant.artifacts.inspect(data)
    with pytest.raises(ant.errors.CausalSerializationError):
        require_success(ant.load(data))


def test_temporal_counterfactual_shared_dispatch_keeps_shared_abduction():
    from test_temporal_counterfactual import run

    original = run()
    artifact = original.export()
    loaded = require_success(ant.load(artifact, expected=artifact.expectation))
    assert type(loaded) is type(original)
    assert loaded.contrast == original.contrast
    assert loaded.receipt == original.receipt
    assert loaded.identity == original.identity
    assert bytes(loaded.export()) == artifact.data


def test_cross_world_execution_bytes_first_export_dispatch():
    from antecedent import cross_world_execution as cw

    from test_cross_world_execution import _joint_request

    original = require_success(cw.seal(_joint_request()))
    artifact = original.export()
    assert bytes(original.export()) == original.to_bytes() == artifact.data
    loaded = require_success(ant.load(artifact, expected=artifact.expectation))
    assert loaded.result.value == original.value
    assert loaded.shared_draws == original.shared_draws
    assert bytes(loaded.export()) == artifact.data


def test_adapter_bindings_are_checked_before_decision_and_keep_lineage():
    from test_mechanism_composition import chain, compose

    q = quantity()
    source = compose(chain()).as_mean_source(quantity=q, contrast=mc.MeanContrast.mean_at("Y", {}))
    contract = decision.Contract(
        actions=[
            decision.Action("use", inputs=(q,), utility=decision.x(0)),
            decision.Action("wait", inputs=(q,), utility=-decision.x(0)),
        ],
        utility_units="u",
        target_population="pop",
        criterion=decision.Criterion.expected_utility(),
    )
    result = require_success(contract.evaluate(source))
    assert source.provider_id in {link.id for link in result.lineage}
    with pytest.raises(ant.errors.CausalValueError, match="retained producer"):
        require_success(contract.evaluate(replace(source, means=(999.0,))))
    with pytest.raises(ant.errors.CausalValueError, match="cannot issue"):
        require_success(contract.evaluate(replace(source, trust="native_licensed")))


def test_outcome_mean_adapter_composes_directly_with_affine_decision():
    from test_mechanism_composition import chain, compose

    q = quantity()
    producer = Outcome(value=compose(chain()))
    source = producer.as_mean_source(quantity=q, contrast=mc.MeanContrast.mean_at("Y", {}))
    assert source.accepted
    assert require_success(source).means == pytest.approx((7.0,))
    contract = decision.Contract(
        actions=[
            decision.Action("use", inputs=(q,), utility=decision.x(0)),
            decision.Action("wait", inputs=(q,), utility=-decision.x(0)),
        ],
        utility_units="u",
        target_population="pop",
        criterion=decision.Criterion.expected_utility(),
    )
    outcome = contract.evaluate(source)
    assert outcome.accepted
    assert require_success(outcome).selected == ("use",)
    assert require_success(outcome).source is require_success(source)
    with pytest.raises(ant.errors.CausalValueError, match="explicit scientific quantity"):
        producer.as_mean_source(contrast=mc.MeanContrast.mean_at("Y", {}))
    with pytest.raises(ant.errors.CausalValueError, match="population"):
        producer.as_mean_source(
            quantity=replace(q, population_id="another-population"),
            contrast=mc.MeanContrast.mean_at("Y", {}),
        )
    with pytest.raises(ant.errors.CausalValueError, match="units"):
        producer.as_mean_source(
            quantity=replace(q, units="another-unit"),
            contrast=mc.MeanContrast.mean_at("Y", {}),
        )


def test_native_effect_outcome_enters_decision_as_contrast_mean_only():
    import numpy as np

    treatment = np.tile([0.0, 1.0], 24)
    noise = np.repeat(np.linspace(-1.0, 1.0, 24), 2)
    native = ant.estimate(
        ant.prepare(
            {"t": treatment, "y": 2.0 * treatment + noise},
            query=ant.AverageEffect("t", "y"),
            graph=[("t", "y")],
            refute=False,
        )
    )
    q = ScientificQuantity.from_effect(native, outcome_units="u", population="target")
    assert q.functional_id == "mean_difference"

    def contract(utility):
        return decision.Contract(
            actions=[
                decision.Action("use", inputs=(q,), utility=utility),
                decision.Action("wait", inputs=(q,), utility=-decision.x(0)),
            ],
            utility_units="u",
            target_population="target",
            criterion=decision.Criterion.expected_utility(),
        )

    affine = contract(decision.x(0)).evaluate(native)
    assert affine.accepted
    result = require_success(affine)
    assert result.actions[0].expected_utility == pytest.approx(2.0)
    assert isinstance(result.source, MeanSource)
    assert result.source.coordinates == (q,)
    assert result.original_execution is require_success(native)._raw
    nonlinear = contract(decision.x(0) * decision.x(0)).evaluate(native)
    assert not nonlinear.accepted
    assert nonlinear.refusal.code == "decision_contract_unsatisfied"


@pytest.mark.parametrize("producer", ["matched", "descriptive"])
def test_existing_json_families_export_typed_artifacts_and_bind_independently(producer):
    import inspect
    import json

    if producer == "matched":
        from test_matched_case_control import fit, pairs

        original = fit(pairs(7, 3))
    else:
        from test_descriptive_comparison import compare

        original = compare()
    assert isinstance(bytes(original.export()), bytes)
    assert isinstance(original.export(), ExportedArtifact)
    assert inspect.signature(original.export).return_annotation is ExportedArtifact
    envelope = original.export()
    assert envelope.expectation.identity == original.identity
    assert ant.artifacts.inspect(envelope).kind == envelope.kind
    loaded = require_success(ant.load(envelope, expected=envelope.expectation))
    assert type(loaded) is type(original)
    assert bytes(loaded.export()) == bytes(original.export())
    with pytest.raises(ant.errors.CausalSerializationError, match="retained expected_identity"):
        require_success(ant.load(envelope, expected="0" * 64))
    damaged = json.loads(envelope.data)
    damaged["claim"] = "licensed_interval"
    with pytest.raises(ant.errors.CausalSerializationError, match="digest"):
        ant.artifacts.inspect(json.dumps(damaged).encode())


def test_inverse_query_accepts_complete_bridge_and_retains_mean_boundaries():
    from antecedent import inverse_query as iq
    from antecedent import scientific_bridge as sb

    from test_scientific_bridge import surrogate_contract, surrogate_source

    claim = require_success(
        sb.prepare_scientific_bridge(surrogate_contract()).derive(surrogate_source())
    )
    q = claim.quantity
    contract = decision.Contract(
        actions=[
            decision.Action("use", inputs=(q,), utility=decision.x(0)),
            decision.Action("wait", inputs=(q,), utility=-decision.x(0)),
        ],
        utility_units=q.units,
        target_population=q.population_id,
        criterion=decision.Criterion.expected_utility(),
    )
    query = iq.InverseQuery(contract, ("use", "wait"), (iq.target_mean(0.0),))
    direct = require_success(query.evaluate(claim))
    adapted = require_success(query.evaluate(claim.as_mean_source()))
    assert direct.to_dict() == adapted.to_dict()
    with pytest.raises(iq.InverseQueryRefusal):
        require_success(
            iq.InverseQuery(contract, ("use", "wait"), (iq.target_quantile(0.5, 0.0),)).evaluate(
                claim
            )
        )


def test_callback_owner_token_refuses_pointer_rebinding_with_colliding_identities():
    from antecedent import provider_callbacks as pc

    from test_provider_callbacks import exported, gaussian

    owner, declaration, execution, _ = exported()
    other = pc.CallbackSession()
    collision = require_success(
        other.execute(gaussian(), declaration, pc.Request(pc.Query.cdf([0.5, 2.0, 3.5]), seed=9))
    )
    assert collision.history_index == execution.history_index
    assert collision.declaration_identity == execution.declaration_identity
    assert collision.request_identity == execution.request_identity
    assert collision.output_digest == execution.output_digest
    with pytest.raises(pc.CallbackRefusal) as refused:
        bytes(other.export(replace(execution, _session=other), "foreign-pointer"))
    assert refused.value.detail == "provider_callbacks.foreign_execution"
    assert bytes(owner.export(execution, "original"))


def test_registered_legacy_transport_prefixes_keep_their_specialized_consumers():
    from antecedent.transport import advanced as transport

    from test_transport_exact import fixture

    identified, catalog, data = fixture()
    exact = transport.prepare_exact(identified, catalog, data, at={"x": 1.0})
    result = require_success(exact.estimate())
    loaded = require_success(ant.load(bytes(exact.export())))
    assert loaded.probabilities == result.probabilities
    assert loaded.mean("y") == result.mean("y")
    reading = ant.artifacts.inspect(bytes(exact.export()))
    assert reading.kind == "ANTECEDENT-EXACT-TRANSPORT"
    assert bytes(reading.export()) == bytes(exact.export())
    certificate = bytes(identified.export())
    assert ant.artifacts.inspect(certificate).kind == "ANTECEDENT-TRANSPORT-CERTIFICATE"
    assert require_success(ant.load(certificate)).outcome == identified.outcome


def test_native_and_portable_mean_adapters_refuse_relabelled_values_and_evidence():
    from antecedent import scientific_bridge as sb

    from test_measurement_models import prepare_classical, simulate
    from test_scientific_bridge import surrogate_contract, surrogate_source

    bridge = require_success(
        sb.prepare_scientific_bridge(surrogate_contract()).derive(surrogate_source())
    )
    measured = require_success(prepare_classical(0.6, 0.0, simulate(120, 37)).estimate())
    q = quantity("outcome", "unit", "cohort", "observational")
    for altered, declarations in (
        (replace(bridge, value=99.0), {}),
        (replace(bridge, evidence=("fabricated-evidence",)), {}),
        (replace(bridge, claim_id="0" * 64), {}),
        (replace(measured, point=99.0), {"quantity": q}),
        (replace(measured, data_identity="0" * 64), {"quantity": q}),
    ):
        with pytest.raises(ant.errors.CausalValueError, match="retained producer binding"):
            altered.as_mean_source(**declarations)
    assert bridge.as_mean_source().means == (bridge.value,)
    assert measured.as_mean_source(quantity=q).means == (measured.point,)


def test_public_artifact_envelopes_validate_and_detach_inputs():
    from antecedent.artifacts import ArtifactExpectation, ExportedArtifact, LoadContext

    raw = bytearray(b"retained")
    metadata = {"reading": [1, 2]}
    envelope = ExportedArtifact(raw, "declared", metadata=metadata)
    raw[0] = 0
    metadata["reading"][0] = 99
    assert envelope.data == b"retained"
    assert envelope.metadata["reading"] == (1, 2)
    for fields in (
        {"data": "bad", "kind": "x"},
        {"data": b"x", "kind": ""},
        {"data": b"x", "kind": "x", "metadata": []},
    ):
        with pytest.raises(ant.errors.CausalTypeError, match="ExportedArtifact"):
            ExportedArtifact(**fields)
    with pytest.raises(ant.errors.CausalTypeError, match="kind"):
        ArtifactExpectation("", None)
    with pytest.raises(ant.errors.CausalTypeError, match="actual_claims"):
        LoadContext(actual_claims=[])


def test_legacy_transport_load_checks_explicit_native_execution_bindings():
    from antecedent.artifacts import ArtifactExpectation, LoadContext
    from antecedent.transport import advanced as transport

    from test_transport_exact import fixture

    identified, catalog, data = fixture()
    exact = transport.prepare_exact(identified, catalog, data, at={"x": 1.0})
    original = require_success(exact.estimate())
    expectation = ArtifactExpectation(
        "ANTECEDENT-EXACT-TRANSPORT", {"execution_id": original.inspect().execution_id}
    )
    loaded = require_success(ant.load(bytes(exact.export()), expected=expectation))
    assert loaded.acceptance.identity_bound and loaded.acceptance.replayable
    assert not loaded.acceptance.resumable
    with pytest.raises(ant.errors.CausalSerializationError, match="execution_id"):
        require_success(ant.load(bytes(exact.export()), expected={"execution_id": "0" * 64}))
    with pytest.raises(ant.errors.CausalSerializationError, match="kind"):
        require_success(
            ant.load(
                bytes(exact.export()), expected=ArtifactExpectation("wrong", expectation.identity)
            )
        )
    with pytest.raises(ant.errors.CausalValueError, match="context"):
        require_success(ant.load(bytes(exact.export()), context=LoadContext()))


@pytest.mark.parametrize("trust", ["native_licensed", "verified_extension"])
def test_mean_declarations_cannot_issue_native_or_verified_trust(trust):
    with pytest.raises(ant.errors.CausalValueError, match="cannot issue"):
        MeanSource(
            (quantity(),),
            (1.0,),
            "declared-provider",
            "declared-data",
            "declared-contract",
            trust=trust,
        )
