"""Transported claims are revalidated through the Python surface when a source, the target or the
support changes, so the rule that a calibration record is never inherited by another target runs in
the product path and its classification surfaces through ``inspect`` and the exported
composition verification.

Oracles (hand-classified; the Python spelling of
``crates/antecedent/tests/transport_revalidation_facade.rs``):

- nothing changed: every claim and the decision resting on it is ``still_valid``;
- one source's evidence changed: the claims counting it and the decision resting on them
  ``recompute``; the claim of another source is untouched;
- the target's identity changed while a record was established at the earlier target: the claim is
  ``withdrawn`` for calibration with its point value left as it was;
- the target outside its support: every transported claim is ``withdrawn``.
"""

from __future__ import annotations

import pytest
from antecedent import composition_verification as cv
from antecedent import transported_response as tr
from antecedent.errors import CausalValueError
from antecedent.transport import revalidation as rv

from _refusal import assert_registered_refusal
from _transport_fixtures import digest, effect, transport_declaration


def response(variable: str, source: str, target: str, value: float) -> tr.TransportedResponse:
    return tr.TransportedResponse.declared_point(
        [effect(target, variable)],
        [value],
        transport_declaration(
            target,
            source=source,
            identity=f"transport:{variable}:{source}",
            snapshot=f"snapshot:{variable}",
            standard_error=None,
        ),
    )


def scene(
    target_data: int, trial_evidence: int, calibration_at: str | None = None
) -> rv.TransportScene:
    """Two transported claims (one per source population) and a decision resting on both."""
    held = (
        rv.Calibration.point_only()
        if calibration_at is None
        else rv.Calibration.held(digest(77), calibration_at)
    )
    return (
        rv.TransportScene.builder("clinic_b", digest(target_data))
        .source("trial", digest(trial_evidence))
        .source("registry", digest(20))
        .transported("effect_trial", response("effect", "trial", "clinic_b", 2.0), held)
        .transported(
            "effect_registry",
            response("effect_r", "registry", "clinic_b", 1.5),
            rv.Calibration.point_only(),
        )
        .decision("treat_or_wait", digest(40), digest(41), ["effect_trial", "effect_registry"])
        .build()
    )


def classes(report: rv.TransportRevalidation) -> dict[str, str]:
    return {c.key: c.classification for c in report.claims}


def refusal_of(call) -> tr.TransportedResponseRefusal:
    with pytest.raises(tr.TransportedResponseRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def test_an_unchanged_scene_is_still_valid_and_its_classification_replays_from_bytes() -> None:
    first = scene(1, 10)
    report = rv.revalidate(first, scene(1, 10))
    for key in ("effect_trial", "effect_registry", "treat_or_wait"):
        claim = report.claim(key)
        assert claim.classification == "still_valid", key
        assert claim.reasons == ()
        assert report.require_valid(key).key == key
    assert len(report.claims) == 3
    assert report.removed == ()

    # The same graphs sealed as `composition_verification_v1`: a fresh consumer replays.
    data = bytes(report.export(artifact_id="transport-revalidation"))
    consumed = cv.consume_composition_verification_artifact(
        data, expected_identity=report.verification_identity
    )
    assert consumed.classify(rv.claim_node_key("effect_trial")).klass == "still_valid"

    # A first observation (nothing earlier) classifies the same scene the same way.
    assert classes(rv.revalidate(None, first))["treat_or_wait"] == "still_valid"


def test_a_changed_source_recomputes_only_the_claims_that_count_it() -> None:
    report = rv.revalidate(scene(1, 10), scene(1, 11))
    assert classes(report) == {
        "effect_registry": "still_valid",
        "effect_trial": "recompute",
        "treat_or_wait": "recompute",
    }
    effect_claim = report.claim("effect_trial")
    assert "source_evidence_changed" in effect_claim.reasons
    # The earlier fit and uncertainty carry no authority and a refusal names the remedy.
    assert effect_claim.prior_authority == "not_retained"
    assert effect_claim.remedies
    refusal = refusal_of(lambda: report.require_valid("effect_trial"))
    assert refusal.remedy


def test_a_calibration_record_is_never_inherited_by_another_target() -> None:
    builder = rv.TransportScene.builder("clinic_b", digest(1))
    own_target = builder.target_identity
    before = scene(1, 10, own_target)
    held = rv.revalidate(before, scene(1, 10, own_target))
    assert held.claim("effect_trial").classification == "still_valid"
    assert not held.claim("effect_trial").calibration_withdrawn

    # The target's data changed: the claim now answers another target, and the record still
    # names the earlier one.
    after = scene(2, 10, own_target)
    report = rv.revalidate(before, after)
    claim = report.claim("effect_trial")
    assert claim.classification == "withdrawn"
    assert "calibration_from_another_target" in claim.reasons
    assert claim.calibration_withdrawn
    refusal = refusal_of(lambda: report.require_valid("effect_trial"))
    assert refusal.reason_code == "cell_not_licensed"
    # A point-only claim of the same scene is recomputed, not withdrawn: it states no guarantee
    # to lose.
    assert report.claim("effect_registry").classification == "recompute"


def test_an_unsupported_target_withdraws_every_transported_claim() -> None:
    before = scene(1, 10)
    after = (
        rv.TransportScene.builder("clinic_b", digest(1))
        .target_support(["age"])
        .source("trial", digest(10))
        .source("registry", digest(20))
        .transported(
            "effect_trial",
            response("effect", "trial", "clinic_b", 2.0),
            rv.Calibration.point_only(),
        )
        .build()
    )
    report = rv.revalidate(before, after)
    claim = report.claim("effect_trial")
    assert claim.classification == "withdrawn"
    assert any(reason == "target_unsupported" for reason in claim.reasons)
    # The claims the later scene dropped are reported as removed.
    assert report.removed == ("effect_registry", "treat_or_wait")


def test_a_refused_observation_leaves_the_state_unchanged() -> None:
    state = rv.TransportClaimState()
    assert state.scene is None
    first = scene(1, 10)
    state.observe(first)
    assert state.scene is not None
    assert state.scene.selection_diagram == first.selection_diagram
    assert state.scene.claims == first.claims

    # A claim about another target does not build, so it never reaches the state.
    refusal_of(
        lambda: (
            rv.TransportScene.builder("clinic_b", digest(1))
            .source("trial", digest(10))
            .transported(
                "elsewhere",
                response("effect", "trial", "clinic_c", 2.0),
                rv.Calibration.point_only(),
            )
            .build()
        )
    )
    assert state.scene is not None
    assert state.scene.claims == first.claims

    # A good scene advances the state.
    report = state.observe(scene(1, 11))
    assert classes(report)["effect_trial"] == "recompute"
    assert state.scene is not None
    assert state.scene.claims == scene(1, 11).claims


def test_the_builder_refuses_what_a_scene_cannot_state() -> None:
    elsewhere = refusal_of(
        lambda: (
            rv.TransportScene.builder("clinic_b", digest(1))
            .source("trial", digest(10))
            .transported(
                "elsewhere",
                response("effect", "trial", "clinic_c", 2.0),
                rv.Calibration.point_only(),
            )
            .build()
        )
    )
    assert elsewhere.detail == "transport_revalidation.claim_target_mismatch"
    assert elsewhere.reason_code == "quantity_semantics_mismatch"

    orphan = refusal_of(
        lambda: (
            rv.TransportScene.builder("clinic_b", digest(1))
            .transported(
                "orphan", response("effect", "trial", "clinic_b", 2.0), rv.Calibration.point_only()
            )
            .build()
        )
    )
    assert orphan.detail == "transport_revalidation.source_undeclared"

    repeated = refusal_of(
        lambda: (
            rv.TransportScene.builder("clinic_b", digest(1))
            .source("trial", digest(10))
            .transported(
                "effect", response("effect", "trial", "clinic_b", 2.0), rv.Calibration.point_only()
            )
            .transported(
                "effect", response("effect", "trial", "clinic_b", 3.0), rv.Calibration.point_only()
            )
            .build()
        )
    )
    assert repeated.detail == "transport_revalidation.duplicate_claim"


def test_the_scene_identity_follows_the_response_value_at_an_unchanged_shape() -> None:
    def build(value: float) -> rv.TransportScene:
        return (
            rv.TransportScene.builder("clinic_b", digest(1))
            .source("trial", digest(10))
            .transported(
                "effect",
                response("effect", "trial", "clinic_b", value),
                rv.Calibration.point_only(),
            )
            .build()
        )

    a, b = build(2.0), build(2.5)
    claim_a, claim_b = a.claim("effect"), b.claim("effect")
    assert claim_a["identity"] != claim_b["identity"]
    # Target, quantity and structure are unchanged: only the claim's own numbers moved.
    assert claim_a["quantity"] == claim_b["quantity"]
    assert claim_a["structure"] == claim_b["structure"]
    assert a.target_identity == b.target_identity
    assert classes(rv.revalidate(a, b))["effect"] == "recompute"


def test_observed_effects_name_the_node_they_were_observed_on() -> None:
    report = rv.revalidate(
        scene(1, 10),
        scene(1, 10),
        [rv.ObservedEffect(rv.claim_node_key("effect_registry"), "recomputed", "stale stage")],
    )
    assert classes(report) == {
        "effect_registry": "recompute",
        "effect_trial": "still_valid",
        "treat_or_wait": "recompute",
    }
    assert "observed_stale" in report.claim("effect_registry").reasons
    inspected = report.inspect()
    assert inspected.family == "TransportRevalidation"


def test_observed_effects_on_a_source_node_recompute_the_claims_that_count_it() -> None:
    assert rv.source_node_key("registry") == "source:registry"
    report = rv.revalidate(
        scene(1, 10),
        scene(1, 10),
        [rv.ObservedEffect(rv.source_node_key("registry"), "recomputed", "registry re-extracted")],
    )
    assert classes(report) == {
        "effect_registry": "recompute",
        "effect_trial": "still_valid",
        "treat_or_wait": "recompute",
    }
    assert "source_evidence_changed" in report.claim("effect_registry").reasons
    assert report.claim("effect_trial").reasons == ()


def native_response(value: float = 2.0) -> tr.TransportedResponse:
    """A response of a library route: it rests on no provider."""
    return tr.TransportedResponse.declared_point(
        [effect("clinic_b", "effect")],
        [value],
        transport_declaration(
            "clinic_b", route="cohort_effect", standard_error=None, diagram=digest(7)
        ),
    )


def test_a_scene_claim_reads_its_trust_from_provenance_and_refuses_a_stated_level_on_a_library_route() -> (
    None
):
    def one_claim(claim_response: tr.TransportedResponse, level: str | None) -> rv.TransportScene:
        return (
            rv.TransportScene.builder("clinic_b", digest(1))
            .source("trial", digest(10))
            .transported_claim("effect", claim_response, provider_trust=level)  # type: ignore[arg-type]
            .build()
        )

    # A declared route is another party's claim: no stated level withdraws it, a verified one holds.
    undeclared = rv.revalidate(None, one_claim(response("effect", "trial", "clinic_b", 2.0), None))
    assert undeclared.claim("effect").classification == "withdrawn"
    with pytest.raises(tr.TransportedResponseRefusal) as unusable:
        undeclared.require_valid("effect")
    assert_registered_refusal(unusable.value)
    unverified = rv.revalidate(
        None, one_claim(response("effect", "trial", "clinic_b", 2.0), "unverified")
    )
    assert unverified.claim("effect").classification == "withdrawn"
    verified = rv.revalidate(
        None, one_claim(response("effect", "trial", "clinic_b", 2.0), "verified_extension")
    )
    assert verified.claim("effect").classification == "still_valid"
    # The scene records the trust each claim was given.
    assert (
        one_claim(response("effect", "trial", "clinic_b", 2.0), None).claim("effect")["trust"]
        == "undeclared"
    )
    assert (
        one_claim(response("effect", "trial", "clinic_b", 2.0), "verified_extension").claim(
            "effect"
        )["trust"]
        == "held"
    )

    # A library route rests on no provider: not_applicable, still valid; a level is refused.
    native = rv.revalidate(None, one_claim(native_response(), None))
    assert native.claim("effect").classification == "still_valid"
    refused = refusal_of(lambda: one_claim(native_response(), "verified_extension"))
    assert refused.detail == "transport_revalidation.provider_trust_not_applicable"
    assert refused.reason_code == "invalid_argument"
    with pytest.raises(CausalValueError, match="provider_trust"):
        one_claim(response("effect", "trial", "clinic_b", 2.0), "certified")


def test_a_standalone_response_states_its_coordinate_standing_and_composition_node() -> None:
    declared = response("effect", "trial", "clinic_b", 2.0)
    coordinate = rv.response_coordinate(declared)
    assert len(coordinate) == 64 and int(coordinate, 16) >= 0
    # The coordinate names the target and the quantity, not the value or the source.
    assert coordinate == rv.response_coordinate(response("effect", "registry", "clinic_b", 9.0))
    assert coordinate != rv.response_coordinate(response("effect_r", "trial", "clinic_b", 2.0))
    assert coordinate != rv.response_coordinate(response("effect", "trial", "clinic_c", 2.0))

    stated = rv.response_claim_standing(
        declared, cv.Calibration.held(digest(77), coordinate), provider_trust="verified_extension"
    )
    assert (stated.support, stated.unsupported, stated.trust) == ("supported", (), "held")
    assert stated.calibration == cv.Calibration.held(digest(77), coordinate)
    assert rv.response_claim_standing(declared).trust == "undeclared"
    assert rv.response_claim_standing(declared, provider_trust="unverified").trust == "withdrawn"
    assert rv.response_claim_standing(native_response()).trust == "not_applicable"
    refused = refusal_of(
        lambda: rv.response_claim_standing(native_response(), provider_trust="unverified")
    )
    assert refused.detail == "transport_revalidation.provider_trust_not_applicable"

    node = rv.response_claim_node(
        "effect",
        declared,
        cv.Calibration.held(digest(77), coordinate),
        provider_trust="verified_extension",
    )
    assert node.kind == "transported_result" and node.key == "effect"
    assert node.coordinate == coordinate
    assert node.trust == "held" and node.transformation == "declared"
    assert node.assumptions == ("selection acts on age only",)
    assert node.calibration == cv.Calibration.held(digest(77), coordinate)
    other = rv.response_claim_node(
        "effect", response("effect", "trial", "clinic_b", 3.0), provider_trust="verified_extension"
    )
    assert other.identity != node.identity and other.coordinate == node.coordinate
    assert other.structure == node.structure

    # The node composes in a verification graph: held trust holds, an unstated level withdraws.
    first = cv.verify_composition(cv.CompositionGraph(nodes=(node,)))
    assert first.classify("effect").klass == "still_valid"
    unstated = rv.response_claim_node("effect", declared)
    assert unstated.trust == "undeclared"
    assert (
        cv.verify_composition(cv.CompositionGraph(nodes=(unstated,))).classify("effect").klass
        == "withdrawn"
    )
    replaced = cv.verify_composition(
        cv.CompositionGraph(nodes=(other,)), previous=cv.CompositionGraph(nodes=(node,))
    )
    assert replaced.classify("effect").klass == "recompute"
