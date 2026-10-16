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
    data = report.export(artifact_id="transport-revalidation")
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
