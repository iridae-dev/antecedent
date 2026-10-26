"""Typed scientific bridges through the Python facade.

Every asserted value has an oracle computed here in pure Python or numpy, never by native code:

* surrogate to terminal outcome: ``numpy.polyfit`` and ``numpy.corrcoef`` over the validation
  trials (slope 0.92, intercept 0.06, ``R^2 = 529/535``, so an effect 0.8 maps to 0.796);
* measured and latent prevalence: the misclassification algebra
  ``p* = Se*pi + (1 - Sp)*(1 - pi)`` inverted by hand (0.31 at Se 0.9, Sp 0.8 is 11/70);
* predictive and causal quantity: an enumerated confounded structural model whose association is
  0.54 and whose interventional effect is 0.30;
* affine utility, parameter contrast and calibration: closed-form products.

Calibration of any interval is deliberately not measured: the interval route is closed and these
tests assert that it refuses.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys

import numpy as np
import pytest
from antecedent import measurement_models as mm
from antecedent import scientific_bridge as sb
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success
from antecedent.scientific_bridge import ScientificBridgeRefusal

from _refusal import assert_registered_refusal

TRIALS = [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)]


# --------------------------------------------------------------------------- fixtures


def outcome(variable, *, units, population="adults", regime="do(t=1)", functional="average_effect"):
    return ScientificQuantity.outcome(
        variable, units=units, population=population, regime=regime, functional=functional
    )


def surrogate_contract(*, population="adults", min_trials=4, min_r_squared=0.9, **overrides):
    fields = dict(
        kind="surrogate_terminal_outcome",
        source=outcome("trial:biomarker", units="mmol_per_l", population=population),
        source_class="surrogate_endpoint",
        target=outcome("trial:survival", units="log_hazard", population=population),
        target_class="terminal_outcome",
        evidence_scope=sb.Scope("adults", "do(t=1)"),
        mapping=sb.TrialLevelMapping(
            trials=TRIALS, min_trials=min_trials, min_r_squared=min_r_squared
        ),
        assumptions=(
            sb.Assumption(
                "validation",
                "surrogate_validation",
                "the surrogate is validated at the trial level",
                evidence="meta-analysis:ckd-2024",
            ),
        ),
    )
    fields.update(overrides)
    return sb.BridgeContract(**fields)


def surrogate_source(value=0.8, label="biomarker-effect", population="adults"):
    return sb.SourceClaim.declared(
        label, value, outcome("trial:biomarker", units="mmol_per_l", population=population)
    )


def assert_refusal(error, code, detail):
    assert isinstance(error, ScientificBridgeRefusal)
    assert_registered_refusal(error)
    assert error.reason_code == code, (error.reason_code, str(error))
    assert error.detail == detail, (error.detail, str(error))
    assert error.remedy, "a refusal states what would license the request"


def refused(call, code, detail):
    with pytest.raises(ScientificBridgeRefusal) as caught:
        call()
    assert_refusal(caught.value, code, detail)
    return caught.value


def clinic_model(sensitivity=0.9, specificity=0.8, min_youden=0.1) -> mm.MeasurementModel:
    return mm.MeasurementModel(
        indicator=mm.MeasurementVariable("schema:flag", "noisy_misclassified", "binary"),
        construct=mm.MeasurementVariable("construct:disease", "latent_scientific", "binary"),
        parameters=mm.BinaryParameters(
            (mm.BinaryRates(sensitivity, specificity, 0.0, 0.0),), min_youden=min_youden
        ),
        population_id="clinic-a",
        regime_id="observational",
        evidence=mm.ParameterEvidence.declared("lab"),
    )


def prevalence(variable, population="clinic-a"):
    return outcome(
        variable,
        units="probability",
        population=population,
        regime="observational",
        functional="prevalence",
    )


def correction_contract(model=None, *, source_class="measured_outcome", **overrides):
    model = model or clinic_model()
    measured, latent = "schema:flag", "construct:disease"
    if source_class == "measured_outcome":
        source, target = prevalence(measured), prevalence(latent)
        target_class = "latent_construct"
    else:
        source, target = prevalence(latent), prevalence(measured)
        target_class = "measured_outcome"
    identity = mm.model_identities(model).model
    fields = dict(
        kind="latent_measured_outcome",
        source=source,
        source_class=source_class,
        target=target,
        target_class=target_class,
        evidence_scope=sb.Scope("clinic-a", "observational"),
        mapping=sb.MeasurementCorrection(model),
        assumptions=(
            sb.Assumption(
                "flag-model",
                "measurement_model",
                "the flag relates to the disease by the declared misclassification model",
                evidence=identity,
            ),
        ),
    )
    fields.update(overrides)
    return sb.BridgeContract(**fields)


# --------------------------------------------------------------------------- licensed bridges


def test_the_surrogate_bridge_is_the_numpy_trial_level_regression():
    s = np.array([t[0] for t in TRIALS])
    t = np.array([t[1] for t in TRIALS])
    slope, intercept = np.polyfit(s, t, 1)
    r_squared = float(np.corrcoef(s, t)[0, 1] ** 2)
    assert slope == pytest.approx(0.92, abs=1e-12)
    assert r_squared == pytest.approx(529 / 535, abs=1e-12)

    bridge = sb.prepare_scientific_bridge(surrogate_contract())
    source = surrogate_source(0.8)
    claim = require_success(bridge.derive(source))

    assert claim.value == pytest.approx(intercept + slope * 0.8, abs=1e-12)
    assert claim.value == pytest.approx(0.796, abs=1e-12)
    assert claim.fit.slope == pytest.approx(slope, abs=1e-12)
    assert claim.fit.intercept == pytest.approx(intercept, abs=1e-12)
    assert claim.fit.r_squared == pytest.approx(r_squared, abs=1e-12)
    assert (claim.fit.trials, claim.fit.surrogate_min, claim.fit.surrogate_max) == (4, 0.0, 1.5)
    assert (claim.value_kind, claim.mapping, claim.units) == (
        "mean",
        "trial_level_linear",
        "log_hazard",
    )
    assert not claim.transported
    # A different claim from its source, which stays inspectable and unchanged.
    assert claim.claim_id != source.claim_id
    assert claim.source_claim_id == source.claim_id
    assert claim.source()["claim_id"] == source.claim_id
    assert claim.source()["value"] == 0.8
    # Nothing is established for the target quantity beyond the declared assumptions.
    assert not claim.identification_available and not claim.uncertainty_available
    assert claim.domains["identification"] == "unknown"
    assert claim.unresolved == ("assumption:validation",)
    assert {"interval", "posterior", "joint_law"} <= set(claim.unavailable)
    assert [node["operation"] for node in claim.provenance] == [
        "source_claim",
        "scientific_bridge:surrogate_terminal_outcome",
    ]
    assert claim.provenance[1]["parents"] == (source.claim_id,)
    assert claim.assumptions[0].evidence == "meta-analysis:ckd-2024"


def test_the_derivation_is_deterministic_and_the_identity_binds_the_source_claim():
    bridge = sb.prepare_scientific_bridge(surrogate_contract())
    first = require_success(bridge.derive(surrogate_source(0.8)))
    again = require_success(bridge.derive(surrogate_source(0.8)))
    assert (first.claim_id, first.value) == (again.claim_id, again.value)
    # A different source claim at the same value is a different derived claim under one contract.
    other = require_success(bridge.derive(surrogate_source(0.8, label="another-claim")))
    assert other.claim_id != first.claim_id
    assert other.contract_identity == first.contract_identity
    assert other.value == first.value
    # The identity function agrees with the prepared bridge, and ignores assumption order.
    assert sb.contract_identity(surrogate_contract()) == bridge.contract_identity
    extra = sb.Assumption("note", "additional", "recorded")
    base = surrogate_contract()
    forward = surrogate_contract(assumptions=(*base.assumptions, extra))
    backward = surrogate_contract(assumptions=(extra, *base.assumptions))
    assert sb.contract_identity(forward) == sb.contract_identity(backward)
    assert sb.contract_identity(forward) != bridge.contract_identity
    reordered = surrogate_contract(
        mapping=sb.TrialLevelMapping(TRIALS[::-1], min_trials=4, min_r_squared=0.9)
    )
    assert sb.contract_identity(reordered) == bridge.contract_identity


def test_a_measured_prevalence_is_the_latent_prevalence_only_through_the_model():
    sensitivity, specificity = 0.9, 0.8
    recorded = 0.31
    expected = (recorded + specificity - 1) / (sensitivity + specificity - 1)
    assert expected == pytest.approx(11 / 70, abs=1e-15)

    bridge = sb.prepare_scientific_bridge(correction_contract())
    source = sb.SourceClaim.declared("flag-prevalence", recorded, prevalence("schema:flag"))
    latent = require_success(bridge.derive(source))
    assert latent.value == pytest.approx(expected, abs=1e-12)
    assert abs(latent.value - recorded) > 0.1, "the measured value is not the construct"
    assert latent.mapping == "measurement_correction"
    assert latent.assumptions[0].kind == "measurement_model"

    # The forward map recovers the recorded prevalence: the algebra p* = Se pi + (1 - Sp)(1 - pi).
    forward = sb.prepare_scientific_bridge(correction_contract(source_class="latent_construct"))
    pi = 0.5
    back = require_success(
        forward.derive(
            sb.SourceClaim.declared("latent-prevalence", pi, prevalence("construct:disease"))
        )
    )
    assert back.value == pytest.approx(sensitivity * pi + (1 - specificity) * (1 - pi), abs=1e-12)
    exact = require_success(
        forward.derive(
            sb.SourceClaim.declared("latent-from-bridge", expected, prevalence("construct:disease"))
        )
    )
    assert exact.value == pytest.approx(recorded, abs=1e-12)


def enumerated_confounded_truth():
    """Z ~ Bernoulli(.5); P(X=1|Z)=.2+.6Z; P(Y=1|X,Z)=.1+.3X+.4Z. (association, effect)."""
    joint = {}
    for z, x, y in itertools.product((0, 1), repeat=3):
        px = 0.2 + 0.6 * z
        py = 0.1 + 0.3 * x + 0.4 * z
        joint[(z, x, y)] = 0.5 * (px if x else 1 - px) * (py if y else 1 - py)
    p_x = {x: sum(p for (z, xx, y), p in joint.items() if xx == x) for x in (0, 1)}
    p_xy = {x: sum(p for (z, xx, y), p in joint.items() if xx == x and y == 1) for x in (0, 1)}
    association = p_xy[1] / p_x[1] - p_xy[0] / p_x[0]
    standardized = {
        x: sum(
            0.5
            * sum(p for (zz, xx, y), p in joint.items() if zz == z and xx == x and y == 1)
            / sum(p for (zz, xx, y), p in joint.items() if zz == z and xx == x)
            for z in (0, 1)
        )
        for x in (0, 1)
    }
    return association, standardized[1] - standardized[0]


def predictive_contract(assumptions=(), source_class="predictive_quantity"):
    return sb.BridgeContract(
        kind="predictive_causal_quantity",
        source=outcome(
            "cohort:y",
            units="probability",
            population="registry",
            regime="observational",
            functional="contrast",
        ),
        source_class=source_class,
        target=outcome(
            "cohort:y",
            units="probability",
            population="registry",
            regime="do(x=1)",
            functional="contrast",
        ),
        target_class="causal_quantity",
        evidence_scope=sb.Scope("registry", "observational"),
        mapping=sb.IdentityMapping(),
        assumptions=tuple(assumptions),
    )


def test_an_associational_quantity_is_never_a_causal_quantity_without_an_identification():
    assert "associational_quantity" in sb.QUANTITY_CLASSES
    error = refused(
        lambda: sb.prepare_scientific_bridge(
            predictive_contract(source_class="associational_quantity")
        ),
        "bridge_substitution_refused",
        "scientific_bridge.association_as_intervention",
    )
    assert "associational_quantity" in error.message


def test_an_association_is_a_causal_quantity_only_under_a_named_identification():
    association, effect = enumerated_confounded_truth()
    assert association == pytest.approx(0.54, abs=1e-12)
    assert effect == pytest.approx(0.30, abs=1e-12)

    error = refused(
        lambda: sb.prepare_scientific_bridge(predictive_contract()),
        "bridge_substitution_refused",
        "scientific_bridge.association_as_intervention",
    )
    assert association - effect == pytest.approx(0.24, abs=1e-12)  # the wrong number refused
    assert "predictive_quantity" in error.message

    identified = (
        sb.Assumption(
            "backdoor",
            "identification",
            "Z blocks every back-door path",
            evidence="identification:backdoor-z",
        ),
    )
    bridge = sb.prepare_scientific_bridge(predictive_contract(identified))
    source = sb.SourceClaim.declared(
        "standardized-contrast",
        effect,
        outcome(
            "cohort:y",
            units="probability",
            population="registry",
            regime="observational",
            functional="contrast",
        ),
    )
    claim = require_success(bridge.derive(source))
    assert claim.value == pytest.approx(effect, abs=1e-12)
    assert claim.mapping == "identity"


def test_stated_parameter_and_utility_bridges_have_closed_form_values():
    stated = outcome(
        "survey:take_up",
        units="probability",
        population="panel",
        regime="observational",
        functional="mean",
    )
    realized = outcome(
        "admin:take_up",
        units="probability",
        population="panel",
        regime="observational",
        functional="mean",
    )
    preference = sb.BridgeContract(
        kind="preference_realized_behavior",
        source=stated,
        source_class="stated_preference",
        target=realized,
        target_class="realized_behavior",
        evidence_scope=sb.Scope("panel", "observational"),
        mapping=sb.AffineMapping(0.0, 0.6),
        assumptions=(
            sb.Assumption(
                "hypothetical-bias",
                "behavioral_linkage",
                "take-up is 0.6 of stated",
                evidence="panel-2023",
            ),
        ),
    )
    value = require_success(
        sb.prepare_scientific_bridge(preference).derive(
            sb.SourceClaim.declared("stated", 0.5, stated)
        )
    ).value
    assert value == pytest.approx(0.6 * 0.5, abs=1e-12)

    parameter = outcome(
        "fit:dose",
        units="per_unit_dose",
        population="trial",
        regime="observational",
        functional="coefficient",
    )
    response = outcome(
        "fit:dose",
        units="outcome_units",
        population="trial",
        regime="do(dose=+0.4)",
        functional="average_effect",
    )
    contrast = sb.BridgeContract(
        kind="parameter_intervention_response",
        source=parameter,
        source_class="model_parameter",
        target=response,
        target_class="intervention_response",
        evidence_scope=sb.Scope("trial", "observational"),
        mapping=sb.AffineMapping(0.0, 0.4),
        assumptions=(
            sb.Assumption(
                "structural-linear",
                "parameter_interpretation",
                "the coefficient is the response per unit dose",
            ),
        ),
    )
    value = require_success(
        sb.prepare_scientific_bridge(contrast).derive(
            sb.SourceClaim.declared("coefficient", 2.5, parameter)
        )
    ).value
    assert value == pytest.approx(2.5 * 0.4, abs=1e-12)

    endpoint = outcome("trial:qaly", units="qaly", population="adults", functional="average_effect")
    utility = ScientificQuantity.utility(
        "payer:value",
        units="currency",
        population="adults",
        regime="do(t=1)",
        functional="expected_utility",
    )
    priced = sb.BridgeContract(
        kind="scientific_decision_utility",
        source=endpoint,
        source_class="scientific_endpoint",
        target=utility,
        target_class="decision_utility",
        evidence_scope=sb.Scope("adults", "do(t=1)"),
        mapping=sb.AffineMapping(0.0, 30_000.0),
        assumptions=(
            sb.Assumption(
                "wtp", "utility_mapping", "30000 per QALY", evidence="utility-function:wtp-30000"
            ),
        ),
    )
    value = require_success(
        sb.prepare_scientific_bridge(priced).derive(
            sb.SourceClaim.declared("qaly-gain", 0.3, endpoint)
        )
    ).value
    assert value == pytest.approx(0.3 * 30_000.0, rel=1e-12)


# --------------------------------------------------------------------------- named refusals


def test_the_six_named_substitutions_are_refused_with_registered_codes_and_remedies():
    # proxy as construct: a measured value with no measurement model
    refused(
        lambda: sb.prepare_scientific_bridge(correction_contract(assumptions=())),
        "proxy_offered_as_construct",
        "scientific_bridge.proxy_as_construct",
    )
    # stated preference as realized behavior
    stated = outcome(
        "survey:take_up",
        units="probability",
        population="panel",
        regime="observational",
        functional="mean",
    )
    realized = outcome(
        "admin:take_up",
        units="probability",
        population="panel",
        regime="observational",
        functional="mean",
    )
    refused(
        lambda: sb.prepare_scientific_bridge(
            sb.BridgeContract(
                kind="preference_realized_behavior",
                source=stated,
                source_class="stated_preference",
                target=realized,
                target_class="realized_behavior",
                evidence_scope=sb.Scope("panel", "observational"),
                mapping=sb.IdentityMapping(),
            )
        ),
        "bridge_substitution_refused",
        "scientific_bridge.stated_preference_as_realized_behavior",
    )
    # scientific endpoint as utility
    endpoint = outcome("trial:qaly", units="qaly", population="adults", functional="average_effect")
    utility = ScientificQuantity.utility(
        "payer:value",
        units="currency",
        population="adults",
        regime="do(t=1)",
        functional="expected_utility",
    )
    refused(
        lambda: sb.prepare_scientific_bridge(
            sb.BridgeContract(
                kind="scientific_decision_utility",
                source=endpoint,
                source_class="scientific_endpoint",
                target=utility,
                target_class="decision_utility",
                evidence_scope=sb.Scope("adults", "do(t=1)"),
                mapping=sb.AffineMapping(0.0, 30_000.0),
            )
        ),
        "bridge_substitution_refused",
        "scientific_bridge.scientific_endpoint_as_utility",
    )
    # predictive importance as causal effect and model score as probability: closed whatever is
    # declared
    everything = (
        sb.Assumption("backdoor", "identification", "declared", evidence="identification:any"),
        sb.Assumption("note", "additional", "declared"),
    )
    feature = outcome(
        "m:feature",
        units="probability",
        population="registry",
        regime="observational",
        functional="importance",
    )
    effect = outcome(
        "m:feature",
        units="probability",
        population="registry",
        regime="do(x=1)",
        functional="effect",
    )
    refused(
        lambda: sb.prepare_scientific_bridge(
            sb.BridgeContract(
                kind="predictive_causal_quantity",
                source=feature,
                source_class="predictive_importance",
                target=effect,
                target_class="causal_quantity",
                evidence_scope=sb.Scope("registry", "observational"),
                mapping=sb.IdentityMapping(),
                assumptions=everything,
            )
        ),
        "bridge_substitution_refused",
        "scientific_bridge.predictive_importance_as_causal_effect",
    )
    score = outcome(
        "m:score", units="score", population="registry", regime="observational", functional="mean"
    )
    probability = outcome(
        "m:score",
        units="probability",
        population="registry",
        regime="observational",
        functional="mean",
    )
    refused(
        lambda: sb.prepare_scientific_bridge(
            sb.BridgeContract(
                kind="predictive_causal_quantity",
                source=score,
                source_class="model_score",
                target=probability,
                target_class="predictive_quantity",
                evidence_scope=sb.Scope("registry", "observational"),
                mapping=sb.IdentityMapping(),
                assumptions=everything,
            )
        ),
        "bridge_substitution_refused",
        "scientific_bridge.model_score_as_probability",
    )


def test_a_missing_assumption_and_an_assumption_naming_no_evidence_are_refused():
    contract = surrogate_contract(assumptions=())
    refused(
        lambda: sb.prepare_scientific_bridge(contract),
        "bridge_assumption_missing",
        "scientific_bridge.assumption_missing",
    )
    anonymous = surrogate_contract(
        assumptions=(sb.Assumption("validation", "surrogate_validation", "validated"),)
    )
    refused(
        lambda: sb.prepare_scientific_bridge(anonymous),
        "bridge_assumption_missing",
        "scientific_bridge.assumption_evidence_missing",
    )


def test_value_kinds_are_not_interchangeable():
    held_and_requested = [(a, b) for a in sb.VALUE_KINDS for b in sb.VALUE_KINDS if a != b]
    assert len(held_and_requested) == 20
    for held, requested in held_and_requested:
        contract = surrogate_contract(source_value=held, target_value=requested)
        error = refused(
            lambda contract=contract: sb.prepare_scientific_bridge(contract),
            "bridge_substitution_refused",
            "scientific_bridge.value_kind_substitution",
        )
        assert held in error.message and requested in error.message
    # A law held and required as itself has no bridge route.
    for kind in ("predictive_law", "posterior", "identified_set", "joint_outcome_law"):
        contract = surrogate_contract(source_value=kind, target_value=kind)
        refused(
            lambda contract=contract: sb.prepare_scientific_bridge(contract),
            "cell_not_licensed",
            "scientific_bridge.value_kind_not_carried",
        )
    # A consumer that requires a law from a bridged mean is refused.
    claim = require_success(
        sb.prepare_scientific_bridge(surrogate_contract()).derive(surrogate_source())
    )
    claim.require_value_kind("mean")
    for needed in ("predictive_law", "posterior", "identified_set", "joint_outcome_law"):
        refused(
            lambda needed=needed: claim.require_value_kind(needed),
            "bridge_substitution_refused",
            "scientific_bridge.value_kind_substitution",
        )


def test_population_regime_and_the_claimed_quantity_must_agree():
    contract = surrogate_contract()
    bridge = sb.prepare_scientific_bridge(contract)

    other_population = surrogate_contract(
        target=outcome("trial:survival", units="log_hazard", population="children")
    )
    refused(
        lambda: sb.prepare_scientific_bridge(other_population),
        "bridge_binding_mismatch",
        "scientific_bridge.population_mismatch",
    )
    other_regime = surrogate_contract(
        target=outcome("trial:survival", units="log_hazard", regime="do(t=2)")
    )
    refused(
        lambda: sb.prepare_scientific_bridge(other_regime),
        "bridge_binding_mismatch",
        "scientific_bridge.regime_mismatch",
    )
    wrong = sb.SourceClaim.declared(
        "another-variable",
        0.8,
        outcome("trial:other", units="mmol_per_l"),
    )
    refused(
        lambda: require_success(bridge.derive(wrong)),
        "bridge_binding_mismatch",
        "scientific_bridge.quantity_mismatch",
    )
    other_units = sb.SourceClaim(
        claim_id=surrogate_source().claim_id,
        value=0.8,
        quantity=contract.source,
        units="mg_per_dl",
        identities=surrogate_source().identities,
    )
    refused(
        lambda: require_success(bridge.derive(other_units)),
        "bridge_binding_mismatch",
        "scientific_bridge.units_mismatch",
    )
    law = sb.SourceClaim.declared("a-law", None, contract.source, kind="mixture")
    refused(
        lambda: require_success(bridge.derive(law)),
        "cell_not_licensed",
        "scientific_bridge.source_not_point",
    )


def test_a_transported_bridge_names_its_premise_or_is_refused():
    children = surrogate_contract(population="children")
    error = refused(
        lambda: sb.prepare_scientific_bridge(children),
        "bridge_assumption_missing",
        "scientific_bridge.transport_premise_missing",
    )
    assert "adults" in error.message and "children" in error.message

    premise = sb.TransportabilityPremise(
        "same-surrogate-relationship",
        "the trial-level relationship is the same in children",
        established_in=sb.Scope("adults", "do(t=1)"),
        applies_to=sb.Scope("children", "do(t=1)"),
        evidence="selection-diagram:pediatric",
    )
    licensed = surrogate_contract(population="children", transport=premise)
    bridge = sb.prepare_scientific_bridge(licensed)
    assert bridge.transported
    claim = require_success(bridge.derive(surrogate_source(0.8, population="children")))
    assert claim.transported and claim.value == pytest.approx(0.796, abs=1e-12)
    assert claim.transportability_premise == "same-surrogate-relationship"
    assert "transportability:same-surrogate-relationship" in claim.unresolved
    # A premise for another transport is not this one.
    wrong = sb.TransportabilityPremise(
        "other",
        "elsewhere",
        established_in=sb.Scope("adults", "do(t=1)"),
        applies_to=sb.Scope("infants", "do(t=1)"),
    )
    refused(
        lambda: sb.prepare_scientific_bridge(
            surrogate_contract(population="children", transport=wrong)
        ),
        "bridge_binding_mismatch",
        "scientific_bridge.transport_premise_mismatch",
    )


def test_validation_minimums_and_the_validated_range_are_enforced():
    source_quantity = outcome("trial:biomarker", units="mmol_per_l")
    refused(
        lambda: sb.prepare_scientific_bridge(surrogate_contract(min_trials=5)),
        "bridge_mapping_not_licensed",
        "scientific_bridge.validation_trials_too_few",
    )
    refused(
        lambda: sb.prepare_scientific_bridge(surrogate_contract(min_r_squared=0.995)),
        "bridge_mapping_not_licensed",
        "scientific_bridge.validation_r_squared_too_low",
    )
    bridge = sb.prepare_scientific_bridge(surrogate_contract())
    for inside, expected in ((0.0, 0.06), (1.5, 1.44)):
        value = require_success(
            bridge.derive(sb.SourceClaim.declared(f"edge-{inside}", inside, source_quantity))
        ).value
        assert value == pytest.approx(expected, abs=1e-12)
    for beyond in (-0.01, 1.51, 3.0):
        refused(
            lambda beyond=beyond: require_success(
                bridge.derive(sb.SourceClaim.declared(f"beyond-{beyond}", beyond, source_quantity))
            ),
            "bridge_mapping_not_licensed",
            "scientific_bridge.outside_validated_range",
        )
    # Recorded prevalences outside [1 - Sp, Se] cannot arise from any latent prevalence.
    correction = sb.prepare_scientific_bridge(correction_contract())
    for impossible in (0.05, 0.95):
        refused(
            lambda impossible=impossible: require_success(
                correction.derive(
                    sb.SourceClaim.declared(
                        f"p-{impossible}", impossible, prevalence("schema:flag")
                    )
                )
            ),
            "bridge_mapping_not_licensed",
            "scientific_bridge.outside_attainable_range",
        )
    refused(
        lambda: sb.prepare_scientific_bridge(correction_contract(clinic_model(0.55, 0.5, 0.1))),
        "measurement_not_identified",
        "scientific_bridge.youden_too_small",
    )


def test_malformed_declarations_are_input_errors_before_native_code():
    with pytest.raises(CausalValueError):
        sb.AffineMapping(float("nan"), 1.0)
    with pytest.raises(CausalValueError):
        sb.BridgeContract(
            kind="not_a_bridge",
            source=None,
            source_class="surrogate_endpoint",
            target=None,
            target_class="terminal_outcome",
            evidence_scope=sb.Scope("a", "b"),
            mapping=sb.IdentityMapping(),
        )
    with pytest.raises(CausalTypeError):
        sb.prepare_scientific_bridge("not a contract")
    with pytest.raises(CausalTypeError):
        require_success(sb.prepare_scientific_bridge(surrogate_contract()).derive("not a claim"))
    with pytest.raises(CausalValueError):
        sb.Assumption("x", "not_a_kind", "statement")


# --------------------------------------------------------------------------- lifecycle


def test_a_refused_derivation_or_replacement_leaves_the_prepared_bridge_unchanged():
    bridge = sb.prepare_scientific_bridge(surrogate_contract())
    identity = bridge.contract_identity
    before = require_success(bridge.derive(surrogate_source(0.8)))

    # A refreshed source about the same quantity recomputes under the same contract.
    refreshed = require_success(bridge.derive(surrogate_source(1.0, label="refreshed")))
    assert refreshed.contract_identity == identity
    assert refreshed.value == pytest.approx(0.06 + 0.92 * 1.0, abs=1e-12)

    refused(
        lambda: require_success(bridge.derive(surrogate_source(2.0, label="beyond"))),
        "bridge_mapping_not_licensed",
        "scientific_bridge.outside_validated_range",
    )
    refused(
        lambda: bridge.with_contract(surrogate_contract(min_r_squared=0.995)),
        "bridge_mapping_not_licensed",
        "scientific_bridge.validation_r_squared_too_low",
    )
    assert bridge.contract_identity == identity
    after = require_success(bridge.derive(surrogate_source(0.8)))
    assert (after.claim_id, after.value) == (before.claim_id, before.value)

    # A licensed replacement is a different bridge with a different identity.
    replaced = bridge.with_contract(surrogate_contract(min_r_squared=0.95))
    assert replaced.contract_identity != identity


def test_a_withdrawn_or_superseded_source_makes_the_bridged_claim_inapplicable():
    claim = require_success(
        sb.prepare_scientific_bridge(surrogate_contract()).derive(surrogate_source())
    )
    current = require_success(claim.freshness())
    assert current.historically_valid and current.currently_applicable is True
    for kwargs in ({"source_withdrawn": True}, {"source_superseded": True}):
        standing = require_success(claim.freshness(**kwargs))
        assert standing.currently_applicable is False
        assert standing.historically_valid, "history is preserved, only applicability changes"


def test_no_interval_posterior_or_joint_law_is_licensed():
    claim = require_success(
        sb.prepare_scientific_bridge(surrogate_contract()).derive(surrogate_source())
    )
    error = refused(claim.interval, "cell_not_licensed", "scientific_bridge.interval_not_licensed")
    assert error.stage == "scientific_bridge"
    consumed = require_success(sb.consume_scientific_bridge_artifact(bytes(claim.export("closed"))))
    refused(consumed.interval, "cell_not_licensed", "scientific_bridge.interval_not_licensed")
    refused(
        lambda: consumed.claim.require_value_kind("posterior"),
        "bridge_substitution_refused",
        "scientific_bridge.value_kind_substitution",
    )


def test_a_scoped_interval_needs_the_source_standard_error_and_a_measured_design():
    import math

    contract = surrogate_contract()
    quantity = outcome("trial:biomarker", units="mmol_per_l")
    # No source standard error: refused naming it.
    claim = require_success(sb.prepare_scientific_bridge(contract).derive(surrogate_source()))
    refused(
        lambda: claim.interval(0.95),
        "cell_not_licensed",
        "scientific_bridge.interval_source_uncertainty_required",
    )
    # Four validation trials are below the measured 15 to 40: refused with the typed reason.
    claim = require_success(
        sb.prepare_scientific_bridge(contract).derive(
            sb.SourceClaim.declared("biomarker-effect", 0.8, quantity, standard_error=0.1)
        )
    )
    refused(
        lambda: claim.interval(0.95),
        "cell_not_licensed",
        "scientific_bridge.interval_scope_outside_measured",
    )
    # Twenty validation trials, source standard error declared: published, bound to a record.
    trials = [
        (0.5 + 1.5 * i / 19.0, 0.1 + 0.9 * (0.5 + 1.5 * i / 19.0) + 0.1 * math.sin(i))
        for i in range(20)
    ]
    wide = surrogate_contract(
        mapping=sb.TrialLevelMapping(trials=trials, min_trials=4, min_r_squared=0.5)
    )
    claim = require_success(
        sb.prepare_scientific_bridge(wide).derive(
            sb.SourceClaim.declared("biomarker-effect", 1.2, quantity, standard_error=0.1)
        )
    )
    interval = claim.interval(0.95)
    assert interval.lower < interval.value < interval.upper
    assert interval.licensed_by == "coverage_record"
    assert interval.coverage_record.startswith("cov.bridged_claim_interval.")
    assert interval.degrees_of_freedom == 18.0
    # An unscoped request is still refused.
    refused(claim.interval, "cell_not_licensed", "scientific_bridge.interval_not_licensed")


# --------------------------------------------------------------------------- artifact


def test_the_artifact_replays_independently_and_refuses_corruption_and_a_wrong_identity():
    claim = require_success(
        sb.prepare_scientific_bridge(surrogate_contract()).derive(surrogate_source())
    )
    exported = claim.export("surrogate-bridge")

    consumed = require_success(
        sb.consume_scientific_bridge_artifact(exported.data, exported.expectation.identity)
    )
    assert consumed.identity == exported.expectation.identity
    assert consumed.claim.value == claim.value
    assert consumed.claim.claim_id == claim.claim_id
    assert consumed.claim.contract_identity == claim.contract_identity
    assert consumed.claim.fit == claim.fit
    # The replayed source is reduced and never claims what it cannot show.
    assert consumed.source["claim_id"] == claim.source_claim_id
    assert consumed.source["value"] == 0.8
    assert consumed.source["program_identity"] is None
    assert consumed.source["identification_domain"] == "unknown"
    assert consumed.source["support_domain"] == "unknown"

    # Another identity is another artifact.
    refused(
        lambda: require_success(sb.consume_scientific_bridge_artifact(exported.data, "0" * 64)),
        "route_not_supported",
        "scientific_bridge.identity_mismatch",
    )
    other = require_success(
        sb.prepare_scientific_bridge(surrogate_contract()).derive(
            surrogate_source(1.0, label="another")
        )
    )
    assert other.export("another").expectation.identity != exported.expectation.identity

    # A flipped byte anywhere in the body, and truncation, are refused.
    for position in (
        len(exported.data) // 3,
        len(exported.data) // 2,
        len(exported.data) - 8,
    ):
        damaged = bytearray(exported.data)
        damaged[position] ^= 0xFF
        with pytest.raises(ScientificBridgeRefusal) as caught:
            require_success(
                sb.consume_scientific_bridge_artifact(bytes(damaged), exported.expectation.identity)
            )
        assert_registered_refusal(caught.value)
    with pytest.raises(ScientificBridgeRefusal):
        require_success(
            sb.consume_scientific_bridge_artifact(exported.data[: len(exported.data) // 3])
        )
    with pytest.raises(CausalTypeError):
        require_success(sb.consume_scientific_bridge_artifact("not bytes"))


def test_a_measurement_correction_artifact_carries_its_model_and_replays():
    bridge = sb.prepare_scientific_bridge(correction_contract())
    claim = require_success(
        bridge.derive(sb.SourceClaim.declared("flag", 0.31, prevalence("schema:flag")))
    )
    exported = claim.export("flag-bridge")
    consumed = require_success(
        sb.consume_scientific_bridge_artifact(exported.data, exported.expectation.identity)
    )
    assert consumed.claim.value == pytest.approx(11 / 70, abs=1e-12)
    assert consumed.claim.mapping == "measurement_correction"
    assert consumed.claim.fit is None
    assert consumed.claim.assumptions[0].evidence == mm.model_identities(clinic_model()).model


def test_a_fresh_process_consumes_the_artifact_against_the_retained_identity(tmp_path):
    claim = require_success(
        sb.prepare_scientific_bridge(surrogate_contract()).derive(surrogate_source())
    )
    exported = claim.export("fresh")
    path = tmp_path / "bridge.bin"
    path.write_bytes(exported.data)
    script = (
        "import json, sys\n"
        "from antecedent import scientific_bridge as sb\n"
        "c = sb.consume_scientific_bridge_artifact(open(sys.argv[1], 'rb').read(), sys.argv[2])\n"
        "print(json.dumps({'value': c.claim.value, 'claim_id': c.claim.claim_id,\n"
        "                  'slope': c.claim.fit.slope, 'identity': c.identity,\n"
        "                  'transported': c.claim.transported}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), exported.expectation.identity],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    wire = json.loads(done.stdout)
    assert wire["value"] == pytest.approx(0.796, abs=1e-12)
    assert wire["slope"] == pytest.approx(0.92, abs=1e-12)
    assert wire["claim_id"] == claim.claim_id
    assert wire["identity"] == exported.expectation.identity
    assert wire["transported"] is False
    refused_run = subprocess.run(
        [sys.executable, "-c", script, str(path), "0" * 64],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert refused_run.returncode != 0
    assert "scientific_bridge.identity_mismatch" in refused_run.stderr
