"""Observation-conditioned queries through the Python facade.

Independent oracles in this file are brute-force enumerations and Bayes' rule written in pure
Python; no value is read from native code to decide what the truth is.

Confounded finite model (binary ``Z``, ``S``, ``Y``): ``P(Z=1)=0.5``, ``P(S=1|Z)=0.2, 0.8``,
``P(Y=1|S,Z)=0.1, 0.5, 0.3, 0.9`` keyed ``(S=0,Z=0), (0,1), (1,0), (1,1)``; hence
``P(Y=1|S=1)=0.78`` against ``P(Y=1|do(S=1))=0.6``.

Measured construct: prevalence 0.1, ``P(Y=1|C)=0.1, 0.7``, sensitivity 0.9, specificity 0.8;
``P(C=1|M=1)=1/3`` and ``P(Y=1|M=1)=0.3`` against ``P(Y=1|C=1)=0.7``.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys

import pytest
from antecedent import measurement_models as mm
from antecedent import mechanism_composition as mc
from antecedent import observation_query as oq
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.observation_query import ObservationQueryRefusal

from _refusal import assert_registered_refusal

TOL = 1e-12
ROLES = {"S": "treatment", "Y": "outcome"}


# --------------------------------------------------------------------------- fixtures


def quantity(variable: str) -> mc.Quantity:
    return mc.Quantity(
        variable,
        units="category",
        population_id="pop",
        role=ROLES.get(variable, "covariate"),
        variable_name=f"label of {variable}",
    )


def table(variable, parents, rows) -> mc.Mechanism:
    declaration = mc.MechanismDeclaration(
        output=quantity(variable),
        parents=tuple(quantity(p) for p in parents),
        artifact_id=f"declared:{variable}",
        evidence_id=f"assumed:{variable}",
    )
    return mc.Mechanism.declared_table(
        declaration,
        levels=["0", "1"],
        rows=[
            (dict(zip(parents, config, strict=True)), {"0": p0, "1": p1})
            for config, (p0, p1) in rows
        ],
    )


# Conditional tables of the confounded model, shared by the model and the oracle.
P_Z1 = 0.5
P_S1_GIVEN_Z = {"0": 0.2, "1": 0.8}
P_Y1 = {("0", "0"): 0.1, ("0", "1"): 0.5, ("1", "0"): 0.3, ("1", "1"): 0.9}  # (s, z)


def confounded(p_s1_given_z=None, p_y1=None) -> mc.PreparedMechanismComposition:
    p_s = p_s1_given_z or P_S1_GIVEN_Z
    p_y = p_y1 or P_Y1
    return mc.compose_mechanisms(
        [
            table("Z", [], [((), (1 - P_Z1, P_Z1))]),
            table("S", ["Z"], [((z,), (1 - p_s[z], p_s[z])) for z in "01"]),
            table(
                "Y",
                ["S", "Z"],
                [((s, z), (1 - p_y[(s, z)], p_y[(s, z)])) for s in "01" for z in "01"],
            ),
        ],
        target_population_id="pop",
        admit_declared_mechanisms=True,
    )


def enumerate_confounded(p_s1_given_z=None, p_y1=None, do_s=None, given_s=None):
    """``P(Y=1)`` by brute force over (z, s, y), optionally under ``do(S)`` or given ``S``."""
    p_s = p_s1_given_z or P_S1_GIVEN_Z
    p_y = p_y1 or P_Y1
    numerator = evidence = 0.0
    for z, s, y in itertools.product("01", repeat=3):
        pz = P_Z1 if z == "1" else 1 - P_Z1
        ps = p_s[z] if s == "1" else 1 - p_s[z]
        if do_s is not None:
            ps = 1.0 if s == do_s else 0.0
        py = p_y[(s, z)] if y == "1" else 1 - p_y[(s, z)]
        joint = pz * ps * py
        if given_s is not None and s != given_s:
            continue
        if y == "1":
            # P(S=s, Z=z) is counted once per (z, s), not once per outcome level.
            evidence += pz * ps
            numerator += joint
    return numerator / evidence if given_s is not None else numerator


PREVALENCE = 0.1
P_Y1_GIVEN_C = {"0": 0.1, "1": 0.7}
SENSITIVITY, SPECIFICITY = 0.9, 0.8


def construct_model() -> mc.PreparedMechanismComposition:
    return mc.compose_mechanisms(
        [
            table("C", [], [((), (1 - PREVALENCE, PREVALENCE))]),
            table("Y", ["C"], [((c,), (1 - P_Y1_GIVEN_C[c], P_Y1_GIVEN_C[c])) for c in "01"]),
        ],
        target_population_id="pop",
        admit_declared_mechanisms=True,
    )


def assay(population: str = "pop") -> mm.MeasurementModel:
    return mm.MeasurementModel(
        indicator=mm.MeasurementVariable("test_result", "noisy_misclassified", "category"),
        construct=mm.MeasurementVariable("C", "latent_scientific", "category"),
        parameters=mm.BinaryParameters((mm.BinaryRates(SENSITIVITY, SPECIFICITY),), min_youden=0.1),
        population_id=population,
        regime_id="observational",
        evidence=mm.ParameterEvidence.declared("lab"),
    )


def bayes_measured(recorded: str):
    """Pure-Python Bayes inversion: ``(P(M), P(C=1|M), P(Y=1|M))``."""
    likelihood = {
        ("1", "1"): SENSITIVITY,
        ("0", "1"): 1 - SENSITIVITY,
        ("1", "0"): 1 - SPECIFICITY,
        ("0", "0"): SPECIFICITY,
    }
    weight = {
        c: likelihood[(recorded, c)] * (PREVALENCE if c == "1" else 1 - PREVALENCE) for c in "01"
    }
    evidence = sum(weight.values())
    posterior = weight["1"] / evidence
    outcome = sum(weight[c] / evidence * P_Y1_GIVEN_C[c] for c in "01")
    return evidence, posterior, outcome


# --------------------------------------------------------------------------- conditioning


def test_conditioning_and_intervention_disagree_and_both_match_the_enumeration():
    model = confounded()
    paired = oq.condition_versus_intervene(model, "Y", {"S": "1"})
    assert paired.operation == "conditioning_and_intervention"
    conditioning = enumerate_confounded(given_s="1")
    intervention = enumerate_confounded(do_s="1")
    assert paired.paired.conditioning["1"] == pytest.approx(conditioning, abs=TOL)
    assert paired.paired.conditioning["1"] == pytest.approx(0.78, abs=TOL)
    assert paired.paired.intervention["1"] == pytest.approx(intervention, abs=TOL)
    assert paired.paired.intervention["1"] == pytest.approx(0.6, abs=TOL)
    assert paired.paired.max_abs_difference == pytest.approx(0.18, abs=TOL)
    assert paired.paired.evidence_probability == pytest.approx(0.5, abs=TOL)
    assert paired.inference_basis == "exact_given_declared_model"


def test_conditioning_intervention_and_the_pair_have_three_distinct_identities():
    model = confounded()
    conditioning = oq.condition_on(model, "Y", {"S": "1"})
    intervention = oq.intervene_on(model, "Y", {"S": "1"})
    paired = oq.condition_versus_intervene(model, "Y", {"S": "1"})
    assert conditioning.distribution["1"] == pytest.approx(0.78, abs=TOL)
    assert intervention.distribution["1"] == pytest.approx(0.6, abs=TOL)
    assert len({conditioning.identity, intervention.identity, paired.identity}) == 3
    assert (conditioning.operation, intervention.operation) == ("conditioning", "intervention")
    # A conditioning claim carries no interventional value and the reverse.
    assert conditioning.paired is None and intervention.paired is None
    assert intervention.evidence_probability is None


def test_without_a_common_cause_the_two_laws_agree():
    flat = confounded(p_s1_given_z={"0": 0.5, "1": 0.5})
    paired = oq.condition_versus_intervene(flat, "Y", {"S": "1"})
    expected = enumerate_confounded({"0": 0.5, "1": 0.5}, do_s="1")
    assert paired.paired.conditioning["1"] == pytest.approx(expected, abs=TOL)
    assert paired.paired.intervention["1"] == pytest.approx(expected, abs=TOL)
    assert paired.paired.max_abs_difference == pytest.approx(0.0, abs=TOL)


def test_conditioning_on_several_values_and_on_a_mapping_or_slots_agree():
    model = confounded()
    by_mapping = oq.condition_on(model, "Y", {"S": "1", "Z": "1"})
    by_slots = oq.condition_on(model, "Y", [oq.Recorded("Z", "1"), oq.Recorded("S", "1")])
    assert by_mapping.distribution["1"] == pytest.approx(0.9, abs=TOL)
    assert by_mapping.evidence_probability == pytest.approx(0.4, abs=TOL)
    assert by_slots.identity == by_mapping.identity


def test_an_unknown_variable_and_null_evidence_are_refused():
    model = confounded()
    with pytest.raises(ObservationQueryRefusal) as unknown:
        oq.condition_on(model, "Y", {"Q": "1"})
    assert unknown.value.reason_code == "unknown_variable"
    assert unknown.value.detail == "observation_query.unknown_variable"
    with pytest.raises(ObservationQueryRefusal) as outcome:
        oq.condition_on(model, "Y", {"Y": "1"})
    assert outcome.value.detail == "observation_query.outcome_assigned"


# --------------------------------------------------------------------------- smuggling


def test_an_intervention_smuggled_into_a_conditioning_slot_is_refused_with_a_remedy():
    model = confounded()
    for slot in (oq.Assigned("S", "1"), oq.Hypothetical("S", "1")):
        with pytest.raises(ObservationQueryRefusal) as caught:
            oq.condition_on(model, "Y", [slot])
        error = caught.value
        assert isinstance(error, CausalUnsupportedError)
        assert_registered_refusal(error)
        assert error.reason_code == "quantity_semantics_mismatch"
        assert error.detail == "observation_query.intervention_in_conditioning_slot"
        assert error.offending == "S"
        assert error.remedy
    # Stated honestly, as an intervention, the same value is answered.
    assert oq.intervene_on(model, "Y", {"S": "1"}).distribution["1"] == pytest.approx(0.6, abs=TOL)


# --------------------------------------------------------------------------- measured


def test_a_measured_value_is_inverted_through_its_model_and_matches_bayes():
    model = construct_model()
    for recorded in ("1", "0"):
        evidence, posterior, outcome = bayes_measured(recorded)
        claim = oq.condition_on(
            model,
            "Y",
            [oq.Measured("C", recorded, "test_result", model=assay(), positive_level="1")],
        )
        assert claim.evidence_probability == pytest.approx(evidence, abs=TOL)
        assert claim.construct_posterior["1"] == pytest.approx(posterior, abs=TOL)
        assert claim.distribution["1"] == pytest.approx(outcome, abs=TOL)
    positive = bayes_measured("1")
    assert positive[1] == pytest.approx(1 / 3, abs=TOL)
    assert positive[2] == pytest.approx(0.3, abs=TOL)
    # Conditioning on the construct itself is a different claim and a different value.
    direct = oq.condition_on(model, "Y", {"C": "1"})
    assert direct.distribution["1"] == pytest.approx(0.7, abs=TOL)


def test_a_measured_value_used_as_the_construct_is_refused():
    model = construct_model()
    with pytest.raises(ObservationQueryRefusal) as caught:
        oq.condition_on(model, "Y", [oq.Measured("C", "1", "test_result", model=None)])
    assert caught.value.reason_code == "proxy_offered_as_construct"
    assert caught.value.detail == "observation_query.measured_as_construct"
    assert caught.value.remedy
    assert_registered_refusal(caught.value)


def test_a_measurement_model_for_another_population_is_not_applied():
    model = construct_model()
    with pytest.raises(ObservationQueryRefusal) as caught:
        oq.condition_on(
            model,
            "Y",
            [oq.Measured("C", "1", "test_result", model=assay("other_clinic"), positive_level="1")],
        )
    assert caught.value.reason_code == "measurement_binding_mismatch"
    assert caught.value.detail == "observation_query.population_mismatch"
    # `expected` is what the model was established for, `supplied` what the query is about.
    assert caught.value.expected == "other_clinic"
    assert caught.value.supplied == "pop"


# --------------------------------------------------------------------------- thresholds


def test_threshold_crossing_lists_the_crossing_observations():
    model = confounded()
    claim = oq.threshold_crossing(model, "Y", "1", "S", threshold=0.5)
    rows = {r.level: r for r in claim.threshold_rows}
    assert rows["1"].conditional == pytest.approx(enumerate_confounded(given_s="1"), abs=TOL)
    assert rows["1"].crosses and not rows["0"].crosses
    assert claim.crossing_probability == pytest.approx(0.5, abs=TOL)
    assert rows["0"].conditional == pytest.approx(enumerate_confounded(given_s="0"), abs=TOL)


def test_a_threshold_no_observation_reaches_is_refused_with_the_supremum():
    model = confounded()
    with pytest.raises(ObservationQueryRefusal) as caught:
        oq.threshold_crossing(model, "Y", "1", "S", threshold=0.9)
    assert caught.value.reason_code == "evidence_cannot_reach_level"
    assert caught.value.detail == "observation_query.level_unreachable"
    assert float(caught.value.supplied) == pytest.approx(0.78, abs=TOL)
    assert float(caught.value.expected) == pytest.approx(0.9, abs=TOL)
    assert caught.value.remedy


# --------------------------------------------------------------------------- Bayes updates


def test_finite_posterior_matches_pure_python_bayes_and_ignores_order():
    prior = {"ill": 0.01, "well": 0.99}
    like = {"ill": {"neg": 0.1, "pos": 0.9}, "well": {"neg": 0.95, "pos": 0.05}}

    def bayes(obs):
        weight = {h: prior[h] for h in prior}
        for o in obs:
            for h in weight:
                weight[h] *= like[h][o]
        total = sum(weight.values())
        return {h: w / total for h, w in weight.items()}, total

    args = (["ill", "well"], [0.01, 0.99], ["neg", "pos"], [[0.1, 0.9], [0.95, 0.05]])
    claim = oq.finite_posterior(*args, ["pos", "neg", "pos"])
    expected, evidence = bayes(["pos", "neg", "pos"])
    assert claim.posterior["ill"] == pytest.approx(expected["ill"], abs=TOL)
    assert claim.evidence_probability == pytest.approx(evidence, abs=TOL)
    assert claim.posterior == oq.finite_posterior(*args, ["neg", "pos", "pos"]).posterior
    assert claim.operation == "posterior_update"
    assert claim.inference_basis == "exact_given_declared_model"
    assert sum(claim.predictive.values()) == pytest.approx(1.0, abs=TOL)
    fitted = oq.finite_posterior(*args, ["pos"], basis="sample_fitted")
    assert fitted.inference_basis == "point_only_sample_fitted"
    assert fitted.identity != oq.finite_posterior(*args, ["pos"]).identity


def test_impossible_evidence_and_malformed_models_are_refused():
    with pytest.raises(ObservationQueryRefusal) as impossible:
        oq.finite_posterior(["a", "b"], [0.5, 0.5], ["x", "y"], [[1.0, 0.0], [1.0, 0.0]], ["y"])
    assert impossible.value.detail == "observation_query.zero_probability_evidence"
    with pytest.raises(ObservationQueryRefusal) as malformed:
        oq.finite_posterior(["a", "b"], [0.5, 0.4], ["x", "y"], [[0.5, 0.5], [0.5, 0.5]], ["x"])
    assert malformed.value.detail == "observation_query.invalid_declaration"


def test_conjugate_posterior_is_the_closed_form():
    claim = oq.conjugate_posterior(1.0, 1.0, successes=7, failures=3)
    alpha, beta, predictive = claim.beta_posterior
    assert (alpha, beta) == (8.0, 4.0)
    assert predictive == pytest.approx(8 / 12, abs=TOL)


def test_required_evidence_matches_closed_forms():
    factor = oq.required_bayes_factor(0.2, 0.8)
    assert factor.required_factor == pytest.approx((0.8 / 0.2) / (0.2 / 0.8), abs=1e-10)
    # prior 0.01, factor 0.9/0.05 = 18, level 0.5: smallest n with odds0 * 18^n >= 1.
    claim = oq.required_observations_binary_signal(
        0.01, 0.5, rate_given_hypothesis=0.9, rate_given_complement=0.05
    )
    odds0 = 0.01 / 0.99
    n = next(k for k in itertools.count(0) if odds0 * 18.0**k >= 1.0)
    assert claim.required_observations == n == 2
    # Beta(1,1): (1 + n) / (2 + n) >= 0.9 first holds at n = 8.
    conjugate = oq.required_observations_conjugate(1.0, 1.0, 0.9)
    assert conjugate.required_observations == 8
    assert conjugate.value_at_requirement == pytest.approx(0.9, abs=TOL)


def test_an_unreachable_evidence_level_states_the_supremum():
    with pytest.raises(ObservationQueryRefusal) as flat:
        oq.required_observations_binary_signal(
            0.3, 0.9, rate_given_hypothesis=0.5, rate_given_complement=0.5
        )
    assert flat.value.reason_code == "evidence_cannot_reach_level"
    assert float(flat.value.supplied) == pytest.approx(0.3, abs=TOL)
    with pytest.raises(ObservationQueryRefusal) as budget:
        oq.required_observations_conjugate(1.0, 1.0, 0.99, max_observations=50)
    assert float(budget.value.supplied) == pytest.approx(51 / 52, abs=TOL)


def test_input_errors_are_typed_not_refusals():
    with pytest.raises(CausalValueError):
        oq.required_bayes_factor(0.2, 0.8, direction="sideways")
    with pytest.raises(CausalTypeError):
        oq.conjugate_posterior(1.0, 1.0, successes=-1, failures=0)
    with pytest.raises(CausalValueError):
        oq.finite_posterior(["a"], [1.0], ["x"], [[1.0]], ["x"], basis="guess")
    with pytest.raises(CausalTypeError):
        oq.condition_on(object(), "Y", {"S": "1"})


# --------------------------------------------------------------------------- closed routes


def test_interval_and_calibrated_probability_are_closed():
    claim = oq.condition_on(confounded(), "Y", {"S": "1"})
    with pytest.raises(ObservationQueryRefusal) as interval:
        claim.interval()
    assert interval.value.reason_code == "cell_not_licensed"
    assert interval.value.detail == "observation_query.interval_route_closed"
    with pytest.raises(ObservationQueryRefusal) as calibrated:
        claim.calibrated_probability()
    assert calibrated.value.reason_code == "cell_not_licensed"
    assert calibrated.value.detail == "observation_query.calibrated_probability_closed"
    fitted = oq.conjugate_posterior(1.0, 1.0, successes=1, failures=1, basis="sample_fitted")
    with pytest.raises(ObservationQueryRefusal):
        fitted.calibrated_probability()


# --------------------------------------------------------------------------- artifact


def test_the_artifact_round_trips_and_replays_against_the_retained_identity():
    claim = oq.condition_versus_intervene(confounded(), "Y", {"S": "1"})
    consumed = oq.consume_observation_query_artifact(claim.artifact, claim.identity)
    assert consumed.identity == claim.identity
    assert consumed.operation == "conditioning_and_intervention"
    assert consumed.claim.paired.conditioning["1"] == pytest.approx(0.78, abs=TOL)
    assert consumed.claim.paired.intervention["1"] == pytest.approx(0.6, abs=TOL)


def test_semantic_mutation_and_another_history_are_refused():
    claim = oq.condition_versus_intervene(confounded(), "Y", {"S": "1"})
    with pytest.raises(ObservationQueryRefusal) as wrong:
        oq.consume_observation_query_artifact(claim.artifact, "f" * 64)
    assert wrong.value.detail == "observation_query.artifact_refused"
    # A consistent artifact of another model fails the identity the consumer retained.
    other = oq.condition_versus_intervene(
        confounded(p_s1_given_z={"0": 0.3, "1": 0.7}), "Y", {"S": "1"}
    )
    assert other.identity != claim.identity
    assert oq.consume_observation_query_artifact(other.artifact).identity == other.identity
    with pytest.raises(ObservationQueryRefusal):
        oq.consume_observation_query_artifact(other.artifact, claim.identity)
    # Every sampled single-byte change is refused: corruption, identity or replay.
    data = bytes(claim.artifact)
    for position in range(0, len(data), max(1, len(data) // 25)):
        mutated = bytearray(data)
        mutated[position] ^= 0x01
        with pytest.raises(ObservationQueryRefusal):
            oq.consume_observation_query_artifact(bytes(mutated), claim.identity)
    with pytest.raises(ObservationQueryRefusal) as truncated:
        oq.consume_observation_query_artifact(data[: len(data) // 2])
    assert_registered_refusal(truncated.value)
    with pytest.raises(CausalTypeError):
        oq.consume_observation_query_artifact("not bytes")  # type: ignore[arg-type]


def test_a_fresh_process_consumes_the_artifact_against_the_retained_identity(tmp_path):
    claim = oq.condition_versus_intervene(confounded(), "Y", {"S": "1"})
    path = tmp_path / "claim.bin"
    path.write_bytes(claim.artifact)
    script = (
        "import json, sys\n"
        "from antecedent import observation_query as oq\n"
        "c = oq.consume_observation_query_artifact(open(sys.argv[1], 'rb').read(), sys.argv[2])\n"
        "p = c.claim.paired\n"
        "print(json.dumps({'conditioning': p.conditioning['1'], 'intervention': "
        "p.intervention['1'], 'operation': c.operation}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), claim.identity],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    wire = json.loads(done.stdout)
    assert wire["conditioning"] == pytest.approx(0.78, abs=TOL)
    assert wire["intervention"] == pytest.approx(0.6, abs=TOL)
    assert wire["operation"] == "conditioning_and_intervention"
    refused = subprocess.run(
        [sys.executable, "-c", script, str(path), "f" * 64],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert refused.returncode != 0
    assert "artifact_refused" in refused.stderr


# --------------------------------------------------------------------------- lifecycle


def test_a_refreshed_composition_gives_a_new_claim_and_leaves_the_earlier_one():
    model = confounded()
    before = oq.condition_on(model, "Y", {"S": "1"})
    refresh = mc.MechanismRefit(
        variable_id="S",
        source={
            "kind": "declared_table",
            "levels": ["0", "1"],
            "rows": [
                {
                    "parent_levels": [["Z", z]],
                    "probabilities": [["0", 0.5], ["1", 0.5]],
                }
                for z in "01"
            ],
        },
        artifact_id="declared:S:v2",
        evidence_id="assumed:S:v2",
    )
    refreshed = model.refresh_mechanisms([refresh])
    after = oq.condition_on(refreshed, "Y", {"S": "1"})
    assert after.identity != before.identity
    assert before.distribution["1"] == pytest.approx(0.78, abs=TOL)
    flat = enumerate_confounded({"0": 0.5, "1": 0.5}, given_s="1")
    assert after.distribution["1"] == pytest.approx(flat, abs=TOL)
    pair = oq.condition_versus_intervene(refreshed, "Y", {"S": "1"})
    assert pair.paired.max_abs_difference == pytest.approx(0.0, abs=TOL)
