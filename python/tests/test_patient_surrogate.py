"""Patient principal-surrogate identification through the typed compiler and bundle."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest
from antecedent import _reason_details as reason_details
from antecedent.errors import CausalTypeError
from antecedent.outcomes import require_success
from antecedent.patient_surrogate import (
    ObservedSurrogatePatient,
    PatientSurrogateDesign,
    PatientSurrogateRefusal,
    PopulationJointSurrogateLaw,
    SurrogatePatientSample,
    compile_patient_surrogate,
)

DESIGN = PatientSurrogateDesign(
    trial_id="trial-001",
    population="adults",
    surrogate="early",
    outcome="terminal_benefit",
    surrogate_visit=1,
    outcome_visit=4,
    control="usual-care",
    treatment="active",
    consistency_and_no_interference=True,
    randomized_exchangeability=True,
    complete_joint_follow_up=True,
    monotone_surrogate=True,
)


def test_patient_surrogate_replays_bounds_and_threshold_in_a_fresh_process() -> None:
    law = PopulationJointSurrogateLaw("joint-population-law", (3, 0, 1, 0), (0, 0, 1, 3))
    plan = require_success(compile_patient_surrogate(DESIGN, law))
    result = require_success(plan.execute())
    assert result.reading == "population_identification"
    assert result.exact_bounds == (8, 12, 12)
    assert result.bounds == (2 / 3, 1.0)
    assert result.improver_probability == 0.75
    assert result.quantity.population_id == "adults"
    assert require_success(result.decide(2, 3)).reading == "all_at_or_above"
    assert require_success(result.decide(4, 5)).reading == "ambiguous"
    with pytest.raises(PatientSurrogateRefusal, match="does not identify one point"):
        require_success(result.point())

    from antecedent import load

    artifact = plan.export()
    assert artifact.expectation is not None
    reopened = require_success(load(artifact, expected=artifact.expectation))
    assert reopened.exact_bounds == result.exact_bounds
    assert reopened.identity == result.identity
    small_program = """
import json
import sys
from antecedent.composition_bundle import Bundle, consume_bundle
from antecedent.outcomes import require_success
from antecedent.patient_surrogate import consume_patient_surrogate
raw = sys.stdin.buffer.read()
live = require_success(consume_patient_surrogate(
    raw, expected_program=sys.argv[1], expected_result=sys.argv[2],
))
decision = require_success(live.decide(2, 3))
graph = Bundle.builder().add_artifact(raw, node_id="surrogate").build()
bundle = require_success(consume_bundle(bytes(graph.export()), expected_identity=graph.identity))
bundle.require_verified()
print(json.dumps([decision.identity, bundle.value("surrogate", "lower"), decision.reading]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", small_program, plan.program_identity, result.identity],
        input=bytes(artifact),
        capture_output=True,
        check=True,
        timeout=30,
    )
    decision = require_success(result.decide(2, 3))
    assert json.loads(completed.stdout) == [decision.identity, 2 / 3, "all_at_or_above"]


def test_sample_bounds_cannot_be_used_as_a_population_decision_claim() -> None:
    sample = SurrogatePatientSample(
        (
            ObservedSurrogatePatient("reference", False, False, True),
            ObservedSurrogatePatient("active", True, True, True),
        )
    )
    result = require_success(require_success(compile_patient_surrogate(DESIGN, sample)).execute())
    assert result.reading == "empirical_plugin"
    assert result.bounds == (0.0, 0.0)
    with pytest.raises(PatientSurrogateRefusal) as refused:
        require_success(result.decide(0, 1))
    assert refused.value.detail == reason_details.patient_surrogate.UNSUPPORTED_READING

    with pytest.raises(CausalTypeError, match="joint patient-level"):
        require_success(compile_patient_surrogate(DESIGN, (0.5, 0.25)))  # type: ignore[arg-type]
    with pytest.raises(PatientSurrogateRefusal, match="complete patient-level joint follow-up"):
        require_success(
            compile_patient_surrogate(
                replace(DESIGN, complete_joint_follow_up=False),
                sample,
            )
        )
