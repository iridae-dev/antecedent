# SPDX-License-Identifier: MIT OR Apache-2.0
"""Plan an HCI study, select a follow-up and assemble an evidence-linked claim audit.

Antecedent plans and analyzes randomized studies, checks identification obligations,
and ranks studies. Application code freezes protocol snapshots, generates task data,
and connects explicitly declared manuscript claims to those results.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from statistics import NormalDist

import antecedent as ant
import numpy as np
from antecedent import decision, design, repair
from antecedent.design import planning
from antecedent.evidence_ledger import EvidenceLedger, Lineage, SourceDeclaration
from antecedent.experiment import (
    ExperimentDesign,
    FactorialRandomization,
    RandomizedEffect,
)
from antecedent.interference import CompleteRandomization
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

POPULATION = "eligible_English_speaking_desktop_volunteers"
TRIALS = 20  # per participant with incorrect AI advice; another 20 have correct advice
SEED = 20261030
SD, MEANINGFUL_EFFECT = 0.16, 0.08
# Only the generator and independent truth checks read these values.
TRUTH = {
    "baseline": 0.64,
    "content": 0.06,
    "pause": 0.02,
    "time_content": 1.0,
    "time_pause": 5.0,
}


def snapshot(record):
    text = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def plan_study():
    assumptions = planning.Assumptions.continuous(SD, effect=MEANINGFUL_EFFECT)
    plan = planning.plan_experiment(
        planning.Complete(),
        assumptions,
        planning.Power(0.8),
        planning.SampleSize(),
        simulation=planning.Simulation(512, 17),
    )
    assert plan.target_met and plan.analytic.value >= 0.8
    previous = planning.operating_characteristics(
        planning.Complete(), assumptions, plan.units - 1
    )
    assert previous.analytic.value < 0.8
    simulation = plan.simulation
    assert abs(simulation.value - plan.analytic.value) < 6 * simulation.monte_carlo_se
    variants = []
    for sd, attrition in ((0.16, 0.0), (0.20, 0.0), (0.16, 0.20)):
        alternative = planning.sample_size(
            planning.Complete(),
            planning.Assumptions.continuous(sd, effect=0.08, attrition=attrition),
        )
        variants.append(
            {
                "participant_sd": sd,
                "attrition": attrition,
                "participants": alternative.units,
            }
        )
    protocol = {
        "version": "before_pilot",
        "population": POPULATION,
        "assignment": "complete randomization by participant to the whole interface",
        "primary_outcome": "participant accuracy on incorrect-AI tasks",
        "primary_target": "mean effect of interface assignment in the trial cohort",
        "secondary_outcomes": [
            "mean seconds per incorrect-AI task",
            "accuracy on correct-AI tasks",
        ],
        "primary_analysis": "complete-randomization difference in participant means",
        "participants": plan.units,
        "tasks_per_condition": TRIALS,
        "stopping": "fixed enrollment",
        "plan_request_identity": plan.request_identity,
    }
    return plan, protocol, variants


def generate_study(n, seed, *, factorial=False):
    rng = np.random.default_rng(seed)
    if factorial:
        assert n % 4 == 0
        labels = rng.permutation(np.repeat(np.arange(4), n // 4))
        content = labels % 2 == 1
        pause = labels >= 2
    else:
        content = np.zeros(n, dtype=bool)
        content[rng.permutation(n)[: n // 2]] = True
        pause = (
            content.copy()
        )  # the first interface bundles information and a review prompt
    ability = rng.uniform(-0.06, 0.06, n)
    motivation = rng.normal(size=n)  # unobserved to the analysis
    baseline = TRUTH["baseline"] + ability + 0.03 * np.tanh(motivation)
    task_noise = rng.random((n, TRIALS))
    probability = baseline + TRUTH["content"] * content + TRUTH["pause"] * pause
    accuracy = np.mean(task_noise < probability[:, None], axis=1)
    # Voluntary verification is affected by motivation as well as the interface.
    verification = rng.random(n) < 1 / (1 + np.exp(-motivation - 0.8 * content))
    seconds = (
        30
        + 2 * ability
        + rng.normal(0, 2, n)
        + TRUTH["time_content"] * content
        + TRUTH["time_pause"] * pause
    )
    correct_ai = np.mean(rng.random((n, TRIALS)) < 0.92, axis=1)
    unit_ids = [f"{'followup' if factorial else 'pilot'}:{i}" for i in range(n)]
    rows = {
        "accuracy": accuracy,
        "seconds": seconds,
        "correct_ai": correct_ai,
        "verification": verification.astype(float),
    }
    # Retain all task observations; only independent participant summaries enter the estimator.
    assert accuracy.shape == (n,) and task_noise.shape == (n, TRIALS)
    return rows, content, pause, unit_ids


def analyze(rows, primary, pause, unit_ids, *, factorial=False):
    assignment = (
        FactorialRandomization(pause.tolist(), (len(primary) // 4,) * 4)
        if factorial
        else CompleteRandomization(int(sum(primary)))
    )
    declared = ExperimentDesign(
        assignment,
        primary.tolist(),
        unit_ids,
        unit_ids,
        treatment_arms=("no_content", "content")
        if factorial
        else ("baseline", "explanation_interface"),
    )
    query = RandomizedEffect("accuracy", declared)
    prepared = ant.prepare(rows, query=query, execution=ant.Execution(seed=SEED))
    assert prepared.inspect().stage == "prepared"
    outcome = ant.estimate(prepared)
    result = require_success(outcome)
    fit = result.randomized_effect
    y = rows["accuracy"]
    if factorial:
        cell = [
            float(np.mean(y[(primary == a) & (pause == b)]))
            for b in (False, True)
            for a in (False, True)
        ]
        expected = 0.5 * ((cell[1] - cell[0]) + (cell[3] - cell[2]))
        expected_pause = 0.5 * ((cell[2] - cell[0]) + (cell[3] - cell[1]))
        expected_interaction = cell[3] - cell[2] - cell[1] + cell[0]
        assert math.isclose(fit.second_factor_effect, expected_pause, abs_tol=1e-12)
        assert math.isclose(
            fit.factorial_interaction, expected_interaction, abs_tol=1e-12
        )
    else:
        expected = float(np.mean(y[primary]) - np.mean(y[~primary]))
    assert math.isclose(fit.effect, expected, abs_tol=1e-12)
    report = result.inspect()
    assert report.identification.available and report.uncertainty.available
    return outcome, result, fit


def identification_followups(n):
    # Projecting the original bundled study onto content/accuracy leaves the other
    # component as a common cause. More bundle data does not separate the components.
    graph = ant.Admg.from_edges(
        ["content", "accuracy", "questionnaire"],
        [("content", "accuracy")],
        [("content", "accuracy")],
    )
    contract = repair.HedgeContract(
        graph=graph,
        treatment="content",
        outcome="accuracy",
        population=POPULATION,
        observed=("content", "accuracy"),
    )
    candidates = []
    for label, kind, measured, interventions, size in (
        ("repeat_bundle", "sample_increase", ("content", "accuracy"), (), n),
        (
            "questionnaire",
            "observation",
            ("content", "accuracy", "questionnaire"),
            (),
            n,
        ),
        ("factorial_128", "experiment", ("accuracy",), ("content",), 128),
        ("factorial_256", "experiment", ("accuracy",), ("content",), 256),
    ):
        candidates.append(
            repair.StudyCandidate(
                label=label,
                kind=kind,
                population=POPULATION,
                measured=measured,
                interventions=interventions,
                sample_size=size,
                recruitment="new volunteers meeting the same eligibility criteria",
                timing="new session after protocol amendment",
                unit="participant",
                cost=size,
                cost_unit="participant_sessions",
                notes=(
                    "Factorial trials independently randomize content and pause; this contract projects onto the content contrast.",
                ),
            )
        )
    checked = repair.repair(contract, candidates)
    assert checked.classification("repeat_bundle") == "insufficient"
    assert checked.classification("questionnaire") == "insufficient"
    assert checked.classification("factorial_128") == "verified_sufficient"
    assert checked.classification("factorial_256") == "verified_sufficient"
    assert checked.inference_claim == "none"
    replay = repair.consume(checked.export())
    assert replay.outcome == checked.outcome
    # A randomized component study does not identify the effect of voluntarily inspecting it.
    mechanism = repair.HedgeContract(
        graph=ant.Admg.from_edges(
            ["content", "pause", "inspection", "accuracy"],
            [
                ("content", "inspection"),
                ("content", "accuracy"),
                ("pause", "accuracy"),
                ("inspection", "accuracy"),
            ],
            [("inspection", "accuracy")],
        ),
        treatment="inspection",
        outcome="accuracy",
        population=POPULATION,
        observed=("content", "pause", "inspection", "accuracy"),
    )
    mediator_check = repair.repair(
        mechanism,
        [
            repair.StudyCandidate(
                label="randomized_components",
                kind="experiment",
                population=POPULATION,
                measured=("inspection", "accuracy"),
                interventions=("content", "pause"),
                sample_size=128,
                recruitment="eligible new volunteers",
                timing="follow-up session",
                unit="participant",
                cost=128,
                cost_unit="participant_sessions",
            )
        ],
    )
    assert mediator_check.classification("randomized_components") == "insufficient"
    return checked, mediator_check


def rank_followups(prior_source):
    state = ScientificQuantity(
        variable_id="schema:content_effect",
        variable_name="content_effect",
        role="outcome",
        units="accuracy_difference",
        population_id=POPULATION,
        regime_id="balanced_pause_conditions",
        horizon=0,
        functional_id="state",
    )
    # A planning judgment, not a component posterior identified by the bundled pilot.
    mean, variance = 0.04, 0.04**2
    contract = decision.Contract(
        actions=(
            decision.Action(
                "retain_information", inputs=(state,), utility=decision.x(0) - 0.03
            ),
            decision.Action(
                "omit_information", inputs=(state,), utility=0 * decision.x(0)
            ),
        ),
        utility_units="accuracy_utility",
        criterion=decision.Criterion.expected_utility(),
        target_population=POPULATION,
    )
    declared = design.DesignDecision.from_contract(
        contract, prior=design.StatePrior.normal(mean, variance)
    )
    signal = design.SignalSpec(
        prior_id="declared_component_design_prior",
        state=state,
        observation=state,
        evidence_lineage=(prior_source, "amendment:component_prior"),
        rng_seed=17,
    )
    noise = 4 * SD**2  # variance of a balanced two-arm marginal contrast, times total N
    candidates = [
        design.Candidate(
            f"factorial_{n}",
            n,
            design.GaussianMeanSignal(noise),
            cost=n * 0.00003,
            cost_unit="accuracy_utility",
        )
        for n in (128, 256)
    ]
    ranking = design.rank_designs(
        candidates,
        decision=declared,
        signal=signal,
        cost_map=design.CostMap("accuracy_utility", "accuracy_utility", 1.0),
    )
    oracle = {}
    for entry in ranking.candidates:
        posterior_variance = 1 / (1 / variance + entry.sample_size / noise)
        spread = math.sqrt(variance - posterior_variance)
        current = mean - 0.03
        z = current / spread
        expected = (
            spread * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
            + current * NormalDist().cdf(z)
            - max(current, 0.0)
        )
        assert abs(entry.evsi - expected) < 1e-9
        assert abs(entry.net_value - (expected - entry.sample_size * 0.00003)) < 1e-9
        assert entry.evsi <= entry.evpi + 1e-12
        oracle[entry.id] = entry.net_value
    assert ranking.best.id == max(oracle, key=oracle.get) == "factorial_128"
    assert ranking.best.net_value > 0
    consumed = design.consume(ranking.export(), expected_identity=ranking.expectation())
    assert consumed.identity == ranking.identity and all(
        e.natively_replayed for e in consumed.entries
    )
    return ranking


def main():
    started = time.perf_counter()
    plan, protocol, variants = plan_study()
    protocol_text = json.dumps(protocol, sort_keys=True)
    protocol_id = snapshot(protocol)  # frozen before generating any observed data
    ledger = EvidenceLedger()
    rows, assigned, pause, ids = generate_study(plan.units, 17)
    pilot_source = ledger.add_source(
        SourceDeclaration(
            "pilot_participants", "experiment", [("pilot", i) for i in ids]
        )
    )
    pilot_outcome, pilot, pilot_fit = analyze(rows, assigned, pause, ids)
    # Independent participant-level variance: repeated tasks are not independent participants.
    variance = float(
        np.var(rows["accuracy"][assigned], ddof=1) / sum(assigned)
        + np.var(rows["accuracy"][~assigned], ddof=1) / sum(~assigned)
    )
    assert math.isclose(pilot_fit.variance_upper_bound, variance, abs_tol=1e-12)
    assert abs(pilot_fit.effect - 0.08) <= 4 * math.sqrt(variance)
    time_design = ExperimentDesign(
        CompleteRandomization(int(sum(assigned))), assigned.tolist(), ids, ids
    )
    time_outcome = ant.analyze(
        rows,
        query=RandomizedEffect("seconds", time_design),
        execution=ant.Execution(seed=SEED),
    )
    time_fit = require_success(time_outcome).randomized_effect
    assert (
        abs(
            time_fit.effect
            - (
                float(
                    np.mean(rows["seconds"][assigned])
                    - np.mean(rows["seconds"][~assigned])
                )
            )
        )
        < 1e-12
    )
    correct_outcome = ant.analyze(
        rows,
        query=RandomizedEffect("correct_ai", time_design),
        execution=ant.Execution(seed=SEED),
    )
    correct_fit = require_success(correct_outcome).randomized_effect
    assert math.isclose(
        correct_fit.effect,
        float(
            np.mean(rows["correct_ai"][assigned])
            - np.mean(rows["correct_ai"][~assigned])
        ),
        abs_tol=1e-12,
    )
    assert correct_fit.interval_95[0] < 0 < correct_fit.interval_95[1]
    checked, mechanism = identification_followups(plan.units)
    ranking = rank_followups(pilot_source)
    amendment = {
        "version": "before_followup",
        "parent_protocol": protocol_id,
        "reason": "separate explanatory information from a review prompt",
        "primary_target": "content main effect averaged over balanced pause conditions",
        "design_prior": {"mean": 0.04, "sd": 0.04},
        "cost_per_session": 0.00003,
        "participants": ranking.best.sample_size,
        "assignment": "independent fixed-cell factorial randomization",
        "primary_contrast": "content main effect",
        "secondary_contrasts": ["pause main effect", "interaction"],
    }
    amendment_id = snapshot(amendment)
    follow_rows, content, pause, follow_ids = generate_study(
        ranking.best.sample_size, 43, factorial=True
    )
    follow_source = ledger.add_source(
        SourceDeclaration(
            "followup_participants", "experiment", [("followup", i) for i in follow_ids]
        )
    )
    follow_outcome, follow, follow_fit = analyze(
        follow_rows, content, pause, follow_ids, factorial=True
    )
    assert math.isclose(TRUTH["content"] + TRUTH["pause"], 0.08, abs_tol=1e-12)
    # A separate native refusal continues to block the inspection-mechanism claim.
    audit = [
        {
            "claim": "The assigned explanation interface improved accuracy in the studied tasks",
            "status": "supported"
            if pilot_fit.interval_95 and pilot_fit.interval_95[0] > 0
            else "estimate_only",
            "evidence": pilot_source,
            "effect": pilot_fit.effect,
            "interval": pilot_fit.interval_95,
        },
        {
            "claim": "The factorial study estimates the effect of content separately from the pause",
            "status": "supported",
            "evidence": follow_source,
            "effect": follow_fit.effect,
            "interval": follow_fit.interval_95,
        },
        {
            "claim": "Explanation content independently improved accuracy",
            "status": "supported"
            if follow_fit.interval_95 and follow_fit.interval_95[0] > 0
            else "inconclusive",
            "evidence": follow_source,
            "effect": follow_fit.effect,
            "interval": follow_fit.interval_95,
        },
        {
            "claim": "The benefit occurred because people inspected and understood the AI",
            "status": "unsupported",
            "evidence": "inspection identification check",
            "reason": mechanism.classification("randomized_components")
            + "; understanding was not measured",
        },
        {
            "claim": "The interface did not slow participants down",
            "status": "contradicted"
            if time_fit.interval_95 and time_fit.interval_95[0] > 0
            else "unresolved",
            "evidence": pilot_source,
            "seconds_added": time_fit.effect,
            "interval": time_fit.interval_95,
        },
        {
            "claim": "There was no accuracy loss when AI advice was correct",
            "status": "not_established",
            "evidence": pilot_source,
            "effect": correct_fit.effect,
            "interval": correct_fit.interval_95,
            "reason": "The pointwise secondary interval includes losses; no noninferiority margin was declared",
        },
        {
            "claim": "The results establish a benefit for all AI users",
            "status": "unsupported",
            "reason": "No population transport or task-generalization argument was supplied",
        },
        {
            "claim": "The study establishes improved trust",
            "status": "unsupported",
            "reason": "Accuracy and verification behavior are not a validated trust measure",
        },
    ]
    assert audit[0]["status"] == "supported"
    assert (
        next(
            c
            for c in audit
            if c["claim"] == "The interface did not slow participants down"
        )["status"]
        == "contradicted"
    )
    assert (
        protocol_text == json.dumps(protocol, sort_keys=True)
        and protocol_id != amendment_id
    )
    record = ledger.add_source(
        SourceDeclaration(
            "paper_evidence",
            "fitted_model",
            [("pilot", i) for i in ids] + [("followup", i) for i in follow_ids],
            lineage=Lineage.derived_from_used_data([pilot_source, follow_source]),
        )
    )
    assert (
        ledger.sources[-1].effective_observations
        == plan.units + ranking.best.sample_size
    )
    artifact = pilot_outcome.export()
    replayed = ant.load(artifact, expected=artifact.expectation)
    assert replayed.inspect().answer == pilot.inspect().answer
    follow_artifact = follow_outcome.export()
    assert (
        ant.load(follow_artifact, expected=follow_artifact.expectation).inspect().answer
        == follow.inspect().answer
    )
    try:
        ant.load(follow_artifact, expected=artifact.expectation)
    except ant.errors.CausalError:
        pass
    else:
        raise AssertionError(
            "The component result must not satisfy the bundle result expectation"
        )
    abstract = (
        f"In the recruited cohort and tested tasks, assignment to the bundled interface changed accuracy by {100 * pilot_fit.effect:.2f} percentage points "
        f"(95% interval {100 * pilot_fit.interval_95[0]:.2f} to {100 * pilot_fit.interval_95[1]:.2f}) and added {time_fit.effect:.2f} seconds per task. "
        f"The independent factorial follow-up estimated a content effect of {100 * follow_fit.effect:.2f} percentage points "
        f"(95% interval {100 * follow_fit.interval_95[0]:.2f} to {100 * follow_fit.interval_95[1]:.2f}). "
        "These results do not establish understanding as the mechanism or a benefit beyond the studied population and tasks."
    )
    print(
        json.dumps(
            {
                "protocol": protocol,
                "protocol_identity": protocol_id,
                "planning": {
                    "participants": plan.units,
                    "arms": plan.arm_units,
                    "analytic_power": plan.analytic.value,
                    "simulated_power": plan.simulation.value,
                    "simulation_se": plan.simulation.monte_carlo_se,
                    "sensitivity": variants,
                },
                "pilot": {
                    "effect": pilot_fit.effect,
                    "participant_neyman_variance": variance,
                    "identification": pilot.inspect().identification.summary,
                    "uncertainty": pilot_fit.uncertainty,
                },
                "identification_repair": {
                    name: checked.classification(name)
                    for name in (
                        "repeat_bundle",
                        "questionnaire",
                        "factorial_128",
                        "factorial_256",
                    )
                },
                "study_ranking": [
                    {"id": r.id, "evsi": r.evsi, "net_value": r.net_value}
                    for r in ranking.candidates
                ],
                "amendment": amendment,
                "amendment_identity": amendment_id,
                "followup": {
                    "content_effect": follow_fit.effect,
                    "pause_effect": follow_fit.second_factor_effect,
                    "interaction": follow_fit.factorial_interaction,
                },
                "claim_audit": audit,
                "supported_abstract": abstract,
                "paper_evidence": record,
                "pilot_artifact_sha256": hashlib.sha256(bytes(artifact)).hexdigest(),
                "followup_artifact_sha256": hashlib.sha256(
                    bytes(follow_artifact)
                ).hexdigest(),
                "mismatched_result_expectation": "refused",
                "distinct_participants": ledger.sources[-1].effective_observations,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
