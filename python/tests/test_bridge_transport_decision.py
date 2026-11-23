"""Bridge transport decisions, their premise assessment and a bridge-terminal bundle, from Python.

Each fixture is the smallest one the Rust route tests use: 64 aligned predictive draws of a
biomarker and a cost, a surrogate-to-survival bridge (an affine map, or the trial-level line of 20
validation trials), a transport premise from ``validation-adults`` to ``adults`` and two actions on
the transported survival quantity. Every asserted number has an oracle computed here with numpy
from the draws or trials the route itself reads:

* a linear utility scores the mean of the bridged column and a squared utility its second moment,
  never the square of the mean;
* both trial lines are the least-squares lines of their trials, and the target trials sit 0.05
  above the source trials, so the discrepancy is 0.05 at every grid point;
* the robust bounds of a linear utility are the bridged mean shifted by the widest assessed band.

Each route exports, is consumed against the identity the consumer retained, refuses another
identity, and reports ``calibration == "not_measured"``.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest
from antecedent import bridge_transport_decision as btd
from antecedent import composition_bundle as cb
from antecedent import decision
from antecedent import scientific_bridge as sb
from antecedent.errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.execution import Execution
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal

Refusal = btd.BridgeTransportDecisionRefusal
ROWS = 64
TRIALS = 20
WRONG = "0" * 64
_NORMAL = NormalDist()


def outcome(variable: str, *, units: str) -> ScientificQuantity:
    return ScientificQuantity.outcome(
        variable, units=units, population="adults", regime="do(t=1)", functional="average_effect"
    )


SOURCE = outcome("trial:biomarker", units="mmol_per_l")
COST = outcome("trial:cost", units="usd")
TARGET = outcome("trial:survival", units="log_hazard")
VALIDATION = sb.Scope("validation-adults", "do(t=1)")
ADULTS = sb.Scope("adults", "do(t=1)")
PREMISE = sb.TransportabilityPremise(
    "validation-to-target",
    "trial surrogate relation transfers",
    established_in=VALIDATION,
    applies_to=ADULTS,
    evidence="independent-target-validation",
)


def draws(mean: float, sd: float, *, weights: tuple[float, ...] | None = None):
    source = [mean + sd * _NORMAL.inv_cdf((i + 0.5) / ROWS) for i in range(ROWS)]
    partner = [0.2 + 0.3 * _NORMAL.inv_cdf(((i * 487) % ROWS + 0.5) / ROWS) for i in range(ROWS)]
    identity = DistributionIdentity(
        semantic="posterior_predictive",
        quantities=(SOURCE, COST),
        alignment="joint",
        source_id="study-1",
        provider_id="posterior.provider",
        rng_id="seed:11",
        snapshot_id="snapshot-1",
        causal_contract_id="contract-1",
    )
    return JointDistributionArtifact(identity, np.column_stack([source, partner]), weights=weights)


def trials(offset: float) -> sb.TrialLevelMapping:
    pairs = []
    for i in range(TRIALS):
        x = -1.0 + 2.0 * i / (TRIALS - 1)
        pairs.append((x, 0.5 + 2.0 * x + offset + (0.01 if i % 2 == 0 else -0.01)))
    return sb.TrialLevelMapping(trials=pairs, min_trials=3, min_r_squared=0.0)


def contract(mapping, *, transport=None, scope=ADULTS, dependence=None) -> sb.BridgeContract:
    return sb.BridgeContract(
        kind="surrogate_terminal_outcome",
        source=SOURCE,
        source_class="surrogate_endpoint",
        target=TARGET,
        target_class="terminal_outcome",
        evidence_scope=scope,
        mapping=mapping,
        assumptions=(
            sb.Assumption(
                "validation",
                "surrogate_validation",
                "the surrogate is validated at the trial level",
                evidence="meta-analysis:ckd-2024",
            ),
        ),
        transport=transport,
        source_value="predictive_law",
        target_value="predictive_law",
        source_dependence=dependence,
    )


def premise_contract() -> sb.BridgeContract:
    """An affine bridge whose evidence was established in the validation population."""
    return contract(
        sb.AffineMapping(intercept=0.5, slope=2.0),
        transport=sb.TransportabilityPremise(
            "validation-to-target",
            "the trial-level surrogate relation holds in the target population",
            established_in=VALIDATION,
            applies_to=ADULTS,
            evidence="selection-diagram-review",
        ),
        scope=VALIDATION,
    )


def trial_contract() -> sb.BridgeContract:
    """The trial-level bridge whose source trials the premise assessment fits."""
    return contract(
        trials(0.0),
        transport=PREMISE,
        scope=VALIDATION,
        dependence=sb.SourceDependence.independent(
            "study-1 is not among the source validation trials"
        ),
    )


def choice(criterion: decision.Criterion | None = None) -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action("use-bridge", inputs=(TARGET,), utility=decision.x(0), kind="external"),
            decision.Action(
                "risk-sensitive",
                inputs=(TARGET,),
                utility=decision.x(0) * decision.x(0),
                kind="external",
            ),
        ),
        utility_units="score",
        criterion=criterion or decision.Criterion.expected_utility(),
        target_population="adults",
        structural_policy="require_invariant_best_action",
    )


def assess(grid=(-0.8, 0.0, 0.8)) -> btd.CompiledPremiseAssessment:
    return btd.compile_premise_assessment(
        PREMISE,
        source=trials(0.0),
        target=trials(0.05),
        source_trial_ids=[f"source-trial-{i}" for i in range(TRIALS)],
        target_trial_ids=[f"target-trial-{i}" for i in range(TRIALS)],
        surrogate_grid=list(grid),
        equivalence_margin=0.2,
    )


def refused(call, code: str) -> Refusal:
    with pytest.raises(Refusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    assert caught.value.reason_code == code, (caught.value.reason_code, str(caught.value))
    return caught.value


def bridged_column(result: btd.BridgeTransportDecision) -> np.ndarray:
    return np.asarray(result.distribution())[:, result.bridge.column]


def test_a_conditional_bridge_decision_scores_the_transported_law_and_replays() -> None:
    execution = Execution(seed=31)
    compiled = btd.compile_bridge_transport_decision(
        premise_contract(), draws(1.0, 0.3), choice(), stream=9, execution=execution
    )
    assert compiled.variant == "conditional"
    view = compiled.plan_view()
    assert view["route"] == "BridgeTransportDecision"
    assert view["identity"] == compiled.program_identity
    result = compiled.execute()
    assert result.variant == "conditional"
    assert result.program_identity == compiled.program_identity
    assert result.calibration == "not_measured"
    assert result.bridge.transported
    assert result.bridge.stream == 9
    assert result.assessment is None and result.robust is None
    assert result.target_posterior_decision is None
    mapped = bridged_column(result)
    assert result.bridge.mean == pytest.approx(mapped.mean(), rel=1e-9)
    chosen = result.decision
    assert chosen.n_draws == ROWS
    assert chosen.action("use-bridge").expected_utility == pytest.approx(mapped.mean(), rel=1e-9)
    second_moment = float(np.mean(mapped * mapped))
    assert chosen.action("risk-sensitive").expected_utility == pytest.approx(
        second_moment, rel=1e-9
    )
    # The squared utility reads every transported row, not the square of the mean.
    assert second_moment - mapped.mean() ** 2 > 1e-3
    assert chosen.verdict.actions[0] == "risk-sensitive"

    exported = result.export()
    assert exported.kind == "bridge_transport_decision_v1"
    assert exported.expectation is not None
    assert exported.expectation.identity == compiled.program_identity
    replay = btd.consume_bridge_transport_decision(
        exported, expected_identity=compiled.program_identity, execution=execution
    )
    assert replay.variant == "conditional"
    assert replay.program_identity == result.program_identity
    assert replay.bridge == result.bridge
    assert replay.decision == result.decision
    assert np.array_equal(np.asarray(replay.distribution()), np.asarray(result.distribution()))
    assert bytes(replay.export()) == bytes(exported)

    refused(
        lambda: btd.consume_bridge_transport_decision(
            exported, expected_identity=WRONG, execution=execution
        ),
        "bridge_binding_mismatch",
    )
    data = bytes(exported)
    with pytest.raises(CausalSerializationError):
        btd.consume_bridge_transport_decision(
            data[: len(data) // 2], expected_identity=compiled.program_identity
        )


def test_a_bridge_without_a_declared_premise_does_not_compile() -> None:
    refused(
        lambda: btd.compile_bridge_transport_decision(
            contract(sb.AffineMapping(intercept=0.5, slope=2.0)),
            draws(1.0, 0.3),
            choice(),
            stream=9,
        ),
        "bridge_binding_mismatch",
    )
    with pytest.raises(CausalTypeError):
        btd.compile_bridge_transport_decision(
            premise_contract(), draws(1.0, 0.3), choice().actions, stream=9
        )
    with pytest.raises(CausalValueError):
        btd.compile_bridge_transport_decision(
            premise_contract(), draws(1.0, 0.3), choice(), stream=9, robust=choice()
        )


def test_the_premise_assessment_fits_both_trial_lines_and_replays() -> None:
    execution = Execution(seed=73)
    compiled = assess()
    view = compiled.plan_view()
    assert view["route"] == "BridgeTransportPremiseAssessment"
    assert view["identity"] == compiled.program_identity
    result = compiled.execute(execution=execution)
    assert result.program_identity == compiled.program_identity
    assert result.calibration == "not_measured"
    assert result.equivalence_screen_passed
    for fit, mapping in ((result.source_fit, trials(0.0)), (result.target_fit, trials(0.05))):
        xs = np.array([x for x, _ in mapping.trials])
        ys = np.array([y for _, y in mapping.trials])
        slope, intercept = np.polyfit(xs, ys, 1)
        assert fit.slope == pytest.approx(slope, rel=1e-9)
        assert fit.intercept == pytest.approx(intercept, rel=1e-9)
        assert fit.trials == TRIALS
    assert [point.surrogate_effect for point in result.grid] == [-0.8, 0.0, 0.8]
    for point in result.grid:
        assert point.difference == pytest.approx(0.05, abs=1e-9)
        assert point.lower < point.difference < point.upper
        assert point.lower > -0.2
        assert point.upper < 0.2

    exported = result.export()
    assert exported.kind == "bridge_transport_premise_assessment_v1"
    replay = btd.consume_premise_assessment(
        exported, expected_identity=compiled.program_identity, execution=execution
    )
    assert replay == result
    assert bytes(replay.export()) == bytes(exported)
    refused(
        lambda: btd.consume_premise_assessment(
            exported, expected_identity=WRONG, execution=execution
        ),
        "bridge_binding_mismatch",
    )


def test_a_grid_outside_the_trial_overlap_is_refused() -> None:
    refused(lambda: assess(grid=(-0.8, 0.0, 1.5)), "scientific_bridge_invalid")
    with pytest.raises(CausalTypeError):
        btd.compile_premise_assessment(
            PREMISE,
            source=[(0.0, 0.1)],  # type: ignore[arg-type]
            target=trials(0.05),
            source_trial_ids=["s"],
            target_trial_ids=[f"target-trial-{i}" for i in range(TRIALS)],
            surrogate_grid=[0.0],
            equivalence_margin=0.2,
        )


def test_an_assessed_decision_binds_the_matched_premise_assessment_and_replays() -> None:
    execution = Execution(seed=43)
    assessment = assess()
    compiled = btd.compile_bridge_transport_decision(
        trial_contract(),
        draws(0.4, 0.2),
        choice(),
        stream=11,
        assessment=assessment,
        execution=execution,
    )
    assert compiled.variant == "assessed"
    assert compiled.plan_view()["route"] == "AssessedBridgeTransportDecision"
    result = compiled.execute()
    assert result.variant == "assessed"
    assert result.calibration == "not_measured"
    assert result.assessment is not None
    assert result.assessment.program_identity == assessment.program_identity
    assert result.assessment.equivalence_screen_passed
    assert result.decision == result.conditional_decision
    mapped = bridged_column(result)
    assert result.decision.action("use-bridge").expected_utility == pytest.approx(
        mapped.mean(), rel=1e-9
    )
    # The assessment is part of the decision's artifact, not one of its own.
    with pytest.raises(CausalUnsupportedError):
        result.assessment.export()

    exported = result.export()
    assert exported.kind == "assessed_bridge_transport_decision_v1"
    replay = btd.consume_bridge_transport_decision(
        exported, expected_identity=compiled.program_identity, execution=execution
    )
    assert replay.variant == "assessed"
    assert replay.assessment == result.assessment
    assert replay.decision == result.decision
    refused(
        lambda: btd.consume_bridge_transport_decision(
            exported, expected_identity=WRONG, execution=execution
        ),
        "bridge_binding_mismatch",
    )


def test_a_robust_decision_shifts_every_row_across_the_assessed_envelope() -> None:
    execution = Execution(seed=43)
    maximin = replace(choice(), criterion=decision.Criterion.maximin_over_structures())
    compiled = btd.compile_bridge_transport_decision(
        trial_contract(),
        draws(0.0, 0.1),
        choice(),
        stream=11,
        assessment=assess(),
        robust=maximin,
        execution=execution,
    )
    assert compiled.variant == "robust"
    result = compiled.execute()
    assert result.variant == "robust"
    assert result.calibration == "not_measured"
    assert result.robust is not None and result.assessment is not None
    assert {bound.id for bound in result.robust.actions} == {"use-bridge", "risk-sensitive"}
    for bound in result.robust.actions:
        assert bound.utility_lower <= bound.utility_upper
    lower = min(point.lower for point in result.assessment.grid)
    upper = max(point.upper for point in result.assessment.grid)
    mean = bridged_column(result).mean()
    linear = result.robust.action("use-bridge")
    assert linear.utility_lower == pytest.approx(mean + lower, rel=1e-9, abs=1e-12)
    assert linear.utility_upper == pytest.approx(mean + upper, rel=1e-9, abs=1e-12)

    exported = result.export()
    assert exported.kind == "assessed_bridge_discrepancy_robust_decision_v1"
    replay = btd.consume_bridge_transport_decision(
        exported, expected_identity=compiled.program_identity, execution=execution
    )
    assert replay.variant == "robust"
    assert replay.robust == result.robust
    assert replay.assessment == result.assessment
    refused(
        lambda: btd.consume_bridge_transport_decision(
            exported, expected_identity=WRONG, execution=execution
        ),
        "bridge_binding_mismatch",
    )


def test_a_target_posterior_decision_is_bound_to_its_premises_and_seed() -> None:
    execution = Execution(seed=43)
    regret = choice(decision.Criterion.expected_regret())
    weights = tuple(float(i) for i in range(ROWS))
    premises = btd.TargetPosteriorPremises("independent-target-trials", True, True)
    compiled = btd.compile_bridge_transport_decision(
        trial_contract(),
        draws(0.0, 0.1, weights=weights),
        regret,
        stream=11,
        assessment=assess(),
        target_posterior=premises,
        execution=execution,
    )
    assert compiled.variant == "target_posterior"
    result = compiled.execute()
    assert result.variant == "target_posterior"
    assert result.calibration == "not_measured"
    posterior = result.target_posterior_decision
    assert posterior is not None
    assert result.decision == posterior
    assert posterior.program_identity != result.conditional_decision.program_identity
    # The zero-weight source row carries no mass; the remaining weights are unequal.
    assert posterior.n_draws == ROWS - 1
    assert posterior.effective_draws < posterior.n_draws
    # The posterior draws are bound to the seed the program was compiled with.
    refused(lambda: compiled.execute(execution=Execution(seed=44)), "bridge_binding_mismatch")

    exported = result.export()
    assert exported.kind == "assessed_bridge_target_posterior_decision_v1"
    replay = btd.consume_bridge_transport_decision(
        exported, expected_identity=compiled.program_identity, execution=execution
    )
    assert replay.variant == "target_posterior"
    assert replay.decision == result.decision
    assert replay.assessment == result.assessment
    for identity, seed in ((WRONG, 43), (compiled.program_identity, 44)):
        refused(
            lambda identity=identity, seed=seed: btd.consume_bridge_transport_decision(
                exported, expected_identity=identity, execution=Execution(seed=seed)
            ),
            "bridge_binding_mismatch",
        )
    undeclared = btd.TargetPosteriorPremises("independent-target-trials", True, False)
    refused(
        lambda: btd.compile_bridge_transport_decision(
            trial_contract(),
            draws(0.0, 0.1, weights=weights),
            regret,
            stream=11,
            assessment=assess(),
            target_posterior=undeclared,
            execution=execution,
        ),
        "bridge_binding_mismatch",
    )


def test_a_bundle_with_a_predictive_bridge_terminal_recomputes_and_replays() -> None:
    execution = Execution(seed=41)
    declaration = contract(sb.AffineMapping(intercept=0.5, slope=2.0))
    source = draws(1.0, 0.3)
    bridged = require_success(sb.prepare_distribution_bridge(declaration).derive(source, seed=41))
    terminal = bridged.export("bridge-bundle-terminal")
    assert terminal.expectation is not None
    bundle = (
        cb.Bundle.builder()
        .add_artifact(
            "distribution", bytes(source.export("bridge-bundle-source-node")), node_id="source"
        )
        .add_artifact("scientific_bridge", bytes(terminal), node_id="terminal")
        .connect("source", "terminal")
        .build()
    )
    compiled = btd.compile_bridge_bundle_recomputation(
        bundle, "terminal", declaration, source, stream=41
    )
    assert compiled.plan_view()["route"] == "BridgeBundleRecomputation"
    receipt = compiled.execute(execution=execution)
    assert receipt.bundle_identity == bundle.identity
    assert receipt.terminal_node == "terminal"
    assert receipt.terminal_identity == terminal.expectation.identity
    assert receipt.program_identity == compiled.program_identity
    assert "source" in {dependency.node_id for dependency in receipt.dependencies}

    exported = receipt.export()
    assert exported.kind == "bridge_bundle_recomputation_portable_v1"
    replay = btd.consume_bridge_bundle_recomputation(
        exported,
        expected_bundle=bundle.identity,
        expected_identity=compiled.program_identity,
        execution=execution,
    )
    assert replay == receipt
    for expected_bundle, identity in (
        (WRONG, compiled.program_identity),
        (bundle.identity, WRONG),
    ):
        refused(
            lambda expected_bundle=expected_bundle, identity=identity: (
                btd.consume_bridge_bundle_recomputation(
                    exported,
                    expected_bundle=expected_bundle,
                    expected_identity=identity,
                    execution=execution,
                )
            ),
            "bridge_binding_mismatch",
        )
    refused(
        lambda: btd.compile_bridge_bundle_recomputation(
            bundle, "source", declaration, source, stream=41
        ),
        "bridge_binding_mismatch",
    )
