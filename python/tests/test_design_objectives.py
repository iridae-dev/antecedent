"""Independent scoring references for the unified typed native objective surface."""

from __future__ import annotations

import math

import numpy as np
import pytest
from antecedent import _native, design
from antecedent.errors import CausalDesignError, CausalTypeError, CausalValueError

MC = design.MonteCarlo(2, 2, 4, 0.0)


def test_structural_declarations_use_shared_result_without_a_fake_score() -> None:
    candidates = [
        design.StructuralCandidate("unknown", False, 0),
        design.StructuralCandidate("costly", True, 3),
        design.StructuralCandidate("cheap", True, 1),
    ]
    result = design.rank_designs(candidates)
    compatibility = design.rank_structural(candidates)
    assert isinstance(result, design.DesignRankingResult)
    assert result.basis == "structural_sufficiency_cost"
    assert result.candidates == compatibility.entries
    assert [c.id for c in result.candidates] == ["cheap", "costly", "unknown"]
    assert not hasattr(result.best, "score")
    assert not hasattr(result.best, "probability")
    assert result.identity == compatibility.identity
    assert design.rank_designs(list(reversed(candidates))).identity == result.identity
    assert result.to_dict()["structural_identity"] == compatibility.identity
    assert "caller-declared" in result.explain()
    assert "score=" not in repr(result)
    with pytest.raises(CausalValueError, match="structural_sufficiency_cost"):
        result.export()
    with pytest.raises(CausalValueError, match="structural"):
        design.rank_designs(candidates, prior=design.StructurePrior.uniform([True]))
    with pytest.raises(CausalTypeError, match="mixed"):
        design.rank_designs([candidates[0], design.Sampling(1)])  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError, match="bool"):
        design.StructuralCandidate("x", "false", 1)  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError, match="integer"):
        design.StructuralCandidate("x", True, 1.9)  # type: ignore[arg-type]


def test_identification_gain_and_absolute_probability_are_distinct() -> None:
    prior = design.StructurePrior((3.0, 1.0), (True, False), (1, 2))
    result = design.rank_designs(
        [design.Measurement((5,), id="unlock"), design.Sampling(1, id="unchanged")],
        prior=prior,
        variable_unlocks={0: [5]},
    )
    unlock = result.candidate("unlock")
    unchanged = result.candidate("unchanged")
    assert isinstance(unlock, design.IdentificationCandidate)
    assert isinstance(unchanged, design.IdentificationCandidate)
    assert unlock.score == pytest.approx(0.25)
    assert unlock.probability == pytest.approx(1.0)
    assert unchanged.score == pytest.approx(0.0)
    assert unchanged.probability == pytest.approx(0.75)


def test_graph_entropy_uses_original_channel_and_retains_heuristic_label() -> None:
    # Equal two-category prior: the symmetric soft channel is 1/2 + r/2 correct,
    # r=n/(50+n); H(prior)-H(posterior) does not depend on the sampled observation.
    prior = design.StructurePrior.uniform([True, False])
    plans = [design.Sampling(1, id="one"), design.Sampling(50, id="fifty")]
    result = design.rank_designs(
        plans, prior=prior, objective=design.GraphEntropy(), monte_carlo=MC, rng_seed=5
    )
    assert result.basis == "graph_entropy"
    assert [c.id for c in result.candidates] == ["fifty", "one"]
    for c in result.candidates:
        assert isinstance(c, design.ObjectiveCandidate)
        n = 1 if c.id == "one" else 50
        posterior = 0.5 + 0.5 * n / (50.0 + n)
        entropy = -posterior * math.log(posterior) - (1 - posterior) * math.log(1 - posterior)
        assert c.score == pytest.approx(math.log(2) - entropy, abs=1e-13)
        assert c.implemented_functional == "heuristic_graph_channel_entropy"
        assert c.evaluation == "monte_carlo"
        assert not hasattr(c, "probability")
    native = _native.rank_designs(
        [1.0, 1.0],
        [1, 0],
        [0, 1],
        [
            {"kind": "increase_sampling_rate", "additional_samples": 1, "tag": 0},
            {"kind": "increase_sampling_rate", "additional_samples": 50, "tag": 1},
        ],
        "reduce_graph_entropy",
        min_batches=2,
        max_batches=2,
        batch_size=4,
        rank_uncertainty_threshold=0.0,
        seed=5,
    )
    assert [c.score for c in result.candidates] == [c.score for c in native.ranked]
    with pytest.raises(CausalValueError, match="prior=StructurePrior"):
        design.rank_designs(plans, objective=design.GraphEntropy())


def test_effect_width_is_exact_gram_se_reduction_with_prospective_information() -> None:
    objective = design.EffectWidth(
        (4.0, 0.0, 0.0, 9.0),
        sigma2=4.0,
        treatment_col=1,
        n=20,
        measurement_columns=(design.MeasurementColumn(7, (0.0, 0.0), 2.0, 1.0),),
        intervention=design.DesignInformation((4.0, 0.0, 0.0, 36.0), 4.0, 20),
        environments=(design.EnvironmentInformation(8, (4.0, 0.0, 0.0, 9.0), 20),),
    )
    result = design.rank_designs(
        [
            design.Sampling(20, id="sample"),
            design.Measurement((7,), id="measure"),
            design.Experiment((0,), id="experiment"),
            design.Environment(8, id="environment"),
        ],
        objective=objective,
    )
    assert result.basis == "effect_width"
    # Independent diagonal covariance: sqrt(4/9)=2/3 at baseline. Equal sampling or
    # pooled environment halves variance. Lower measurement noise / larger intervention
    # treatment Gram quarters variance.
    expected = {
        "sample": 2 / 3 - math.sqrt(2) / 3,
        "environment": 2 / 3 - math.sqrt(2) / 3,
        "measure": 1 / 3,
        "experiment": 1 / 3,
    }
    for c in result.candidates:
        assert isinstance(c, design.ObjectiveCandidate)
        assert c.score == pytest.approx(expected[c.id], abs=1e-13)
        assert c.evaluation == "exact" and c.stderr == 0.0
        assert c.implemented_functional == "ols_gram_se_reduction"
    with pytest.raises(CausalValueError, match="square"):
        design.EffectWidth((1.0, 2.0), 1.0, 0, 10)
    with pytest.raises(CausalDesignError, match="Gram|gram|singular"):
        design.rank_designs([design.Sampling(1)], objective=design.EffectWidth((0.0,), 1.0, 0, 10))


def test_model_distinction_matches_independent_constant_loglik_gap() -> None:
    result = design.rank_designs(
        [design.Sampling(1, id="one"), design.Sampling(50, id="fifty")],
        objective=design.ModelDistinction((10, 20), ((1.0, 1.0), (4.0, 4.0))),
        monte_carlo=MC,
        rng_seed=5,
    )
    assert result.basis == "model_distinction"
    for c in result.candidates:
        assert isinstance(c, design.ObjectiveCandidate)
        n = 1 if c.id == "one" else 50
        assert c.score == pytest.approx(3.0 * n / (50.0 + n), abs=1e-13)
        assert c.implemented_functional == "heuristic_reliability_scaled_loglik_gap"
        assert c.evaluation == "monte_carlo"
    with pytest.raises(CausalValueError, match="distinct"):
        design.ModelDistinction((1, 1), ((0.0,), (0.0,)))
    with pytest.raises(CausalValueError, match="draw count"):
        design.ModelDistinction((1, 2), ((0.0,), (0.0, 1.0)))


def test_callable_regret_reaches_native_exact_preposterior_without_affine_restriction() -> None:
    # The callback is nonlinear in state: for state {1/4,3/4}, betting has utility
    # theta**2-1/4. Prior betting utility=1/16. After one head it is 3/16,
    # after a tail abstain. Heads mass=1/2 => EVSI=3/32-1/16=1/32.
    def utility(actions: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
        return np.outer(actions, outcomes**2 - 0.25).ravel()

    objective = design.DecisionRegret(
        (0.0, 1.0), utility, design.StatePrior.draws((0.25, 0.75)), design.BinomialSignal()
    )
    result = design.rank_designs(
        [design.Sampling(1, id="one"), design.Measurement((0,), id="unlicensed")],
        objective=objective,
        monte_carlo=MC,
        rng_seed=5,
    )
    assert result.basis == "decision_regret"
    assert len(result.candidates) == 1
    assert isinstance(result.best, design.ObjectiveCandidate)
    assert result.best.score == pytest.approx(0.03125, abs=1e-13)
    assert result.best.evaluation == "exact"
    assert result.best.implemented_functional == "preposterior_expected_value_of_sample_information"
    assert result.violations[0].id == "unlicensed"
    assert result.violations[0].constraint == "unlicensed_candidate"
    assert "decision_regret" in result.explain()
    with pytest.raises(CausalValueError, match="decision_regret"):
        result.export()
    with pytest.raises(CausalValueError, match="do not apply"):
        design.rank_designs(
            [design.Sampling(1)], objective=objective, cost_map=design.CostMap("a", "a", 1.0)
        )


def test_callable_regret_preserves_native_callback_refusal() -> None:
    def bad(actions: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
        raise RuntimeError("utility exploded")

    with pytest.raises(CausalDesignError, match="utility exploded") as caught:
        design.rank_designs(
            [design.Sampling(1)],
            objective=design.DecisionRegret(
                (0.0, 1.0), bad, design.StatePrior.draws((0.25, 0.75)), design.BinomialSignal()
            ),
        )

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "utility exploded"


def test_value_gate_uses_absolute_identified_mass_including_unchanged_plans() -> None:
    from antecedent.joint_distribution import ScientificQuantity

    def quantity(name: str) -> ScientificQuantity:
        return ScientificQuantity(
            variable_id=f"schema:{name}",
            variable_name=name,
            role="outcome",
            units="dimensionless",
            population_id="target",
            regime_id="observational",
            horizon=0,
            functional_id="state",
        )

    decision = design.DesignDecision(
        contract="test-absolute-gate",
        utility_units="utility",
        actions=(design.ActionUtility("no", 1.0, -1.0), design.ActionUtility("yes", 0.0, 1.0)),
        prior=design.StatePrior.draws((0.0, 1.0)),
    )
    structure = design.StructurePrior((3.0, 1.0), (True, False), (1, 2))
    studies = [
        design.Candidate("unlock", 1, design.BinomialSignal(), plan=design.Measurement((5,))),
        design.Candidate("unchanged", 1, design.BinomialSignal(), plan=design.Sampling(1)),
    ]
    signal = design.SignalSpec("gate-prior", quantity("state"), quantity("signal"), ("test:prior",))
    result = design.rank_designs(
        studies,
        decision=decision,
        prior=structure,
        variable_unlocks={0: [5]},
        signal=signal,
        min_identification=0.9,
    )
    assert result.gate is not None
    by_id = {entry.id: entry for entry in result.gate.entries}
    assert by_id["unlock"].probability == pytest.approx(1.0)
    assert by_id["unchanged"].probability == pytest.approx(0.75)
    assert result.gate.rejected == ("unchanged",)
    assert [entry.id for entry in result.candidates] == ["unlock"]
