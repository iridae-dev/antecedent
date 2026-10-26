"""Observation queries: one test per Python entry and per native entry.

Values are hand-derived. Confounded binary model: ``Z ~ Bernoulli(0.5)``,
``P(S=1 | Z) = 0.9, 0.1``, ``P(Y=1 | S, Z) = 0.1, 0.5, 0.3, 0.9`` keyed
``(S=0,Z=0), (0,1), (1,0), (1,1)``:

* ``P(S=1) = 0.5``, ``P(Y=1 | S=1) = 0.36``, ``P(Y=1 | S=0) = 0.46``;
* ``P(Y=1 | S=1, Z=0) = 0.3`` with ``P(S=1, Z=0) = 0.45``;
* ``P(Y=1 | do(S=1)) = 0.6``, ``P(Y=1 | do(S=0)) = 0.3``.
"""

from __future__ import annotations

import json

import pytest
from antecedent import _native
from antecedent import mechanism_composition as mc
from antecedent import observation_query as oq
from antecedent.observation_query import ObservationQueryRefusal

TOL = 1e-12
ROLES = {"S": "treatment", "Y": "outcome"}


def _quantity(variable: str) -> mc.Quantity:
    return mc.Quantity(
        variable,
        units="category",
        population_id="pop",
        role=ROLES.get(variable, "covariate"),
        variable_name=f"label of {variable}",
    )


def _table(variable, parents, rows) -> mc.Mechanism:  # type: ignore[no-untyped-def]
    declaration = mc.MechanismDeclaration(
        output=_quantity(variable),
        parents=tuple(_quantity(p) for p in parents),
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


def _confounded() -> mc.PreparedMechanismComposition:
    p_s = {"0": 0.9, "1": 0.1}
    p_y = {("0", "0"): 0.1, ("0", "1"): 0.5, ("1", "0"): 0.3, ("1", "1"): 0.9}
    return mc.compose_mechanisms(
        [
            _table("Z", [], [((), (0.5, 0.5))]),
            _table("S", ["Z"], [((z,), (1 - p_s[z], p_s[z])) for z in "01"]),
            _table(
                "Y",
                ["S", "Z"],
                [((s, z), (1 - p_y[(s, z)], p_y[(s, z)])) for s in "01" for z in "01"],
            ),
        ],
        target_population_id="pop",
        admit_declared_mechanisms=True,
    )


ILL_WELL = (["ill", "well"], ["neg", "pos"], [[0.1, 0.9], [0.95, 0.05]])


# ------------------------------------------------------------ composed queries


def test_condition_on_reads_the_observational_conditional_and_its_evidence() -> None:
    model = _confounded()
    zero = oq.condition_on(model, "Y", {"S": "0"})
    assert zero.distribution["1"] == pytest.approx(0.46, abs=TOL)
    assert zero.evidence_probability == pytest.approx(0.5, abs=TOL)
    cell = oq.condition_on(model, "Y", {"S": "1", "Z": "0"})
    assert cell.distribution["1"] == pytest.approx(0.3, abs=TOL)
    assert cell.evidence_probability == pytest.approx(0.45, abs=TOL)
    assert cell.operation == "conditioning"


def test_condition_versus_intervene_pairs_both_laws_with_their_gap() -> None:
    model = _confounded()
    one = oq.condition_versus_intervene(model, "Y", {"S": "1"})
    assert one.paired.conditioning["1"] == pytest.approx(0.36, abs=TOL)
    assert one.paired.intervention["1"] == pytest.approx(0.6, abs=TOL)
    assert one.paired.max_abs_difference == pytest.approx(0.24, abs=TOL)
    zero = oq.condition_versus_intervene(model, "Y", {"S": "0"})
    assert zero.paired.conditioning["1"] == pytest.approx(0.46, abs=TOL)
    assert zero.paired.intervention["1"] == pytest.approx(0.3, abs=TOL)
    assert zero.paired.max_abs_difference == pytest.approx(0.16, abs=TOL)


def test_the_native_observe_entry_answers_a_sealed_intervention_request() -> None:
    model = _confounded()
    wire = {
        "operation": "intervention",
        "outcome": "Y",
        "conditioning": [],
        "interventions": [["S", "0"]],
    }
    claim = oq._claim(_native.observation_query_observe(oq._handle(model), json.dumps(wire)))
    assert claim.operation == "intervention"
    assert claim.distribution["1"] == pytest.approx(0.3, abs=TOL)
    # Z has no parent, so setting it equals observing it: 0.1 * 0.9 + 0.9 * 0.5.
    wire["interventions"] = [["Z", "1"]]
    root = oq._claim(_native.observation_query_observe(oq._handle(model), json.dumps(wire)))
    assert root.distribution["1"] == pytest.approx(0.54, abs=TOL)


def test_the_native_pair_entry_returns_both_laws_of_one_assignment() -> None:
    model = _confounded()
    claim = oq._claim(
        _native.observation_query_pair(oq._handle(model), "Y", json.dumps([["S", "1"]]))
    )
    assert claim.operation == "conditioning_and_intervention"
    assert claim.paired.conditioning["1"] == pytest.approx(0.36, abs=TOL)
    assert claim.paired.intervention["1"] == pytest.approx(0.6, abs=TOL)


def test_threshold_crossing_lists_the_levels_that_cross_in_each_direction() -> None:
    model = _confounded()
    raised = oq.threshold_crossing(model, "Y", "1", "S", threshold=0.4)
    rows = {row.level: row for row in raised.threshold_rows}
    assert rows["0"].conditional == pytest.approx(0.46, abs=TOL)
    assert rows["1"].conditional == pytest.approx(0.36, abs=TOL)
    assert rows["0"].crosses and not rows["1"].crosses
    assert raised.crossing_probability == pytest.approx(0.5, abs=TOL)
    with pytest.raises(ObservationQueryRefusal) as caught:
        oq.threshold_crossing(model, "Y", "1", "S", threshold=0.5)
    assert caught.value.detail == "observation_query.level_unreachable"
    assert float(caught.value.supplied) == pytest.approx(0.46, abs=TOL)


def test_the_native_threshold_entry_reads_the_lowering_direction() -> None:
    model = _confounded()
    wire = {
        "outcome": "Y",
        "outcome_level": "1",
        "signal": "S",
        "background": [],
        "threshold": 0.4,
        "direction": "lower",
    }
    claim = oq._claim(_native.observation_query_threshold(oq._handle(model), json.dumps(wire)))
    rows = {row.level: row for row in claim.threshold_rows}
    # Lowering below 0.4: only S = 1 (0.36) crosses, with probability P(S=1) = 0.5.
    assert rows["1"].crosses and not rows["0"].crosses
    assert claim.crossing_probability == pytest.approx(0.5, abs=TOL)


# -------------------------------------------------------------- declared models


def test_finite_posterior_applies_bayes_rule_once() -> None:
    hypotheses, levels, likelihood = ILL_WELL
    claim = oq.finite_posterior(hypotheses, [0.5, 0.5], levels, likelihood, ["neg"])
    # ill 0.5 * 0.1 = 0.05, well 0.5 * 0.95 = 0.475, evidence 0.525, 2 / 21.
    assert claim.posterior["ill"] == pytest.approx(2.0 / 21.0, abs=TOL)
    assert claim.evidence_probability == pytest.approx(0.525, abs=TOL)


def test_the_native_finite_posterior_entry_combines_repeated_readings() -> None:
    hypotheses, levels, likelihood = ILL_WELL
    model = {
        "hypotheses": hypotheses,
        "prior": [0.01, 0.99],
        "observation_levels": levels,
        "likelihood": [p for row in likelihood for p in row],
    }
    claim = oq._claim(
        _native.observation_query_finite_posterior(
            json.dumps(model), ["pos", "pos"], "declared_model"
        )
    )
    # ill 0.01 * 0.81 = 0.0081, well 0.99 * 0.0025 = 0.002475, 36 / 47.
    assert claim.posterior["ill"] == pytest.approx(36.0 / 47.0, abs=TOL)
    assert claim.evidence_probability == pytest.approx(0.010575, abs=TOL)


def test_conjugate_posterior_adds_counts_to_the_beta_shapes() -> None:
    claim = oq.conjugate_posterior(2.0, 3.0, successes=4, failures=6)
    alpha, beta, predictive = claim.beta_posterior
    assert (alpha, beta) == (6.0, 9.0)
    assert predictive == pytest.approx(0.4, abs=TOL)


def test_the_native_conjugate_posterior_entry_returns_the_updated_shapes() -> None:
    claim = oq._claim(
        _native.observation_query_conjugate_posterior(1.0, 4.0, 3, 2, "declared_model")
    )
    alpha, beta, predictive = claim.beta_posterior
    assert (alpha, beta) == (4.0, 6.0)
    assert predictive == pytest.approx(0.4, abs=TOL)


# ------------------------------------------------------------ required evidence


def test_required_bayes_factor_is_target_odds_over_prior_odds() -> None:
    # 0.1 has odds 1/9 and 0.5 has odds 1: factor 9.
    assert oq.required_bayes_factor(0.1, 0.5).required_factor == pytest.approx(9.0, abs=1e-10)
    # Lowering 0.8 (odds 4) to 0.5 (odds 1) is a factor of 1/4.
    lowered = oq.required_bayes_factor(0.8, 0.5, direction="lower")
    assert lowered.required_factor == pytest.approx(0.25, abs=1e-10)


def test_the_native_required_bayes_factor_entry_returns_the_odds_ratio() -> None:
    # 0.2 has odds 1/4 and 0.6 has odds 3/2: factor 6.
    claim = oq._claim(_native.observation_query_required_bayes_factor(0.2, 0.6, "raise"))
    assert claim.required_factor == pytest.approx(6.0, abs=1e-10)


def test_required_binary_signal_readings_are_the_exact_minimum() -> None:
    claim = oq.required_observations_binary_signal(
        0.2, 0.9, rate_given_hypothesis=0.8, rate_given_complement=0.2
    )
    # Per-reading factor 4, prior odds 1/4, target odds 9: 4^n / 4 >= 9 at n = 3, 16 / 17.
    assert claim.required_observations == 3
    assert claim.value_at_requirement == pytest.approx(16.0 / 17.0, abs=TOL)
    with pytest.raises(ObservationQueryRefusal) as budget:
        oq.required_observations_binary_signal(
            0.2, 0.9, rate_given_hypothesis=0.8, rate_given_complement=0.2, max_observations=2
        )
    assert budget.value.reason_code == "evidence_cannot_reach_level"
    assert float(budget.value.supplied) == pytest.approx(0.8, abs=TOL)


def test_the_native_binary_signal_entry_solves_another_factor() -> None:
    claim = oq._claim(
        _native.observation_query_required_observations_binary(
            0.5, 0.95, "raise", 0.9, 0.3, 100, "declared_model"
        )
    )
    # Factor 3 per reading from even odds; target odds 19: 3^3 = 27 >= 19, 27 / 28.
    assert claim.required_observations == 3
    assert claim.value_at_requirement == pytest.approx(27.0 / 28.0, abs=TOL)


def test_required_conjugate_observations_match_the_predictive_closed_form() -> None:
    claim = oq.required_observations_conjugate(2.0, 2.0, 0.7)
    # (2 + n) / (4 + n) >= 0.7 first holds at n = 3, with value 5 / 7.
    assert claim.required_observations == 3
    assert claim.value_at_requirement == pytest.approx(5.0 / 7.0, abs=TOL)


def test_the_native_conjugate_entry_solves_another_shape() -> None:
    claim = oq._claim(
        _native.observation_query_required_observations_conjugate(
            1.0, 3.0, 0.6, "raise", 100, "declared_model"
        )
    )
    # (1 + n) / (4 + n) >= 0.6 first holds at n = 4, with value 5 / 8.
    assert claim.required_observations == 4
    assert claim.value_at_requirement == pytest.approx(0.625, abs=TOL)


# ---------------------------------------------------------------------- artifact


def test_consume_replays_a_posterior_claim_against_the_retained_identity() -> None:
    hypotheses, levels, likelihood = ILL_WELL
    claim = oq.finite_posterior(hypotheses, [0.5, 0.5], levels, likelihood, ["pos"])
    consumed = oq.consume_observation_query_artifact(bytes(claim.export()), claim.identity)
    assert consumed.identity == claim.identity
    assert consumed.operation == "posterior_update"
    # ill 0.5 * 0.9 = 0.45, well 0.5 * 0.05 = 0.025: 0.45 / 0.475 = 18 / 19.
    assert consumed.claim.posterior["ill"] == pytest.approx(18.0 / 19.0, abs=TOL)
    with pytest.raises(ObservationQueryRefusal) as wrong:
        oq.consume_observation_query_artifact(bytes(claim.export()), "f" * 64)
    assert wrong.value.detail == "observation_query.artifact_refused"


def test_the_native_consume_entry_returns_the_replayed_summary_or_a_refusal() -> None:
    claim = oq.conjugate_posterior(1.0, 1.0, successes=1, failures=0)
    raw, refusal = _native.observation_query_consume(bytes(claim.export()), claim.identity)
    assert refusal is None and raw is not None
    wire = json.loads(raw)
    assert wire["identity"] == claim.identity
    assert wire["operation"] == "posterior_update"
    raw, refusal = _native.observation_query_consume(bytes(claim.export()), "0" * 64)
    assert raw is None and refusal is not None
