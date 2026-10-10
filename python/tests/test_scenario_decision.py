"""B: decisions over the claims of real 2.2 scenario sets and CPDAG completions.

Oracles are derived by hand from the laws, as in ``crates/antecedent/tests/decision_claims.rs``.

The scenarios share ``z -> x``, ``z -> y``, ``x -> y``, ``x <-> y`` and differ in the
mechanisms that may change. ``direct`` gives ``P(y = 1) = 0.08 + 0.48 = 0.56``;
``standardize`` gives ``0.75 * 0.2 + 0.25 * 0.8 = 0.35``; ``outcome_shift`` selects on ``y`` and
is not transportable; ``zeta`` is entered after the shared budget is spent and stays
unevaluated. Two actions read the mean of ``y``: ``treat`` earns ``y - cost`` and ``hold``
earns ``0.5 * y``.

* cost 0.2: treat is 0.36 / 0.15 and hold 0.28 / 0.175 under direct / standardize, so treat
  leads under ``direct``, hold under ``standardize``, and neither is invariant; the worst case
  favors hold (0.175 > 0.15);
* cost 0.05: treat is 0.51 / 0.30 against hold 0.28 / 0.175, so treat leads everywhere.

A CPDAG chain ``a - b - c`` has three completions; with ``c`` the outcome of ``do(b = 1)`` they
give 0.8, 0.8 and 0.38 (see ``test_cpdag_scenarios.py``).
"""

import json

import pytest
from antecedent import Admg, Cpdag, decision
from antecedent import scenario_decision as sd
from antecedent.decision_robust import Support, admissible_contract
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.joint_distribution import ScientificQuantity
from antecedent.transport import advanced as transport

from _refusal import assert_registered_refusal

NAMES = ["z", "x", "y"]
SOURCE = (0.32, 0.08, 0.12, 0.48)  # do(x=1) over (z, y)
TARGET = (0.3, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1)  # (z, x, y), P*(z=1) = 0.25
STANDARDIZED = 0.75 * 0.2 + 0.25 * 0.8
DIRECT = 0.08 + 0.48
CAUSAL = "causal-identification-1"
SUPPORTED = Support("supported")


def graph():
    return Admg.from_edges(NAMES, [("z", "x"), ("z", "y"), ("x", "y")], [("x", "y")])


def coordinates():
    return [transport.VariableCoordinate(n, "binary") for n in NAMES]


def scenario_set(weights=None, *, late=True):
    """standardize, direct, outcome_shift and (sorted last) zeta."""
    w = weights or (None, None, None, None)
    scenarios = [
        transport.TransportScenario("standardize", graph(), ["z"], w[0]),
        transport.TransportScenario("direct", graph(), [], w[1]),
        transport.TransportScenario("outcome_shift", graph(), ["y"], w[2]),
    ]
    if late:
        scenarios.append(transport.TransportScenario("zeta", graph(), ["z", "y"], w[3]))
    return transport.TransportScenarioSet(scenarios, coordinates())


def two_scenarios(weights=None):
    w = weights or (None, None)
    return transport.TransportScenarioSet(
        [
            transport.TransportScenario("standardize", graph(), ["z"], w[0]),
            transport.TransportScenario("direct", graph(), [], w[1]),
        ],
        coordinates(),
    )


def catalog():
    return transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["z", "y"]
            ),
            transport.EvidenceRegime("obs", "target", measured=["z", "x", "y"]),
        ]
    )


def laws():
    return transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "source",
                "trial",
                (("z", (0.0, 1.0)), ("y", (0.0, 1.0))),
                SOURCE,
                "trial",
                interventions=(("x", 1.0),),
            ),
            transport.ExactDiscreteLaw(
                "target",
                "obs",
                (("z", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
                TARGET,
                "target",
            ),
        )
    )


def prepare(sets, **limits):
    return transport.prepare_transport_scenarios(
        sets,
        outcomes=["y"],
        treatments=["x"],
        source="source",
        target="target",
        catalog=catalog(),
        laws=laws(),
        at={"x": 1.0},
        **limits,
    )


def estimated(sets, **limits):
    stage = prepare(sets, **limits)
    report = json.loads(stage.estimate())
    return stage, {s["name"]: s for s in report["scenarios"]}


def operations_for_three():
    """The fewest shared operations that decide direct, outcome_shift and standardize."""
    sets = scenario_set(late=False)
    for steps in range(1, 10_000):
        if json.loads(prepare(sets, max_steps=steps).estimate())["receipt"] is None:
            return steps
    raise AssertionError("small graphs decide within 10k operations")


def mixed(weights=None):
    """Two identified scenarios that disagree, one unidentified, one unevaluated."""
    stage, by_name = estimated(scenario_set(weights), max_steps=operations_for_three())
    assert by_name["direct"]["status"] == by_name["standardize"]["status"] == "identified"
    assert by_name["outcome_shift"]["status"] == "structurally_unidentified"
    assert by_name["zeta"]["status"] == "unevaluated"
    return stage


def two(weights=None):
    stage, by_name = estimated(two_scenarios(weights))
    assert by_name["direct"]["status"] == by_name["standardize"]["status"] == "identified"
    return stage


def quantity(variable="y", regime="do(x=1)"):
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units="units",
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="outcome",
    )


def contract(cost, policy, *, variable="y", regime="do(x=1)"):
    y = quantity(variable, regime)
    return decision.Contract(
        actions=(
            decision.Action("treat", inputs=(y,), utility=decision.x(0) - cost),
            decision.Action("hold", inputs=(y,), utility=0.5 * decision.x(0), kind="regime"),
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        structural_policy=policy,
    )


def decide(stage, cost=0.2, policy="report_only", weights=None, **kwargs):
    options = {
        "outcomes": [sd.OutcomeBinding("y", quantity())],
        "causal_contract_id": CAUSAL,
        "default_support": SUPPORTED,
    }
    options.update(kwargs)
    return sd.decide_from_scenarios(contract(cost, policy), stage, policy, weights, **options)


def near(value, expected):
    assert value == pytest.approx(expected, abs=1e-12)


# ------------------------------------------------------------------ the atom table


def test_b_claims_scenario_atoms_keep_status_weight_support_and_unrenormalized_mass():
    # standardize 0.2, direct 0.3, outcome_shift 0.25, zeta 0.15; 0.1 is declared to no scenario.
    weights = {"standardize": 0.2, "direct": 0.3, "outcome_shift": 0.25, "zeta": 0.15}
    decided = decide(mixed(tuple(weights.values())), weights=weights)

    assert decided.kind == "transport_scenarios" and decided.declared_weights
    statuses = {a.id: (a.status, a.kind) for a in decided.atoms}
    assert statuses == {
        "direct": ("identified", "evaluated"),
        "outcome_shift": ("structurally_unidentified", "unidentified"),
        "standardize": ("identified", "evaluated"),
        "zeta": ("unevaluated", "unevaluated"),
    }
    for atom in decided.atoms:
        near(atom.weight, weights[atom.id])
    # Identified scenarios carry the supplied support; the rest have none to assess.
    assert {a.id: a.support for a in decided.atoms} == {
        "direct": "supported",
        "outcome_shift": "missing_evidence",
        "standardize": "supported",
        "zeta": "missing_evidence",
    }
    assert decided.atom("zeta").detail and decided.atom("zeta").values == {}

    # The report's own masses are retained: 0.25 unidentified, 0.15 unevaluated, 0.1 undeclared.
    mass = {m.status: m for m in decided.masses}
    near(mass["structurally_unidentified"].mass, 0.25)
    near(mass["unevaluated"].mass, 0.15)
    near(decided.residual_mass, 0.1)

    # Nothing is renormalized over the two identified scenarios.
    near(decided.evaluated_mass, 0.5)
    near(decided.unidentified_mass, 0.25)
    near(decided.unevaluated_mass, 0.25)
    assert decided.verdict.kind == "report_only" and decided.selected is None

    # treat: direct 0.56 - 0.2 = 0.36, standardize 0.35 - 0.2 = 0.15; hold: 0.28 and 0.175.
    treat, hold = decided.action("treat"), decided.action("hold")
    near(treat.range[0], STANDARDIZED - 0.2)
    near(treat.range[1], DIRECT - 0.2)
    near(hold.range[0], 0.175)
    near(hold.range[1], 0.28)
    near(treat.per_atom["direct"], 0.36)
    near(hold.per_atom["standardize"], 0.175)
    assert treat.per_atom["outcome_shift"] is None and treat.per_atom["zeta"] is None
    # Weighted values sum over the evaluated mass only: 0.3 * 0.36 + 0.2 * 0.15 and so on.
    near(treat.weighted_value, 0.138)
    near(hold.weighted_value, 0.119)
    # Leaders are hand-derived: treat under direct, hold under standardize.
    assert dict(decided.leaders) == {"direct": ("treat",), "standardize": ("hold",)}
    assert decided.atom("direct").leaders == ("treat",)
    assert "not renormalized" in decided.explain()


def test_b_claims_unweighted_scenarios_give_unweighted_atoms():
    decided = decide(mixed())
    assert not decided.declared_weights
    assert all(a.weight is None for a in decided.atoms)
    assert decided.residual_mass is None
    assert decided.unidentified_mass is None and decided.unevaluated_mass is None
    assert all(m.mass is None for m in decided.masses)


def test_b_claims_policies_reach_the_hand_derived_choices():
    stage = two()
    # Cost 0.05: treat beats hold in both scenarios (0.51 > 0.28, 0.30 > 0.175).
    cheap = decide(stage, 0.05, "require_invariant_best_action")
    assert cheap.verdict.kind == "invariant_best" and cheap.selected == "treat"
    assert dict(cheap.leaders) == {"direct": ("treat",), "standardize": ("treat",)}

    # Cost 0.2: treat leads under direct (0.36 > 0.28), hold under standardize
    # (0.175 > 0.15), so nothing is invariant.
    split = decide(stage, 0.2, "require_invariant_best_action")
    assert split.verdict.kind == "no_invariant_best"
    assert split.verdict.leaders == (("direct", ("treat",)), ("standardize", ("hold",)))

    # Worst cases: treat min(0.36, 0.15) = 0.15 < hold min(0.28, 0.175) = 0.175.
    maximin = decide(stage, 0.2, "maximin")
    assert maximin.verdict.kind == "worst_case_choice" and maximin.selected == "hold"

    assert decide(stage, 0.2, "report_only").verdict.kind == "report_only"


def test_b_claims_unresolved_scenarios_leave_invariance_and_worst_case_unchecked():
    stage = mixed()
    for policy in ("require_invariant_best_action", "maximin"):
        decided = decide(stage, 0.05, policy)
        assert decided.verdict.kind == "insufficient_science"
        assert decided.selected is None
        # The two identified scenarios still name their leaders.
        assert dict(decided.leaders) == {"direct": ("treat",), "standardize": ("treat",)}


# ------------------------------------------------------------------ Bayes and weights


def test_b_claims_bayes_is_refused_without_declared_weights():
    with pytest.raises(sd.ScenarioDecisionRefusal) as refused:
        decide(two(), 0.2, "bayes_over_structures")
    assert refused.value.detail == "decision_claims.probabilities_not_declared"
    assert refused.value.reason_code == "decision_contract_unsatisfied"
    assert refused.value.supplied == "none"
    assert isinstance(refused.value, CausalUnsupportedError)
    assert_registered_refusal(refused.value)


def test_b_claims_bayes_reads_declared_weights_and_unresolved_mass_blocks_a_choice():
    # Weights 1/2 and 1/2: treat 0.5 * 0.36 + 0.5 * 0.15 = 0.255, hold 0.5 * 0.28 + 0.5 * 0.175
    # = 0.2275, so treat; every mass is evaluated.
    weights = {"standardize": 0.5, "direct": 0.5}
    decided = decide(two((0.5, 0.5)), 0.2, "bayes_over_structures", weights)
    assert decided.verdict.kind == "bayes_choice" and decided.selected == "treat"
    near(decided.verdict.evaluated_mass, 1.0)
    near(decided.action("treat").weighted_value, 0.255)
    near(decided.action("hold").weighted_value, 0.2275)

    # With unresolved scenarios the declared weights are still read, but mass that can
    # change the Bayes action blocks the choice.
    mixed_weights = {"standardize": 0.2, "direct": 0.3, "outcome_shift": 0.25, "zeta": 0.15}
    blocked = decide(
        mixed(tuple(mixed_weights.values())), 0.2, "bayes_over_structures", mixed_weights
    )
    assert blocked.verdict.kind == "insufficient_science"
    near(blocked.unidentified_mass, 0.25)


def test_b_claims_weights_must_be_the_scenario_sets_own():
    stage = two((0.5, 0.5))
    # Weights the set did not declare are not applied to it.
    with pytest.raises(sd.ScenarioDecisionRefusal) as other:
        decide(stage, 0.2, "bayes_over_structures", {"standardize": 0.9, "direct": 0.1})
    assert other.value.detail == "decision_claims.identity_mismatch"
    assert other.value.offending == "weights"
    # An unweighted set cannot be weighted from outside.
    with pytest.raises(sd.ScenarioDecisionRefusal) as undeclared:
        decide(two(), 0.2, "bayes_over_structures", {"standardize": 0.5, "direct": 0.5})
    assert undeclared.value.detail == "decision_claims.unsupported_claim_shape"


def test_b_claims_a_policy_or_causal_identity_other_than_the_contracts_is_refused():
    stage = two()
    contract_maximin = contract(0.2, "maximin")
    with pytest.raises(sd.ScenarioDecisionRefusal) as policy:
        sd.decide_from_scenarios(
            contract_maximin,
            stage,
            "report_only",
            outcomes=[sd.OutcomeBinding("y", quantity())],
            causal_contract_id=CAUSAL,
        )
    assert policy.value.detail == "decision_adapters.policy_mismatch"
    with pytest.raises(sd.ScenarioDecisionRefusal) as foreign:
        decide(stage, expected_causal_contract_id="another-identification")
    assert foreign.value.detail == "decision_claims.identity_mismatch"
    assert foreign.value.offending == "causal_contract_id"


def test_b_claims_the_result_must_be_an_estimated_stage():
    with pytest.raises(CausalTypeError):
        decide({"scenarios": []})
    with pytest.raises(CausalUnsupportedError) as unestimated:
        decide(prepare(two_scenarios()))
    assert_registered_refusal(unestimated.value)
    with pytest.raises(CausalValueError):
        decide(two(), policy="always_treat")
    with pytest.raises(sd.ScenarioDecisionRefusal):
        sd.decide_from_cpdag_completions(
            contract(0.2, "report_only"),
            object(),
            "report_only",
            outcomes=[sd.OutcomeBinding("y", quantity())],
            causal_contract_id=CAUSAL,
        )


def test_b_claims_the_contract_may_be_admissible_or_plain_and_digests_are_retained():
    stage = two()
    plain = decide(stage, 0.2, "report_only")
    wrapped = sd.decide_from_scenarios(
        admissible_contract(contract(0.2, "report_only")),
        stage,
        "report_only",
        outcomes=[sd.OutcomeBinding("y", quantity())],
        causal_contract_id=CAUSAL,
        default_support=SUPPORTED,
        premises_digest="premises-1",
        data_digest="data-1",
    )
    assert (wrapped.premises_digest, wrapped.data_digest) == ("premises-1", "data-1")
    assert plain.premises_digest == plain.data_digest and len(plain.premises_digest) == 64
    assert wrapped.source_identity != plain.source_identity
    assert {a.id: a.digest for a in wrapped.atoms} == {a.id: a.digest for a in plain.atoms}


# ------------------------------------------------------------------ CPDAG completions

CPDAG_NAMES = ["a", "b", "c"]
FORWARD = frozenset({("a", "b"), ("b", "c")})
FORK = frozenset({("b", "a"), ("b", "c")})
BACKWARD = frozenset({("b", "a"), ("c", "b")})


def joint(a, b, c):
    def p(one, p_one):
        return p_one if one else 1.0 - p_one

    return p(a == 1, 0.4) * p(b == 1, (0.2, 0.7)[a]) * p(c == 1, (0.1, 0.8)[b])


CPDAG_TABLE = tuple(joint(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))


def completion_run():
    axes = tuple((n, (0.0, 1.0)) for n in CPDAG_NAMES)
    return transport.cpdag_completion_scenarios(
        Cpdag.from_directed_undirected(CPDAG_NAMES, [], [("a", "b"), ("b", "c")]),
        outcomes=["c"],
        treatments=["b"],
        source="src_pop",
        target="target",
        coordinates=[transport.VariableCoordinate(n, "binary") for n in CPDAG_NAMES],
        evidence=transport.CompletionEvidence(
            transport.EvidenceCatalog(
                regimes=[transport.EvidenceRegime("obs", "target", measured=CPDAG_NAMES)]
            ),
            "target-law",
        ),
        laws=transport.ExactTransportData(
            (transport.ExactDiscreteLaw("target", "obs", axes, CPDAG_TABLE, "target"),)
        ),
        at={"b": 1.0},
    )


def decide_completions(result, policy="report_only", weights=None, **kwargs):
    options = {
        "outcomes": [sd.OutcomeBinding("c", quantity("c", "do(b=1)"))],
        "causal_contract_id": CAUSAL,
        "default_support": SUPPORTED,
    }
    options.update(kwargs)
    return sd.decide_from_cpdag_completions(
        contract(0.2, policy, variable="c", regime="do(b=1)"), result, policy, weights, **options
    )


def test_b_claims_cpdag_atoms_are_graph_dependent_and_unweighted():
    result = completion_run()
    ids = {frozenset(c.edges): c.id for c in result.completions}
    decided = decide_completions(result)
    assert decided.kind == "cpdag_completions" and not decided.declared_weights
    assert {a.id for a in decided.atoms} == set(ids.values())
    assert all(a.weight is None and a.status == "identified" for a in decided.atoms)
    assert decided.completion_counts == {}
    assert decided.unidentified_mass is None and decided.residual_mass is None
    # Each completion names its own leader: treat 0.8 - 0.2 = 0.6 > 0.4 under the first two,
    # and hold 0.19 > 0.38 - 0.2 = 0.18 under the third.
    assert dict(decided.leaders) == {
        ids[FORWARD]: ("treat",),
        ids[FORK]: ("treat",),
        ids[BACKWARD]: ("hold",),
    }
    near(decided.action("treat").per_atom[ids[FORWARD]], 0.6)
    near(decided.action("treat").per_atom[ids[BACKWARD]], 0.18)
    near(decided.action("hold").per_atom[ids[BACKWARD]], 0.19)
    near(decided.action("treat").range[0], 0.18)
    assert decided.action("treat").weighted_value is None

    # With a policy that needs no probability the same atoms give the worst case:
    # treat min 0.18 < hold min 0.19.
    maximin = decide_completions(result, "maximin")
    assert maximin.verdict.kind == "worst_case_choice" and maximin.selected == "hold"


def test_b_claims_cpdag_completion_counts_are_never_probabilities():
    result = completion_run()
    with pytest.raises(sd.ScenarioDecisionRefusal) as bayes:
        decide_completions(result, "bayes_over_structures")
    assert bayes.value.detail == "decision_claims.probabilities_not_declared"
    assert bayes.value.supplied == "completion counts (never probabilities)"
    # Weights cannot be declared for completions either.
    with pytest.raises(sd.ScenarioDecisionRefusal) as weighted:
        decide_completions(result, weights={c.id: 1 / 3 for c in result.completions})
    assert weighted.value.detail == "decision_claims.unsupported_claim_shape"
    assert_registered_refusal(weighted.value)
