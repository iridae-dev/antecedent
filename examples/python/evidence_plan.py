#!/usr/bin/env python3
"""Plan which evidence to collect when an effect is not identified from what you have.

Question: a registry records treatment ``t`` and outcome ``y`` but not the confounder ``z`` that
drives both. The effect of ``t`` is not identified from those two columns. Which declared study
would close the gap, in what order, and what is each worth for a decision once its cost is
counted?

Data: simulated from a declared mechanism. ``z ~ N(0, 1)``; ``t = 1`` when ``z`` plus standard
noise is positive; ``y = 1 + 2 t + z + noise``. The average effect of ``t`` is 2. The naive
difference in means is biased upward by the confounder; adjusting for ``z`` recovers 2.

Decision model for the value step: a stand-in two-point decision declared by hand. A rate
``theta`` is 1/4 or 3/4 with equal prior weight; betting pays ``theta - 1/2`` and abstaining
pays 0. A study's signal is a binomial count with its declared sample size. The value of the
information is computed here by hand and compared with the plan.

The assertions check: the unadjusted data are refused as not identified, and the naive
difference is far from 2; a study that measures ``z`` is verified sufficient and a study that
adds rows of ``t`` and ``y`` is not; without a value model the best verified option is the
cheapest one; with one, the plan's expected value of sample information equals the hand
computation, the best option equals the hand-computed maximum of value minus cost, and a dearer
cost map reverses the order; the plan replays from its bytes; after collecting ``z`` the
adjusted estimate is near 2; and the structural ordering equals a hand sort.

What this does not claim: a plan is hypothetical identification, not an estimate. It does not
establish that the graph is true, its ordering is point-only with calibration not measured,
and the value model is a declared stand-in, not something learned from the registry.
"""

from __future__ import annotations

# %%
from math import comb

import antecedent as ant
import numpy as np
from antecedent import design, repair
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

N = 2000
TRUE_EFFECT = 2.0
TOLERANCE = 0.15
THETAS = (0.25, 0.75)
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]
QUERY = ant.AverageEffect("t", "y")


def simulate(seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=N)
    t = (z + rng.normal(size=N) > 0).astype(float)
    y = 1.0 + TRUE_EFFECT * t + z + rng.normal(scale=0.3, size=N)
    return {"z": z, "t": t, "y": y}


def evsi_by_hand(n: int) -> float:
    """EVSI of ``n`` binomial draws when betting pays ``theta - 1/2`` (no-information value 0)."""
    total = 0.0
    for k in range(n + 1):
        joint_gain = sum(
            0.5 * comb(n, k) * th**k * (1 - th) ** (n - k) * (th - 0.5) for th in THETAS
        )
        total += max(0.0, joint_gain)  # bet on this count only if it raises expected payoff
    return total


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


# %% [markdown]
# ## The registry does not identify the effect
#
# The registry has `t` and `y` only. Declaring that they share an unmeasured cause makes the
# average effect not identified, and `analyze` returns that as a typed refusal. The naive
# contrast is printed beside the truth to show what a careless reading would claim.

# %%
data = simulate()
registry = {"t": data["t"], "y": data["y"]}
bow = ant.Admg.from_edges(["t", "y"], [("t", "y")], [("t", "y")])
refused = ant.analyze(registry, graph=bow, query=QUERY)
assert not refused.accepted and refused.refusal is not None
print("refused:", refused.refusal.code, "|", refused.refusal.detail)
assert refused.refusal.code == "effect_not_identified"

naive = data["y"][data["t"] == 1].mean() - data["y"][data["t"] == 0].mean()
print(f"naive difference {naive:.2f} (truth {TRUE_EFFECT})")
assert naive > TRUE_EFFECT + 0.5


# %% [markdown]
# ## Declare the failure and the candidate studies
#
# The failure contract names the graph, the question and what is observed today. Cost, sample
# size, recruitment and timing of a study are facts only you know, so a template from the
# contract cannot be used until you complete it. Costs share one unit.

# %%
graph = ant.Dag.from_edges(["t", "y", "z"], [("t", "y"), ("z", "t"), ("z", "y")])
contract = repair.FailureContract(
    graph=graph, treatment="t", outcome="y", population="study", observed=["t", "y"]
)
template = next(
    t
    for t in repair.candidate_templates(contract)
    if t.kind == "observation" and set(t.measured) == {"t", "y", "z"}
)


def completed(label: str, cost: int, n: int):
    return template.complete(
        sample_size=n,
        recruitment="consecutive recruitment",
        timing="baseline",
        unit="patient",
        cost=cost,
        cost_unit="usd",
        sample_budget=n,
        feasible=True,
        label=label,
    )


cheap = completed("cheap cohort", cost=1, n=2)
large = completed("large cohort", cost=3, n=4)
blind = repair.StudyCandidate.observation(
    "more registry rows",
    population="study",
    measured=["t", "y"],
    cost=1,
    sample_size=4,
    recruitment="consecutive recruitment",
    timing="baseline",
    unit="patient",
    cost_unit="usd",
)
print("template measures:", sorted(template.measured), "| kind:", template.kind)


# %% [markdown]
# ## Order by identification first
#
# Each route's own theorem checker says which studies would identify the effect. A study that
# measures `z` closes the back door; more rows of `t` and `y` cannot. Without a value model
# the verified options are ordered by declared cost.

# %%
plan = design.evidence_plan(contract, [cheap, large, blind])
print(plan.explain())
assert plan.outcome == "closing"
assert plan.best is not None and plan.best.gate == "verified_sufficient"

verified = [o for o in plan.options if o.gate == "verified_sufficient"]
assert {o.labels[0] for o in verified} == {"cheap cohort", "large cohort"}
assert plan.best.labels[0] == min(verified, key=lambda o: o.cost).labels[0] == "cheap cohort"
rows_only = [o for o in plan.options if o.labels == ("more registry rows",)]
assert rows_only and all(o.gate != "verified_sufficient" for o in rows_only)
print("unlocks (verified):", plan.unlocks.variables, plan.unlocks.verified)
assert plan.unlocks.variables == ("z",)
print("ranking guarantee:", plan.ranking_guarantee, "| calibration:", plan.calibration)
assert (plan.ranking_guarantee, plan.calibration) == ("point_only", "not_measured")

replayed = design.consume_plan(bytes(plan.export()))
assert replayed.identity == plan.identity
assert [o.labels for o in replayed.options] == [o.labels for o in plan.options]


# %% [markdown]
# ## Value the studies net of cost
#
# With a decision, a signal for each study and a `CostMap` from cost to utility, options of the
# same gate are ordered by expected value of sample information net of cost. The hand
# computation above gives the same EVSI, and a dearer cost map reverses the order.

# %%
bet = design.DesignDecision(
    contract="contract-bet",
    actions=(design.ActionUtility("abstain", 0.0, 0.0), design.ActionUtility("bet", -0.5, 1.0)),
    prior=design.StatePrior.draws(list(THETAS)),
    utility_units="utility",
)
signal = design.SignalSpec(
    prior_id="prior-1",
    state=quantity("state"),
    observation=quantity("signal"),
    evidence_lineage=("snapshot:a",),
    rng_seed=3,
)


def valued_plan(utility_per_cost: float):
    value = design.EvidenceValue(
        decision=bet,
        studies={
            "cheap cohort": design.StudyValue(design.BinomialSignal(), signal),
            "large cohort": design.StudyValue(design.BinomialSignal(), signal),
        },
        cost_map=design.CostMap("usd", "utility", utility_per_cost),
        rng_seed=5,
    )
    return design.evidence_plan(contract, [cheap, large, blind], value=value)


for per_cost in (0.01, 0.02):
    valued = valued_plan(per_cost)
    ordered = [o for o in valued.options if o.gate == "verified_sufficient"]
    hand = {"cheap cohort": (2, 1), "large cohort": (4, 3)}  # label -> (sample size, cost)
    net = {label: evsi_by_hand(n) - per_cost * c for label, (n, c) in hand.items()}
    for option in ordered:
        label = option.labels[0]
        assert abs(option.value.evsi - evsi_by_hand(hand[label][0])) < 1e-9, label
        assert abs(option.value.net_value - net[label]) < 1e-9, label
        assert abs(option.value.evpi - 0.125) < 1e-9
    expected_best = max(net, key=net.get)
    print(
        f"utility per cost {per_cost}: net values "
        f"{ {k: round(v, 4) for k, v in net.items()} } -> best {valued.best.labels[0]}"
    )
    assert valued.basis == "net_value"
    assert valued.best.labels[0] == expected_best
    assert [o.labels[0] for o in ordered] == sorted(net, key=net.get, reverse=True)
    last = valued
assert valued_plan(0.01).best.labels[0] == "large cohort"
assert last.best.labels[0] == "cheap cohort"  # a dearer cost map reverses the order


# %% [markdown]
# ## Collect the evidence and estimate
#
# Suppose the plan's first choice is run and `z` is recorded. With it the graph identifies the
# effect and the adjusted estimate is compared with the declared truth.

# %%
result = require_success(
    ant.analyze(
        data,
        graph=GRAPH,
        query=QUERY,
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
        execution=ant.Execution(seed=3),
    )
)
print(f"adjusted effect {result.answer.value:.3f} (truth {TRUE_EFFECT})")
print("calibration:", result.calibration.status)
assert abs(result.answer.value - TRUE_EFFECT) < TOLERANCE


# %% [markdown]
# ## Bind the ranking to a prepared study, and read the structural view
#
# `rank_studies` takes a prepared study and a structure prior; only studies the repair search
# verified are scored, by the identified mass they unlock. `rank_structural` orders by
# declared sufficiency, then cost, budget and id, with no probability, so a hand sort must
# agree. A repeated id is refused with a typed reason.

# %%
study = ant.prepare(
    data,
    graph=GRAPH,
    query=QUERY,
    refute="none",
    inference=ant.Frequentist(bootstrap=0),
    execution=ant.Execution(seed=3),
)
prior = design.StructurePrior((1.0, 1.0), (True, False), (1, 2))
ranking = design.rank_studies(study, contract, [blind, cheap, large], prior, route="backdoor")
print("verified:", ranking.verified, "| unlock:", ranking.unlock)
print("unverified:", [(u.label, u.classification) for u in ranking.unverified])
assert ranking.verified == (1, 2) and ranking.unlock == ("z",)
assert [u.label for u in ranking.unverified] == ["more registry rows"]
# Unlocking z identifies the structure that was not: identified mass 1/2 -> 1, a gain of 1/2.
assert all(abs(r.score - 0.5) < 1e-9 for r in ranking.ranked)
assert ranking.calibration == "not_measured"

candidates = [
    design.StructuralCandidate(
        o.labels[0], o.gate == "verified_sufficient", o.cost, o.sample_budget
    )
    for o in plan.options
    if o.route == "backdoor"
]
by_hand = sorted(
    candidates, key=lambda c: (not c.verified_sufficient, c.cost_units, c.sample_budget, c.id)
)
structural = design.rank_structural(candidates)
print("structural order:", [e.id for e in structural.entries])
assert [e.id for e in structural.entries] == [c.id for c in by_hand]

try:
    design.rank_structural(
        [design.StructuralCandidate("a", True, 1), design.StructuralCandidate("a", False, 2)]
    )
except design.DesignRankingRefusal as refusal:
    print("refused:", refusal.reason_code, "|", refusal.detail)
    assert refusal.reason_code and refusal.detail
else:
    raise AssertionError("a repeated candidate id should be refused")
