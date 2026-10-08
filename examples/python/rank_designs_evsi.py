"""Rank candidate studies by the expected value of sample information (EVSI).

Declaring a decision switches ``rank_designs`` to the "net_value" basis: each study is valued
for the decision it would inform, net of its cost under an explicit cost map. The exported
artifact is replayed by an independent consumer against identities you retained."""

from __future__ import annotations

from antecedent import design
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


# Bet on a coin whose bias is 1/4 or 3/4 with equal prior weight: the bet pays theta - 1/2.
decision = design.DesignDecision(
    contract="contract-bet",
    actions=(design.ActionUtility("abstain", 0.0, 0.0), design.ActionUtility("bet", -0.5, 1.0)),
    prior=design.StatePrior.draws([0.25, 0.75]),
    utility_units="utility",
)
signal = design.SignalSpec(
    prior_id="prior-1",
    state=quantity("state"),
    observation=quantity("flips"),
    evidence_lineage=("snapshot:example",),
    rng_seed=3,
)
studies = [
    design.Candidate("two-flips", 2, design.BinomialSignal(), cost=0.02),
    design.Candidate("four-flips", 4, design.BinomialSignal(), cost=0.10),
]

ranking = design.rank_designs(
    studies,
    decision=decision,
    signal=signal,
    cost_map=design.CostMap("utility", "utility", 1.0),
)

print(ranking)
print(ranking.explain())
assert ranking.basis == "net_value"
assert ranking.candidate("two-flips").evsi <= ranking.evpi + 1e-12

# Replay the exported artifact against the identities retained from the ranking.
consumed = design.consume(ranking.export(), expected=ranking.expectation())
assert consumed.identity == ranking.identity
assert all(entry.natively_replayed for entry in consumed.entries)
