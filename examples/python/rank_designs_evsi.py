"""Which study is worth its cost for the decision it would inform?

No data are used. The decision is a bet on a coin whose bias is 1/4 or 3/4 with equal prior
weight, and the candidate studies are two or four flips of that coin at declared costs.
Declaring a decision switches ``rank_designs`` to the "net_value" basis: each study is valued by
its expected value of sample information (EVSI), net of its cost under an explicit cost map. The
script asserts the basis, that the two-flip EVSI does not exceed the expected value of perfect
information (EVPI), and that an independent consumer recomputes the exported ranking against the
identity retained from it.

The values are conditional on the declared prior, utilities and costs; no study is run.

Run: ``python examples/python/rank_designs_evsi.py``."""

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


# Bet on a coin whose bias theta is 1/4 or 3/4 with equal prior weight: the bet pays
# theta - 1/2 and abstaining pays 0, so at the prior both actions are worth 0 and information
# about theta has value.
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

# A consumer holding only the bytes and the retained identity rebuilds each binomial signal and
# recomputes its value.
consumed = design.consume(ranking.export(), expected_identity=ranking.expectation())
assert consumed.identity == ranking.identity
assert all(entry.natively_replayed for entry in consumed.entries)
