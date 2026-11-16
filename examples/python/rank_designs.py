"""Which candidate study would most raise the probability that an effect is identified?

No data are used. A structure prior puts weights 0.5, 0.3 and 0.2 on three candidate
structures, and the query is identified only in the first; ``variable_unlocks`` and
``environment_unlocks`` declare which measurement and which environment would identify it.
With no decision declared, ``rank_designs`` ranks the four typed plans on the "identification"
basis by Monte Carlo. The script asserts the basis, that every candidate is ranked and that a
best plan is named.

The order holds only under the declared prior and unlock assumptions, and the ranking runs no
study. After choosing, use ``analyze(...)`` for the causal question. See rank_designs_evsi.py for
ranking by value of information instead.

Run: ``python examples/python/rank_designs.py``."""

from __future__ import annotations

import antecedent
from antecedent import design

ranking = design.rank_designs(
    [
        design.Measurement([3], tag=1),
        design.Environment(7, additional_rows=50),
        design.Sampling(10),
        design.Experiment([0]),
    ],
    prior=design.StructurePrior(
        weights=(0.5, 0.3, 0.2), identified=(True, False, False), keys=(10, 20, 30)
    ),
    query_id=0,
    variable_unlocks={0: [3]},
    environment_unlocks={0: [7]},
    monte_carlo=design.MonteCarlo(
        min_batches=2, max_batches=4, batch_size=4, rank_uncertainty_threshold=1.0
    ),
    rng_seed=3,
)

print(ranking)
print(ranking.explain())
assert ranking.basis == "identification"
assert len(ranking.candidates) == 4  # every candidate is ranked
assert ranking.best is not None
assert ranking.mc_samples > 0
for row in ranking.candidates:
    print(f"  candidate={row.index} kind={row.kind} score={row.score:.4f}")
assert antecedent.design is design
