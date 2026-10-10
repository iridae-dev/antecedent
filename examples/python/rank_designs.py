"""Compare experiments by how much they could help identify an effect.

With no decision declared, ``rank_designs`` ranks typed plans on the "identification" basis.
The ranking depends on the planning assumptions you supply. It recommends
an action but does not run it. After choosing, use analyze(...) for the
causal question and keep result.study if you need to estimate again.
See rank_designs_evsi.py for ranking by value of information instead."""

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
