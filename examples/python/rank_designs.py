"""Rank candidate experimental designs by identification probability.

See ADR 0016: ranking is advisory — it does not auto-rerun analyses.
"""

from __future__ import annotations

import antecedent

ranking = antecedent.rank_designs(
    [0.5, 0.3, 0.2],
    [1, 0, 0],
    [10, 20, 30],
    [
        {"kind": "measure", "variables": [3], "tag": 1},
        {"kind": "observe_environment", "environment": 7, "additional_rows": 50},
        {"kind": "increase_sampling_rate", "additional_samples": 10},
        {"kind": "intervene", "targets": [0]},
    ],
    objective="increase_identification_probability",
    query_id=0,
    query_id_unlock=[(0, [3])],
    env_id_unlock=[(0, [7])],
    min_batches=2,
    max_batches=4,
    batch_size=4,
    rank_uncertainty_threshold=1.0,
    seed=3,
)

print(f"best_index={ranking.best_index} mc_samples={ranking.mc_samples}")
for row in ranking.ranked:
    print(f"  candidate={row.candidate_index} kind={row.kind} score={row.score:.4f}")
