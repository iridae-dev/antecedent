"""Incremental CausalState: append batches, register query, refresh results.

State never auto-reruns analyses (ADR 0016). Call ``refresh_results`` after
appending data when you want updated summaries.
"""

from __future__ import annotations

import numpy as np

import causal

rng = np.random.default_rng(1)
state = causal.CausalState(cache_bytes=1 << 20)

n = 40
t = rng.normal(size=n)
y = 0.5 * t + rng.normal(size=n) * 0.1
ver = state.append_data(["t", "y"], [t, y])
print(f"version after append={ver}")

_, qid = state.register_average_effect(0, 1)
state.refresh_results([(qid, 1, 8)])
print(f"stale_queries={state.stale_query_count()} batches={len(state.batch_ids())}")

# Replace data → query becomes stale until refresh.
state.replace_data(["t", "y"], [t, y])
print(f"after replace stale={state.stale_query_count()}")
state.refresh_results([(qid, 1, 8)])
print(f"after refresh stale={state.stale_query_count()}")
