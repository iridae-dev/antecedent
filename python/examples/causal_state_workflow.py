"""CausalState as the primary online append path (ADR 0016 / backlog B).

Interactive products should prefer this over fresh ``analyze()`` on every batch:

1. Append (or replace) data — state versions and marks registered queries stale.
2. Call ``refresh_results`` under the ``CacheBudget`` — never auto-reruns on
   ``append`` / ``apply`` (ADR 0016).
3. UI code must key on ``version`` / stale queries so it never mixes an old
   identification summary with a new estimate.

For full re-estimate on a fixed graph/query with a new table (same schema), use
``causal.PreparedAnalysis`` instead — compile once, ``estimate`` / ``refresh``
many times.
"""

from __future__ import annotations

import numpy as np

import antecedent

rng = np.random.default_rng(1)
# Bound retained result bytes; over-budget refresh refuses instead of silent drop.
state = causal.CausalState(cache_bytes=1 << 20)

n = 40
t = rng.normal(size=n)
y = 0.5 * t + rng.normal(size=n) * 0.1
ver = state.append_data(["t", "y"], [t, y])
print(f"version after append={ver}")

# Register a query; refresh stores a versioned fingerprint (does not run estimators).
_, qid = state.register_average_effect(0, 1)
state.refresh_results([(qid, 1, 8)])
print(f"stale_queries={state.stale_query_count()} batches={len(state.batch_ids())}")

# Incremental OLS: append rows, then compare to a full recompute on the same design.
state.ols_ensure("m1", 2)
xs = [[1.0, float(ti)] for ti in t]
for x_row, yi in zip(xs, y):
    state.ols_append_row("m1", x_row, float(yi))
ols = state.ols_get("m1")
print(f"ols n={ols['n']} ncols={ols['ncols']}")

# Replace data → registered query becomes stale until explicit refresh.
state.replace_data(["t", "y"], [t, y])
print(f"after replace stale={state.stale_query_count()}")
state.refresh_results([(qid, 1, 8)])
print(f"after refresh stale={state.stale_query_count()} version={state.version}")
