"""Update an analysis as new data arrive.

CausalState tracks which results are out of date after data change.
Appending data does not rerun queries: call refresh_results explicitly and
check the version before displaying results together.

For a full re-estimate on a fixed graph and question, retain result.study
from analyze(...) and call study.refresh(new_data) instead."""

from __future__ import annotations

import antecedent
import numpy as np

rng = np.random.default_rng(1)
# Bound retained result bytes; over-budget refresh refuses instead of silent drop.
state = antecedent.state.CausalState(cache_bytes=1 << 20)

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
