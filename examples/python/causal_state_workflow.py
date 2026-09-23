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
assert state.stale_query_count() == 0  # an explicit refresh clears staleness
assert len(state.batch_ids()) == 1

# Incremental OLS: append rows, then compare to a full recompute on the same design.
state.ols_ensure("m1", 2)
xs = [[1.0, float(ti)] for ti in t]
for x_row, yi in zip(xs, y):
    state.ols_append_row("m1", x_row, float(yi))
ols = state.ols_get("m1")
print(f"ols n={ols['n']} ncols={ols['ncols']}")
# The retained normal equations equal the full recompute on the same design.
design = np.column_stack([np.ones(n), t])
assert ols["n"] == n and ols["ncols"] == 2
assert np.allclose(np.asarray(ols["xtx"]).reshape(2, 2), design.T @ design, rtol=0, atol=1e-9)
assert np.allclose(np.asarray(ols["xty"]), design.T @ y, rtol=0, atol=1e-9)

# Replace data → registered query becomes stale until explicit refresh.
state.replace_data(["t", "y"], [t, y])
print(f"after replace stale={state.stale_query_count()}")
assert state.stale_query_count() >= 1  # replacing data stales the registered query
state.refresh_results([(qid, 1, 8)])
print(f"after refresh stale={state.stale_query_count()} version={state.version}")
assert state.stale_query_count() == 0
