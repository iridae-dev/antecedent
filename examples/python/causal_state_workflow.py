"""Which registered results are out of date after the data change?

CausalState tracks staleness; it never re-estimates. Capture the state version before
computing, then publish with refresh_results_at so a result computed on older data is refused.
The script appends 40 simulated rows (y = 0.5 t + noise), registers an average-effect query,
and asserts that replacing the data marks the query stale until an explicit refresh, and that
the retained least-squares normal equations equal a full recompute. No effect is estimated.

For a full re-estimate on a fixed graph and question, retain result.study from analyze(...)
and call antecedent.estimate(antecedent.rebind(study, data=new_data)) instead.

Run: python examples/python/causal_state_workflow.py"""

from __future__ import annotations

import antecedent
import numpy as np

rng = np.random.default_rng(1)
# The execution's cache policy bounds the bytes of retained results (1 MiB here).
state = antecedent.state.CausalState(
    execution=antecedent.Execution(cache=antecedent.CachePolicy.bounded(1 << 20))
)

n = 40
t = rng.normal(size=n)
y = 0.5 * t + rng.normal(size=n) * 0.1
ver = state.append_data(["t", "y"], [t, y])
print(f"version after append={ver}")

# Register a query. A refresh records a result's fingerprint and size against the state
# version it was computed at; it runs no estimator. Each entry is (query, fingerprint, bytes).
_, qid = state.register_average_effect(0, 1)
state.refresh_results_at(state.version, [(qid, 1, 8)])
print(f"stale_queries={state.stale_query_count()} batches={len(state.batch_ids())}")
assert state.stale_query_count() == 0  # an explicit refresh clears staleness
assert len(state.batch_ids()) == 1

# Retained least squares: rows update the normal equations one at a time.
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

# Replacing the data makes the registered query stale until an explicit refresh.
state.replace_data(["t", "y"], [t, y])
print(f"after replace stale={state.stale_query_count()}")
assert state.stale_query_count() >= 1  # replacing data stales the registered query
state.refresh_results_at(state.version, [(qid, 1, 8)])
print(f"after refresh stale={state.stale_query_count()} version={state.version}")
assert state.stale_query_count() == 0
