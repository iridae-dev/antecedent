# Keep results current: retained refresh and state publication

Two pieces keep a long-lived analysis honest as data change. A **retained refresh** keeps the
least-squares, scatter or conjugate summary a prepared study's estimator rests on, and updates it
as rows arrive, leave or change weight, checking every change against a fresh reconstruction.
**State publication** records which results a data or premise event made stale and accepts a
recomputed result only through the identities of the prepared study that recomputed it. Neither
re-identifies or re-estimates anything on its own.

## Retained summary refresh

Start a refresh from a prepared study. The study's estimator decides the summary family:
`linear.adjustment.ate` keeps `ols`, `aipw` keeps `weighted_ols` and `bayesian.gcomp` under a
conjugate backend keeps `conjugate_blocks`.

```python
import numpy as np
import antecedent as ant
from antecedent.state import RefreshMutation, RefreshObservation, RetainedRefresh

t = np.tile([0.0, 1.0], 32)
z = np.repeat(np.linspace(-1.0, 1.0, 32), 2)
study = ant.prepare(
    {"t": t, "z": z, "y": 2.0 * t + z},
    graph=ant.Dag.from_edges(["t", "z", "y"], [("z", "t"), ("z", "y"), ("t", "y")]),
    query=ant.AverageEffect("t", "y"),
    estimator="linear.adjustment.ate",
    refute=False,
)
print(RetainedRefresh.family_of(study))                        # ols


def row(i: int) -> RefreshObservation:
    x = float(i % 7) / 3.0
    return RefreshObservation(i, (1.0, x), 1.0 + 2.0 * x + 0.1 * np.sin(1.7 * i))


refresh = RetainedRefresh.start(study, source="batch-0", width=2, rows=[row(i) for i in range(20)])
print(refresh.result()["beta"])

print(refresh.apply(RefreshMutation.append("batch-0", [row(i) for i in range(20, 30)])))  # updated
print(refresh.apply(RefreshMutation.window(15)))                                          # recomputed
print(refresh.apply(RefreshMutation.evict(20)))                                           # recomputed
refresh.verify()                           # cached result equals a fresh rebuild from the rows
```

An append updates the summary in place; a window, eviction, weight, mask or replacement
recomputes it from the retained rows. A refused change (an unknown source, a duplicate row id, a
wrong width, an unknown row) raises and leaves the summary exactly as it was.
`refresh.fresh_result()` rebuilds the result from the retained rows alone, which is what `verify()`
compares against.

## Say what you retain

A refresh that keeps every row can refit from them; one that keeps only the summary cannot, and
names what it no longer serves.

```python
from antecedent.errors import CausalStateError
from antecedent.state import RetentionPolicy

policy = RetentionPolicy.sufficient_statistics()
print(policy.result_mode, policy.no_longer_serves)
try:
    policy.require("refit_from_retained_rows")
except CausalStateError as error:
    print(error)                                   # summary-only cannot serve refit ...

summary = refresh.with_retention(policy)           # or refresh.summary_only()
print(summary.result()["beta"] == refresh.result()["beta"])
```

`RetentionPolicy.raw_history()` keeps every row and `RetentionPolicy.bounded_window(n)` keeps the
trailing `n`. A summary-only refresh refuses every change that needs rows, `verify()` included.

## Spill under the admission budget

```python
import tempfile
from antecedent.state import SpillRegistry

with tempfile.TemporaryDirectory() as directory:
    registry = SpillRegistry(directory, 1 << 20)   # budget never exceeds the execution's pool
    spilled = refresh.spill(registry, "study")
    print(spilled.mode, spilled.on_disk, registry.stats["spills"])
    back = spilled.reload()                        # an independent branch with its rows
    assert back.result().values == refresh.result().values
```

A spilled result serves a read of its disk artifact and refuses everything else until it is
reloaded. `RetainedBootstrap` keeps bootstrap replicate estimates the same way and serves
`percentile_interval(level)` from the retained estimates.

## Publish a recomputed result into a `CausalState`

`CausalState` records what each data or premise event made stale. It never recomputes. A result is
published back only with the identities of the study that recomputed it:

```python
from antecedent.state import CausalState, publish_recomputed_results

state = CausalState(1 << 20)
state.append_data(["x"], [np.arange(8, dtype=np.float64)])
_, query = state.register_average_effect(0, 1)
print(state.stale_queries() == [query])

identities = dict(ant.estimate(study).identities)
publish_recomputed_results(state, state.version, [(query, identities, 32)])
print(state.stale_queries())                       # []

state.append_data(["x"], [np.arange(8, dtype=np.float64)])
branch = ant.rebind(study, data={"t": t, "z": z, "y": 4.0 * t + z})
fresh = dict(ant.estimate(branch).identities)
publish_recomputed_results(state, state.version, [(query, fresh, 32)])
```

Publication refuses, and leaves every freshness stamp as it was, a version that advanced during the
recompute, a data snapshot recorded before the latest data event, a premise event answered by the
identical lineage, an unknown query and a cache-budget overrun. A query published this way is
lineage-bound, so the unchecked `CausalState.refresh_results` refuses to restamp it.
`antecedent.state.result_lineage_fingerprint(identities)` gives the 64-hex fingerprint of an outcome's
identity layers if you need to compare lineages yourself.

Transported claims that rest on state queries recompute or withdraw from the same log; see
[assumption range and revalidation](assumption-range-and-revalidation.md).
