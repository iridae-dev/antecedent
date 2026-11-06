#!/usr/bin/env python3
"""Explain an anomaly and a change in distribution in a structural causal model.

Question: when a downstream measurement looks wrong, which upstream mechanism is the root
cause? Two cases: (a) one unusual observation, and (b) a shift of the whole distribution
between two periods.

Data. A chain ``x -> m -> y``: ``m = 0.8 x + shift + 0.5 e`` and ``y = 1.5 m + 0.5 e``, with
1,200 rows. In case (b) the mechanism of ``m`` changes between the two halves (``shift`` is 0,
then 3), so the declared change of ``y`` is ``1.5 * 3 = 4.5`` and ``m`` is the root cause. In
case (a) there is no shift, and one row has a shock of 6 added to ``m`` (12 noise standard
deviations) that then propagates to ``y``.

What is checked. The total change of ``y`` is compared with the difference of the two period
means and with 4.5. The mechanism that changed carries nearly all of it, ranks first in the
root-cause ranking. Mechanism-change detection is refused when m shifts (y's parent law moves)
and flags only y when y's own mechanism shifts. In case (a) the
anomaly score of each node is compared with the definition (median, 1.4826 times MAD, a
Gaussian two-sided tail); the shocked row is the top row for ``m`` and ``y``, and the node
whose score first leaves the ordinary range along the graph order is ``m``. Python exposes
anomaly scores per node and row, not a per-node Shapley split of one row, so the root cause
is read from where along the graph the row stops looking ordinary.

What this does NOT claim. The attributions assume the declared graph and mechanisms fitted
from the data; they are not tests of that graph. The bootstrap standard error is
uncalibrated and unsupported (see ``antecedent.attribution``), so it is a spread, not a
coverage claim.
"""

from __future__ import annotations

# %%
import math

import antecedent as ant
import numpy as np
from antecedent import attribution as attr
from antecedent.errors import CausalUnsupportedError
from antecedent.outcomes import require_success

N, HALF = 1200, 600
NAMES = ["x", "m", "y"]
EDGES = [("x", "m"), ("m", "y")]
SHIFT, SLOPE_MY = 3.0, 1.5
TRUE_CHANGE = SLOPE_MY * SHIFT
SHOCK = 6.0
EXEC = ant.Execution(seed=4)


def simulate(
    seed: int, *, shift: float = 0.0, leaf_shift: float = 0.0, shock_row: int | None = None
):
    """Chain data; ``shift`` moves m's mechanism in the second half, ``shock`` hits one row."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    level = np.where(np.arange(N) < HALF, 0.0, shift)
    noise_m = 0.5 * rng.normal(size=N)
    if shock_row is not None:
        noise_m[shock_row] += SHOCK
    m = 0.8 * x + level + noise_m
    y = SLOPE_MY * m + np.where(np.arange(N) < HALF, 0.0, leaf_shift) + 0.5 * rng.normal(size=N)
    return [x, m, y]


def it_scores(values: np.ndarray) -> np.ndarray:
    """-log P(|Z| >= |z|) at the robust reference: median and 1.4826 * MAD, written out here."""
    center = np.median(values)
    scale = 1.482602218505602 * np.median(np.abs(values - center))
    return np.array([-math.log(math.erfc(abs(v - center) / scale / math.sqrt(2))) for v in values])


# %% [markdown]
# ## A distribution change and the mechanism behind it
#
# The second half of the rows has a different mechanism for `m`. `attribute_distribution_change`
# splits the change of `y` between the two row ranges across the mechanisms of the graph
# (Shapley attribution). The contributions should add up to the total, and the total should be
# the difference of the period means, which here is also close to the declared 4.5.

# %%
changed = simulate(8, shift=SHIFT)
y = changed[2]
period_gap = float(y[HALF:].mean() - y[:HALF].mean())
change = attr.attribute_distribution_change(
    NAMES, changed, EDGES, "y", 0, HALF, HALF, N, n_samples=400, execution=EXEC
)
shares = dict(change.contributions)
print(f"total change {change.total_change:.3f}; period-mean gap {period_gap:.3f}; truth 4.5")
print({k: round(v, 3) for k, v in shares.items()})
assert abs(change.total_change - period_gap) < 0.1
assert abs(change.total_change - TRUE_CHANGE) < 0.3
assert abs(sum(shares.values()) - change.total_change) < 0.15
assert max(shares, key=shares.get) == "m"
assert shares["m"] / change.total_change > 0.9

# %% [markdown]
# ## Which mechanisms changed, and the root-cause ranking
#
# Mechanism-change detection tests every node's mechanism between the periods, with a
# multiplicity correction. Its residual tests are not calibrated when a node's parents are
# distributed differently in the two periods, so here (m shifts, and m is y's parent) it is
# refused. `attribute_root_causes` ranks the attribution, largest first. The test is valid
# when only a leaf changes: a second table shifts y's own mechanism and flags only y.

# %%
try:
    attr.mechanism_change_detection(NAMES, changed, EDGES, 0, HALF, HALF, N, execution=EXEC)
except ant.CausalError as error:
    print("refused:", getattr(error, "reason_code", None) or type(error).__name__, "-", error)
else:
    raise AssertionError("a test under a parent-covariate shift was accepted")
ranks = attr.attribute_root_causes(
    NAMES, changed, EDGES, "y", 0, HALF, HALF, N, n_samples=400, execution=EXEC
)
print("ranking:", [(r.name, round(r.score, 3)) for r in ranks])
assert ranks[0].name == "m"
assert [r.score for r in ranks] == sorted((r.score for r in ranks), reverse=True)

# When only a leaf's mechanism changes, no parent law moves and the test is valid.
leaf = simulate(8, leaf_shift=SHIFT)
tests = attr.mechanism_change_detection(NAMES, leaf, EDGES, 0, HALF, HALF, N, execution=EXEC)
for t in tests:
    print(f"{t.node}: stat {t.statistic:+.3f}  adj. p {t.adjusted_p_value:.2g}  {t.changed}")
assert {t.node for t in tests if t.changed} == {"y"}

# %% [markdown]
# ## The same question as a query, and an uncertainty readout
#
# `ChangeAttribution` runs the attribution through `ant.analyze` on an explicit graph and
# returns a result with identification and calibration records. A bootstrap over refits of
# every mechanism gives the spread of the total; the module documents that spread as
# uncalibrated, and the script prints the calibration status the result itself reports.

# %%
dag = ant.Dag.from_edges(NAMES, EDGES)
table = dict(zip(NAMES, changed, strict=True))
query = ant.ChangeAttribution(
    "y", baseline_start=0, baseline_end=HALF, comparison_start=HALF, comparison_end=N
)
result = require_success(ant.analyze(table, graph=dag, query=query))
print("query total:", round(result.change_attribution.total_change, 3))
print("calibration status:", result.calibration.status)
assert abs(result.change_attribution.total_change - change.total_change) < 0.1
top = max(result.change_attribution.contributions, key=lambda c: c[1])
assert top[0] == "m"

boot = attr.attribute_distribution_change(
    NAMES, changed, EDGES, "y", 0, HALF, HALF, N,
    n_samples=200, execution=EXEC, bootstrap_replicates=20,
)  # fmt: skip
se = boot.uncertainty.bootstrap.se["total_change"]
print(f"bootstrap SE of the total: {se:.3f} over {boot.uncertainty.bootstrap.replicates_ok} refits")
assert se is not None and 0.0 < se < 0.5

# %% [markdown]
# ## An anomalous observation
#
# Now there is no shift, but row 300 has a shock added to `m` that flows on to `y`. The anomaly
# score of a node is the negative log of a two-sided tail probability at a robust reference, so
# it is large for any node whose value in that row is far out. The shocked row tops `m` and
# `y`; `x` is untouched. Following the graph order, the first node that leaves the ordinary
# range is the root cause.

# %%
ROW = 300
anomalous = simulate(9, shock_row=ROW)
scores = {s.outcome: s for s in attr.anomaly_attribution(NAMES, anomalous, EDGES, NAMES)}
for name, column in zip(NAMES, anomalous, strict=True):
    reference = it_scores(column)
    np.testing.assert_allclose(scores[name].scores, reference, rtol=0, atol=1e-6)
at_row = {n: scores[n].scores[scores[n].rows.index(ROW)] for n in NAMES}
print("score at the shocked row:", {k: round(v, 1) for k, v in at_row.items()})
print("top row for m and y:", scores["m"].top_row, scores["y"].top_row)
assert scores["m"].top_row == ROW and scores["y"].top_row == ROW
THRESHOLD = 10.0  # well above the largest score of an ordinary row of 1,200 (about 7)
ordinary = [s for n in NAMES for i, s in zip(scores[n].rows, scores[n].scores) if i != ROW]
assert max(ordinary) < THRESHOLD
root = next(name for name in NAMES if at_row[name] > THRESHOLD)
print("first node out of range along the graph:", root)
assert root == "m"

query_a = ant.AnomalyAttribution(NAMES, max_units=N)
found = require_success(
    ant.analyze(dict(zip(NAMES, anomalous, strict=True)), graph=dag, query=query_a)
)
assert [s.top_row for s in found.anomaly if s.outcome in {"m", "y"}] == [ROW, ROW]
print("AnomalyAttribution query agrees; calibration:", found.calibration.status)

# %% [markdown]
# ## A refusal
#
# Cheap and full validation suites are not defined for a change attribution: there is no
# detectable violation for them to probe. The request is refused with a typed error instead
# of returning an unvalidated answer.

# %%
try:
    require_success(ant.analyze(table, graph=dag, query=query, refute="cheap"))
except CausalUnsupportedError as error:
    assert "Cheap and full validation of a change attribution is refused" in str(error)
    print("refused:", getattr(error, "reason_code", None) or type(error).__name__, "-", error)
else:
    raise AssertionError("validation of a change attribution was accepted")
