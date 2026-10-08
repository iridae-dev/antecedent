"""Recompute a decision without refitting: frozen cell scores, and resuming from their bytes.

``recalc_cell.CellSession`` runs a cell-saturated AIPW interaction contrast over two binary
treatments and keeps the per-unit scores. Each change is then planned and receipted stage by stage:
a new utility recomputes only the decision, a compatible target-population change reweights the
frozen scores, and neither refits (the fit count comes from the estimator's own instrument).
``export_frozen_scores`` writes the scores, with no data and no model, as a checksummed artifact
that ``resume_from_scores`` reloads in any interpreter; the one licensed operation there is a
retarget, and anything needing the data or a fit is refused before any work."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from antecedent import recalc_cell
from antecedent.recalc import Stage, TargetWeights, Utility

N = 1200
SEED = 72
FIRST_RUN_FITS = 25  # 5 folds x (1 multinomial propensity + 4 cell outcome regressions)
VARIABLES = ("a", "d", "y", "z", "w", "y2")
EDGES = (
    ("z", "a"),
    ("z", "d"),
    ("z", "y"),
    ("a", "y"),
    ("d", "y"),
    ("z", "y2"),
    ("a", "y2"),
    ("d", "y2"),
)
UTILITY = Utility(2.0, 0.5)  # net benefit = 2 * interaction - 0.5

# Two confounded binary treatments with an interaction effect of 1.5, an unrelated column ``w``
# and a second outcome ``y2`` (the draw order is the one the estimator's tests are checked on).
rng = np.random.default_rng(SEED)
z = rng.standard_normal(N)
w = rng.standard_normal(N)
a = (rng.random(N) < 1.0 / (1.0 + np.exp(-0.5 * z))).astype(np.float64)
d = (rng.random(N) < 0.5).astype(np.float64)
y = 1.5 * a * d + 0.5 * a + 0.2 * z + 0.25 * rng.standard_normal(N)
y2 = -a + 0.5 * d + 0.3 * z + 0.25 * rng.standard_normal(N)

request = recalc_cell.CellRequest(
    data={"a": a, "d": d, "y": y, "z": z, "w": w, "y2": y2},
    edges=EDGES,
    treatments=("a", "d"),
    outcome="y",
    utility=UTILITY,
    adjustment=("z",),
)

session = recalc_cell.CellSession()
first = session.execute(request, seed=SEED)
print(first.receipt.explain())
assert first.receipt.totals.fold_fits == FIRST_RUN_FITS
assert abs(first.law.ate - 1.5) < 0.5
assert np.isclose(first.decision.net_benefit, 2.0 * first.law.ate - 0.5)

# A new utility recomputes the decision only: zero fits, the law is untouched.
priced = session.execute(replace(request, utility=Utility(3.0, 0.1)), seed=SEED)
assert priced.receipt.totals.fold_fits == 0
assert priced.law == first.law
assert np.isclose(priced.decision.net_benefit, 3.0 * first.law.ate - 0.1)

# A new target population reweights the frozen scores: still zero fits.
weights = TargetWeights(np.exp(0.4 * z), ("z",))
retargeted = session.execute(replace(request, target=weights), seed=SEED)
assert retargeted.receipt.totals.fold_fits == 0
assert abs(retargeted.law.ate - first.law.ate) > 1e-3

# Export the frozen scores, retain their identity, and resume from the bytes alone.
frozen = session.export_frozen_scores()
resumed = recalc_cell.resume_from_scores(
    frozen.export(),
    variables=VARIABLES,
    edges=EDGES,
    utility=UTILITY,
    expected_identity=frozen.identity,
)
again = resumed.retarget(weights, row_ids=resumed.row_ids)
print(again.receipt.explain())
assert again.receipt.totals.fold_fits == 0
assert np.isclose(again.law.ate, retargeted.law.ate, atol=1e-10)

# Anything that needs the data or a fit is refused before any work; the session is unchanged.
try:
    resumed.retarget(declared_changes={Stage.QUERY: "outcome=y2"})
except recalc_cell.ScoreResumeUnavailable as refusal:
    assert refusal.detail == "recalc.unavailable_data" and refusal.missing == "data"
else:
    raise AssertionError("a changed outcome needs the data and must be refused")
