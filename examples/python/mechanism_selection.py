"""Choosing a causal mechanism among linear, spline and tree-ensemble families.

``y = 1 + 0.5 x + 0.8 x^2 + noise`` is not linear in ``x``. Every candidate family is scored on the
same held-out folds and the best is kept, together with each candidate's score, the families that
could not be fit and why, and the rule that chose. A per-node constraint can require a family
whatever its rank (the others are still scored), a contradictory constraint is refused before any
fit, and rows that are dependent in time say so with a split scheme."""

from __future__ import annotations

import antecedent
import numpy as np

CANDIDATES = ["linear", "additive_spline", "random_forest", "gradient_boosted_trees"]

rng = np.random.default_rng(11)
x = rng.normal(size=600)
y = 1.0 + 0.5 * x + 0.8 * x**2 + 0.3 * rng.normal(size=600)
EDGES = [("x", "y")]


def record(fitted, variable):
    return next(r for r in fitted.mechanism_selection() if r["variable"] == variable)


def score(rec, family):
    return next(c["score"] for c in rec["candidates"] if c["family"] == family)


# 1. Best score wins on shared folds, and the choice is auditable.
fitted = antecedent.model.fit_gcm(["x", "y"], [x, y], EDGES, candidates=CANDIDATES)
chosen = record(fitted, "y")
assert chosen["selected"] != "linear_gaussian", "a quadratic is not linear"
assert chosen["rule"] == "best_score" and chosen["splits"] == "shuffled"
assert chosen["n_rows"] == 600 and chosen["seed"] is not None
assert score(chosen, chosen["selected"]) > score(chosen, "linear_gaussian")
for candidate in chosen["candidates"]:
    held_out = candidate["held_out_log_likelihood"]
    assert np.isclose(candidate["score"], held_out - candidate["complexity_penalty"])
assert (
    chosen["noise_inference"] == "invertible"
    and chosen["attribution_compatible"] is True
)

# 2. A root has no parents: tree families fail there and the failure is recorded, not dropped.
root = record(fitted, "x")
failed = {f["family"]: f["reason"] for f in root["failed"]}
assert "at least one parent" in failed["tree_boosted_additive"]

# 3. Requiring a family selects it whatever its rank, and the other candidates are still scored.
forced = antecedent.model.fit_gcm(
    ["x", "y"],
    [x, y],
    EDGES,
    candidates=CANDIDATES,
    constraints={"y": {"required": "linear_gaussian"}},
)
required = record(forced, "y")
assert required["selected"] == "linear_gaussian"
assert required["rule"] == "required_by_node_constraint"
assert len(required["candidates"]) > 1

# 4. A contradictory constraint is refused before any family is fit.
try:
    antecedent.model.fit_gcm(
        ["x", "y"],
        [x, y],
        EDGES,
        constraints={
            "y": {"required": "linear_gaussian", "excluded": ["linear_gaussian"]}
        },
    )
except Exception as refusal:  # noqa: BLE001 - the native refusal's class is not the point
    assert "mechanism_candidates_invalid" in str(refusal), refusal
else:
    raise AssertionError("a required and excluded family must be refused")

# 5. Serially dependent rows are scored by forward chaining, so no validation row precedes a
#    training row; the scheme and its purge are recorded with the selection.
walk = np.cumsum(rng.normal(size=300))
response = np.sin(walk / 5.0) + 0.1 * rng.normal(size=300)
ordered = antecedent.model.fit_gcm(
    ["w", "r"],
    [walk, response],
    [("w", "r")],
    candidates=CANDIDATES,
    splits="time_ordered",
    purge=5,
)
timed = record(ordered, "r")
assert (
    timed["splits"] == "time_ordered" and timed["purge"] == 5 and timed["seed"] is None
)
print(
    f"y: {chosen['selected']} (score {score(chosen, chosen['selected']):.3f}); x: {root['selected']}"
)
