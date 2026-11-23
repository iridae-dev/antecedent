"""Candidate sets, per-node constraints, split schemes and selection provenance of ``fit_gcm``."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest

TREE_CANDIDATES = ["linear", "additive_spline", "random_forest", "gradient_boosted_trees"]


def _threshold_chain(n: int = 1500, seed: int = 11):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    y = 2.0 * (x > 0.3) + rng.normal(scale=0.3, size=n)
    return ["x", "y"], [x, y], [("x", "y")]


def _record(fitted, variable: str) -> dict:
    return next(r for r in fitted.mechanism_selection() if r["variable"] == variable)


def _score(record: dict, family: str) -> float:
    return next(c["score"] for c in record["candidates"] if c["family"] == family)


def test_default_fit_records_shuffled_splits_and_no_constraint():
    names, cols, edges = _threshold_chain(n=300)
    fitted = antecedent.model.fit_gcm(names, cols, edges)
    record = _record(fitted, "y")
    assert record["splits"] == "shuffled"
    assert record["constraint"] is None
    assert record["rule"] == "best_score"
    assert record["seed"] is not None
    assert record["n_rows"] == 300
    assert record["candidate_set"] == ["linear_gaussian", "constant"]


def test_tree_candidates_win_threshold_truth_and_scores_are_comparable():
    names, cols, edges = _threshold_chain()
    fitted = antecedent.model.fit_gcm(names, cols, edges, candidates=TREE_CANDIDATES)
    record = _record(fitted, "y")
    assert record["selected"].startswith("tree_")
    assert record["noise_inference"] == "invertible"
    assert record["attribution_compatible"] is True
    families = {c["family"] for c in record["candidates"]}
    assert {"linear_gaussian", "tree_forest_additive", "tree_boosted_additive"} <= families
    best = max(record["candidates"], key=lambda c: c["score"])
    assert best["family"] == record["selected"]
    for c in record["candidates"]:
        assert c["score"] == pytest.approx(c["held_out_log_likelihood"] - c["complexity_penalty"])
    assert _score(record, record["selected"]) > _score(record, "linear_gaussian")


def test_per_node_constraint_restricts_only_the_constrained_node():
    rng = np.random.default_rng(5)
    n = 600
    x = rng.uniform(-1.0, 1.0, size=n)
    m = x**2 + rng.normal(scale=0.2, size=n)
    y = np.sin(2.0 * m) + rng.normal(scale=0.2, size=n)
    fitted = antecedent.model.fit_gcm(
        ["x", "m", "y"],
        [x, m, y],
        [("x", "m"), ("m", "y")],
        candidates=TREE_CANDIDATES,
        constraints={"y": {"allowed": ["linear", "additive_spline"]}},
    )
    constrained = _record(fitted, "y")
    assert constrained["candidate_set"] == ["linear_gaussian", "linear_spline"]
    # Aliases name both outcome types; the resolved candidate set keeps the node's type only.
    assert constrained["constraint"]["allowed"] == [
        "linear_gaussian",
        "discrete",
        "linear_spline",
        "discrete_spline",
    ]
    assert not any(c["family"].startswith("tree_") for c in constrained["candidates"])
    free = _record(fitted, "m")
    assert any(c["family"].startswith("tree_") for c in free["candidates"])
    assert free["constraint"] is None


def test_required_family_is_selected_whatever_its_rank():
    names, cols, edges = _threshold_chain(n=600)
    fitted = antecedent.model.fit_gcm(
        names,
        cols,
        edges,
        candidates=TREE_CANDIDATES,
        constraints={"y": {"required": "linear_gaussian"}},
    )
    record = _record(fitted, "y")
    assert record["selected"] == "linear_gaussian"
    assert record["rule"] == "required_by_node_constraint"
    # The other candidates were still scored, so the choice stays auditable.
    assert len(record["candidates"]) > 1


def test_excluded_family_is_not_scored():
    names, cols, edges = _threshold_chain(n=600)
    fitted = antecedent.model.fit_gcm(
        names,
        cols,
        edges,
        candidates=TREE_CANDIDATES,
        constraints={"y": {"excluded": ["random_forest"]}},
    )
    record = _record(fitted, "y")
    assert "tree_forest_additive" not in record["candidate_set"]


def test_contradictory_constraint_is_refused_with_its_reason_code():
    names, cols, edges = _threshold_chain(n=300)
    with pytest.raises(Exception, match="mechanism_candidates_invalid"):
        antecedent.model.fit_gcm(
            names,
            cols,
            edges,
            constraints={"y": {"required": "linear_gaussian", "excluded": ["linear_gaussian"]}},
        )
    with pytest.raises(Exception, match="mechanism_candidates_invalid"):
        antecedent.model.fit_gcm(names, cols, edges, candidates=["not_a_family"])


def test_an_unknown_or_empty_candidate_is_refused_with_its_reason_code():
    names, cols, edges = _threshold_chain(n=300)
    # Every constraint field resolves its tokens through the same candidate vocabulary.
    for constraint in (
        {"allowed": ["not_a_family"]},
        {"excluded": ["not_a_family"]},
        {"required": "not_a_family"},
    ):
        with pytest.raises(Exception, match="mechanism_candidates_invalid"):
            antecedent.model.fit_gcm(names, cols, edges, constraints={"y": constraint})
    with pytest.raises(Exception, match="mechanism_candidates_invalid"):
        antecedent.model.fit_gcm(names, cols, edges, candidates=[])
    # A required alias that names two families is not one family.
    with pytest.raises(Exception, match="one family"):
        antecedent.model.fit_gcm(names, cols, edges, constraints={"y": {"required": "linear"}})


def test_unknown_constraint_node_or_key_is_rejected():
    names, cols, edges = _threshold_chain(n=300)
    with pytest.raises(Exception, match="nope"):
        antecedent.model.fit_gcm(names, cols, edges, constraints={"nope": {"allowed": ["linear"]}})
    with pytest.raises(Exception, match="unknown key"):
        antecedent.model.fit_gcm(names, cols, edges, constraints={"y": {"prefer": ["linear"]}})


def test_time_ordered_splits_are_recorded_and_deterministic():
    rng = np.random.default_rng(2)
    n = 900
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal()
    y = np.sin(x) + rng.normal(scale=0.3, size=n)
    first = antecedent.model.fit_gcm(
        ["x", "y"], [x, y], [("x", "y")], candidates=TREE_CANDIDATES, splits="time_ordered", purge=5
    )
    second = antecedent.model.fit_gcm(
        ["x", "y"], [x, y], [("x", "y")], candidates=TREE_CANDIDATES, splits="time_ordered", purge=5
    )
    record = _record(first, "y")
    assert record["splits"] == "time_ordered"
    assert record["purge"] == 5
    assert record["seed"] is None
    assert record["folds"] == 5
    assert _record(second, "y")["candidates"] == record["candidates"]


def test_purge_without_time_ordered_splits_is_rejected():
    names, cols, edges = _threshold_chain(n=300)
    with pytest.raises(Exception, match="purge"):
        antecedent.model.fit_gcm(names, cols, edges, purge=3)
    with pytest.raises(Exception, match="unknown splits"):
        antecedent.model.fit_gcm(names, cols, edges, splits="blocked")


def test_binary_node_can_select_a_tree_classifier():
    rng = np.random.default_rng(9)
    n = 2000
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(3.0 * x**2 - 3.0)))
    t = (rng.random(n) < p).astype(np.float64)
    fitted = antecedent.model.fit_gcm(
        ["x", "t"],
        [x, t],
        [("x", "t")],
        candidates=["logistic", "tree_ensemble_classifier"],
    )
    record = _record(fitted, "t")
    assert record["noise_inference"] == "posterior"
    assert record["selected"].startswith("tree_") and record["selected"].endswith("categorical")
