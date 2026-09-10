"""Structured CausalReviewError attrs and TemporalPag completion."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _lag1_series(n: int = 120, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.9 * x[t - 1] + 0.05 * rng.normal()
    return {"x": x, "y": y}


def test_fci_review_required_attrs():
    n = 80
    rng = np.random.default_rng(11)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n) * 0.3
    y = 1.2 * t + z + rng.normal(size=n) * 0.3
    with pytest.raises(antecedent.errors.CausalReviewError) as ei:
        antecedent.analyze(
            {"t": t, "y": y, "z": z},
            discovery=antecedent.discovery.FCI(alpha=0.2, fdr=False, max_cond_size=2),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            accept_discovered=False,
            refute=False,
            bootstrap=0,
            seed=1,
        )
    err = ei.value
    assert getattr(err, "kind", None) == "static_pag"
    assert getattr(err, "algorithm", None) == "fci"
    assert isinstance(getattr(err, "pending_edge_count", None), int)
    # Pin the hint content (mirrors PagReview::into_accepted's fixed message in
    # crates/antecedent/src/accepted.rs), not just its truthiness — a non-empty
    # string is not an actionable hint.
    hint = getattr(err, "hint", None)
    assert hint
    assert "circle" in hint
    assert "generalized adjustment" in hint


@pytest.mark.parametrize("endpoint", ["tail", "circle"])
def test_temporal_pag_does_not_bypass_mag_visibility(endpoint):
    data = _lag1_series()
    pag = antecedent.graph.TemporalPag.from_marked_lagged_edges(
        ["x", "y"], [("x", 1, "y", 0, endpoint, "arrow")]
    )
    query = antecedent.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1)
    identified = antecedent.identify(graph=pag, query=query)
    assert identified.status == "NotIdentified"
    assert identified.certificate["graph_class"] == "TemporalPag"
    with pytest.raises(antecedent.errors.CausalCompileError, match="no identified mass"):
        antecedent.analyze(data, graph=pag, query=query, bootstrap=0, seed=1, refute=False)


def test_review_required_is_still_a_causal_review_error():
    """`except CausalReviewError` must keep catching `ReviewRequired` after the
    alias -> real-subclass change (P5a)."""
    assert issubclass(antecedent.errors.ReviewRequired, antecedent.errors.CausalReviewError)
    with pytest.raises(antecedent.errors.CausalReviewError):
        raise antecedent.errors.ReviewRequired("boom")


def test_review_required_reexport_catches_too():
    """`except antecedent.ReviewRequired` must catch the same error the native layer
    (and `build_review_error`) raise."""
    with pytest.raises(antecedent.ReviewRequired):
        raise antecedent.errors.build_review_error(
            "boom",
            kind="generic",
            algorithm=None,
            pending_edge_count=0,
            hint="n/a",
        )


def test_build_review_error_round_trips_pending_edges():
    edge = antecedent.errors.PendingEdge(
        source="V0", target="V1", at_source="tail", at_target="arrow"
    )
    err = antecedent.errors.build_review_error(
        "cannot execute while graph review is required",
        kind="static_cpdag",
        algorithm="pc",
        pending_edge_count=1,
        hint="orient the remaining edge",
        pending_edges=[edge],
    )
    assert isinstance(err, antecedent.errors.ReviewRequired)
    assert isinstance(err, antecedent.errors.CausalReviewError)
    assert str(err) == "cannot execute while graph review is required"
    assert err.message == str(err)
    got = antecedent.errors.pending_edges(err)
    assert got == (edge,)
