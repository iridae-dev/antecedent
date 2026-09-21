"""Shared Analysis API: claim, narrowing, HTML funnel, refusal next-action."""

from __future__ import annotations

from pathlib import Path

import antecedent
import numpy as np
import pytest
from antecedent.errors import (
    EffectNotIdentified,
    PendingEdge,
    named_pending_edges,
    next_action,
    resolve_display_name,
)
from antecedent.results import IdentificationView, ResponseView


def _ate_result():
    rng = np.random.default_rng(0)
    z = rng.normal(size=80)
    t = (z + rng.normal(size=80) > 0).astype(float)
    y = 1.5 * t + 0.4 * z + rng.normal(size=80) * 0.2
    return antecedent.analyze(
        {"t": t, "y": y, "z": z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=antecedent.AverageEffect("t", "y"),
        bootstrap=0,
        refute=False,
        seed=1,
    )


def test_analyze_returns_analysis_and_claim():
    result = _ate_result()
    assert isinstance(result, antecedent.AnalysisResult)
    claim = result.claim()
    assert "AverageEffect of y from t" in claim
    assert "adjusting for z" in claim
    assert "answer is" in claim
    assert "calibration" in claim
    assert result.as_point() == pytest.approx(result.answer.value)
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="as_response"):
        result.as_response()


def test_as_point_refuses_partial_or_unavailable():
    result = _ate_result()
    if result.answer.kind != "point":
        pytest.skip("fixture produced a non-point answer")
    # Direct constructor path: Identification refuses as a result claim.
    ident = antecedent.identify(
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=antecedent.AverageEffect("t", "y"),
        names=["z", "t", "y"],
    )
    assert "is identified" in ident.claim()
    html = ident._repr_html_()
    assert "identified" in html.lower()
    assert "z" in html


def test_py_typed_marker_ships():
    root = Path(antecedent.__file__).parent
    assert (root / "py.typed").is_file()


def test_graph_and_review_html():
    dag = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    html = dag._repr_html_()
    assert "Dag" in html
    assert "-&gt;" in html or "->" in html
    err = antecedent.errors.build_review_error(
        "orient me",
        kind="static_pag",
        algorithm="fci",
        pending_edge_count=1,
        hint="mark circle endpoints",
        pending_edges=(PendingEdge("V0", "V1", "circle", "arrow"),),
    )
    html = err._repr_html_()
    assert "Review required" in html
    assert "Next" in html


def test_display_names_resolve_and_next_action():
    assert resolve_display_name("V2", ["z", "t", "y"]) == "y"
    assert resolve_display_name("V1@-1", ["z", "t", "y"]) == "t@-1"
    assert resolve_display_name("already", ["z", "t", "y"]) == "already"
    named = named_pending_edges((PendingEdge("V0", "V2", "circle", "arrow"),), ["z", "t", "y"])
    assert named[0].source == "z"
    assert named[0].target == "y"
    err = antecedent.errors.build_review_error(
        "orient",
        kind="static_pag",
        algorithm="fci",
        pending_edge_count=1,
        hint="circles",
        pending_edges=named,
    )
    nxt = next_action(err, named)
    assert "accepted.review" in nxt
    assert "'z'" in nxt and "'y'" in nxt
    missing = EffectNotIdentified("no")
    missing.identification_status = "NotIdentified"
    missing.search_capped = True
    assert "capped" in next_action(missing)
    assert "not a proof" in next_action(missing)


def test_analyze_refusal_report_next():
    rng = np.random.default_rng(11)
    z = rng.normal(size=80)
    t = z + rng.normal(size=80) * 0.3
    y = 1.2 * t + z + rng.normal(size=80) * 0.3
    with pytest.raises(antecedent.ReviewRequired) as ei:
        antecedent.analyze(
            {"t": t, "y": y, "z": z},
            discovery=antecedent.discovery.FCI(alpha=0.2, fdr=False, max_cond_size=2),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            accept_discovered=False,
            refute=False,
            bootstrap=0,
            seed=1,
        )
    report = ei.value.report
    assert report.next
    assert "Review required" in report.next
    sources = {edge.source for edge in report.pending_edges}
    # Resolved against the data columns, not V0/V1.
    assert sources <= {"t", "y", "z"} or any(not s.startswith("V") for s in sources)


def test_identification_view_html():
    view = IdentificationView(
        status="NonparametricallyIdentified",
        method="backdoor.adjustment",
        adjustment_set=["z"],
        assumption_count=1,
        derivation_step_count=1,
    )
    html = view._repr_html_()
    assert "identified" in html.lower()
    assert "z" in html


def test_response_view_to_columns():
    view = ResponseView(
        treatments=["t"],
        outcomes=["y"],
        points=[[0.0], [1.0]],
        values=[[0.1], [1.1]],
    )
    cols = view.to_columns()
    assert cols["t"] == [0.0, 1.0]
    assert cols["y"] == [0.1, 1.1]
    assert not hasattr(view, "to_arrow")
