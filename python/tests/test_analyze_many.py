"""Batch multi-query helper (BACKLOG E)."""

from __future__ import annotations

import math
import random

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError


def _two_treatment_scm(n: int = 500, seed: int = 9):
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t1 = np.empty(n, dtype=np.float64)
    t2 = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p1 = 1.0 / (1.0 + math.exp(-(-0.3 + 0.8 * zi)))
        p2 = 1.0 / (1.0 + math.exp(-(-0.2 + 0.7 * zi)))
        a = 1.0 if rng.random() < p1 else 0.0
        b = 1.0 if rng.random() < p2 else 0.0
        z[i] = zi
        t1[i] = a
        t2[i] = b
        y[i] = 2.0 * a + 1.5 * b + zi + 0.4 * rng.gauss(0.0, 1.0)
    data = {"t1": t1, "t2": t2, "y": y, "z": z}
    edges = [("z", "t1"), ("z", "t2"), ("z", "y"), ("t1", "y"), ("t2", "y")]
    return data, edges


def test_analyze_many_matches_solo():
    data, edges = _two_treatment_scm()
    q1 = antecedent.AverageEffect(treatment="t1", outcome="y")
    q2 = antecedent.AverageEffect(treatment="t2", outcome="y")
    batch = antecedent.estimation.analyze_many(
        data,
        graph=edges,
        queries=[q1, q2],
        refute=False,
        bootstrap=0,
        seed=3,
    )
    assert len(batch) == 2
    solo1 = antecedent.analyze(data, graph=edges, query=q1, refute=False, bootstrap=0, seed=3)
    solo2 = antecedent.analyze(data, graph=edges, query=q2, refute=False, bootstrap=0, seed=3)
    assert abs(batch[0].ate - solo1.ate) < 1e-12
    assert abs(batch[1].ate - solo2.ate) < 1e-12
    assert abs(batch[0].ate - 2.0) < 0.45
    assert abs(batch[1].ate - 1.5) < 0.45


def _shared_design_note(result) -> str:
    notes = [d for d in result.diagnostics if d.startswith("batch.shared_design:")]
    assert len(notes) == 1, result.diagnostics
    return notes[0]


def test_analyze_many_shares_identical_nuisance_fits_bit_for_bit():
    data, edges = _two_treatment_scm()
    data = dict(data)
    data["y2"] = np.asarray(data["y"]) * 0.5 - np.asarray(data["z"])
    edges = [*edges, ("z", "y2"), ("t1", "y2")]
    # (t1, y) and (t1, y2) share the t1 propensity; (t2, y) shares nothing.
    queries = [
        antecedent.AverageEffect(treatment="t1", outcome="y"),
        antecedent.AverageEffect(treatment="t1", outcome="y2"),
        antecedent.AverageEffect(treatment="t2", outcome="y"),
    ]
    common = dict(estimator="aipw", refute=False, bootstrap=0, seed=5)
    batch = antecedent.estimation.analyze_many(data, graph=edges, queries=queries, **common)
    for result, query in zip(batch, queries, strict=True):
        solo = antecedent.analyze(data, graph=edges, query=query, **common)
        assert result.ate == solo.ate
    assert "propensity shared" in _shared_design_note(batch[0])
    assert "propensity shared" in _shared_design_note(batch[1])
    assert "propensity not shared" in _shared_design_note(batch[2])
    assert "outcome regressions not shared" in _shared_design_note(batch[0])


def test_analyze_many_refuses_other_queries_naming_analyze():
    data, edges = _two_treatment_scm()
    ate = antecedent.AverageEffect(treatment="t1", outcome="y")
    others = [
        antecedent.InterventionResponse(
            "y",
            intervention=[
                antecedent.intervention.Set("t1", 1.0),
                antecedent.intervention.Set("t2", 1.0),
            ],
        ),
        antecedent.ConditionalEffect("t1", "y", "z"),
    ]
    for other in others:
        with pytest.raises(CausalUnsupportedError) as caught:
            antecedent.estimation.analyze_many(data, graph=edges, queries=[ate, other])
        assert caught.value.reason_code == "route_not_supported"
        assert "analyze" in str(caught.value)
        assert "antecedent.analyze" in caught.value.remedy
        with pytest.raises(CausalUnsupportedError) as prepared:
            antecedent.estimation.PreparedBatch.prepare(data, graph=edges, queries=[ate, other])
        assert prepared.value.reason_code == "route_not_supported"
        assert "antecedent.analyze" in prepared.value.remedy


def test_prepare_cells_refuses_non_joint_queries_naming_analyze():
    data, edges = _two_treatment_scm()
    joint = antecedent.InterventionResponse(
        "y",
        intervention=[
            antecedent.intervention.Set("t1", 1.0),
            antecedent.intervention.Set("t2", 1.0),
        ],
    )
    single = antecedent.InterventionResponse(
        "y", intervention=antecedent.intervention.Set("t1", 1.0)
    )
    for other in (single, antecedent.AverageEffect(treatment="t1", outcome="y")):
        with pytest.raises(CausalUnsupportedError) as caught:
            antecedent.estimation.PreparedBatch.prepare_cells(
                data, graph=edges, queries=[joint, other]
            )
        assert caught.value.reason_code == "route_not_supported"
        assert "antecedent.analyze" in caught.value.remedy
